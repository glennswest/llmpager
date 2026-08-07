//! The decode loop: one token through the resident core with expert FFNs
//! pulled through the pager.
//!
//! Per layer: rmsnorm -> q/k/v projections -> per-head q/k norm -> RoPE ->
//! KV append -> GQA attention -> o projection (+residual) -> rmsnorm ->
//! router logits (top-k on host) -> paged expert FFNs, weighted-accumulated
//! (+residual). Then final norm -> lm_head -> greedy argmax on host.
//!
//! Correctness over speed (M2): the stream is synced once per layer so
//! expert handles can be released safely, and router/sampling round-trip to
//! the host. M3 moves release to events and overlaps prefetch.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};
use llmpager_core::pack::PackReader;
use llmpager_core::quant::q4g64_bytes;
use llmpager_cuda::driver::{CUdeviceptr, CUstream, Cuda};
use llmpager_cuda::kernels::Kernels;
use llmpager_cuda::pager::{Pager, PagerConfig};

use crate::model::{Config, CoreWeights, Mat};

/// GEMV through whichever encoding the matrix carries.
fn mat_gemv(
    ke: &Kernels,
    cu: &Cuda,
    m: &Mat,
    x: CUdeviceptr,
    y: CUdeviceptr,
    st: CUstream,
) -> Result<()> {
    if m.q4 {
        ke.q4g64_gemv(cu, m.dev, x, y, m.rows, m.cols, st)
    } else {
        ke.bf16_gemv(cu, m.dev, x, y, m.rows, m.cols, st)
    }
}

pub struct Decoder {
    cuda: Arc<Cuda>,
    kernels: Kernels,
    stream: CUstream,
    pub cfg: Config,
    core: CoreWeights,
    pager: Pager,
    max_seq: usize,
    // Device buffers.
    h: CUdeviceptr,
    h_norm: CUdeviceptr,
    q: CUdeviceptr,
    k: CUdeviceptr,
    v: CUdeviceptr,
    attn_out: CUdeviceptr,
    proj_out: CUdeviceptr,
    router_logits: CUdeviceptr,
    gate_out: CUdeviceptr,
    up_out: CUdeviceptr,
    act_out: CUdeviceptr,
    expert_out: CUdeviceptr,
    moe_acc: CUdeviceptr,
    logits: CUdeviceptr,
    attn_scratch: CUdeviceptr,
    kcache: Vec<CUdeviceptr>,
    vcache: Vec<CUdeviceptr>,
    // Blob region offsets (bytes) within an expert blob.
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    logits_host: Vec<u8>,
    router_host: Vec<u8>,
    // Deferred expert-handle release: handles stay pinned until the compute
    // stream passes the recorded event, so no per-layer sync is needed.
    release_events: Vec<llmpager_cuda::driver::CUevent>,
    pending_release: std::collections::VecDeque<(usize, Vec<llmpager_cuda::pager::ExpertHandle>)>,
    next_event: usize,
}

impl Decoder {
    pub fn new(
        pack_path: &Path,
        core_path: &Path,
        slots: u32,
        io_threads: usize,
        max_seq: usize,
        core_q4: bool,
    ) -> Result<Self> {
        let meta = PackReader::open(pack_path)?.meta().clone();
        let cfg = Config::from_json(&meta.config)?;
        if meta.dtype != "q4g64-gud" {
            bail!("pack dtype {} unsupported (want q4g64-gud)", meta.dtype);
        }

        let cuda = Arc::new(Cuda::init()?);
        let kernels = Kernels::load(&cuda)?;
        let stream = cuda.stream()?;
        eprintln!(
            "loading core ({} layers, hidden {}, {} experts top-{}) ...",
            cfg.layers, cfg.hidden, cfg.experts, cfg.top_k
        );
        let core = CoreWeights::load(&cuda, core_path, &cfg, core_q4, stream)?;
        let pager = Pager::new(
            Arc::clone(&cuda),
            pack_path,
            PagerConfig { slots_per_layer: slots, io_threads, decay_interval: 64.max(slots * 4) },
        )?;

        let f = |n: usize| cuda.alloc_device(n * 4);
        let qkv = cfg.heads * cfg.head_dim;
        let kvd = cfg.kv_heads * cfg.head_dim;
        let mut kcache = Vec::with_capacity(cfg.layers);
        let mut vcache = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            kcache.push(f(cfg.kv_heads * max_seq * cfg.head_dim)?);
            vcache.push(f(cfg.kv_heads * max_seq * cfg.head_dim)?);
        }

        let gate_bytes = q4g64_bytes(cfg.moe_inter, cfg.hidden) as u64;
        Ok(Self {
            h: f(cfg.hidden)?,
            h_norm: f(cfg.hidden)?,
            q: f(qkv)?,
            k: f(kvd)?,
            v: f(kvd)?,
            attn_out: f(qkv)?,
            proj_out: f(cfg.hidden)?,
            router_logits: f(cfg.experts)?,
            gate_out: f(cfg.moe_inter)?,
            up_out: f(cfg.moe_inter)?,
            act_out: f(cfg.moe_inter)?,
            expert_out: f(cfg.hidden)?,
            moe_acc: f(cfg.hidden)?,
            logits: f(cfg.vocab)?,
            attn_scratch: f(cfg.heads * max_seq)?,
            kcache,
            vcache,
            gate_off: 32,
            up_off: 32 + gate_bytes,
            down_off: 32 + 2 * gate_bytes,
            logits_host: vec![0u8; cfg.vocab * 4],
            router_host: vec![0u8; cfg.experts * 4],
            release_events: (0..8).map(|_| cuda.event()).collect::<Result<_>>()?,
            pending_release: std::collections::VecDeque::new(),
            next_event: 0,
            cuda,
            kernels,
            stream,
            cfg,
            core,
            pager,
            max_seq,
        })
    }

    /// Run one token at `pos`; returns the argmax over the vocab (greedy).
    pub fn step(&mut self, token: u32, pos: usize) -> Result<u32> {
        if pos >= self.max_seq {
            bail!("position {pos} exceeds max_seq {}", self.max_seq);
        }
        let c = self.cfg.clone();
        let cu = Arc::clone(&self.cuda);
        let cu = &*cu;
        let ke = self.kernels;
        let ke = &ke;
        let st = self.stream;
        let qkv = (c.heads * c.head_dim) as i32;
        let kvd = (c.kv_heads * c.head_dim) as i32;
        let hid = c.hidden as i32;

        ke.bf16_row(cu, self.core.embed, token as i32, hid, self.h, st)?;

        for l in 0..c.layers {
            let w = &self.core.layers[l];

            // Attention block.
            ke.rmsnorm(cu, self.h, w.input_ln, self.h_norm, 1, hid, c.rms_eps, st)?;
            mat_gemv(ke, cu, &w.q, self.h_norm, self.q, st)?;
            mat_gemv(ke, cu, &w.k, self.h_norm, self.k, st)?;
            mat_gemv(ke, cu, &w.v, self.h_norm, self.v, st)?;
            ke.rmsnorm(cu, self.q, w.q_norm, self.q, c.heads as i32, c.head_dim as i32, c.rms_eps, st)?;
            ke.rmsnorm(cu, self.k, w.k_norm, self.k, c.kv_heads as i32, c.head_dim as i32, c.rms_eps, st)?;
            ke.rope(cu, self.q, c.heads as i32, c.head_dim as i32, pos as i32, c.rope_theta, st)?;
            ke.rope(cu, self.k, c.kv_heads as i32, c.head_dim as i32, pos as i32, c.rope_theta, st)?;
            ke.kv_append(
                cu, self.k, self.v, self.kcache[l], self.vcache[l],
                c.kv_heads as i32, c.head_dim as i32, pos as i32, self.max_seq as i32, st,
            )?;
            ke.attn_decode(
                cu, self.q, self.kcache[l], self.vcache[l], self.attn_out, self.attn_scratch,
                c.heads as i32, c.kv_heads as i32, c.head_dim as i32,
                (pos + 1) as i32, self.max_seq as i32,
                1.0 / (c.head_dim as f32).sqrt(), st,
            )?;
            mat_gemv(ke, cu, &w.o, self.attn_out, self.proj_out, st)?;
            ke.add(cu, self.h, self.proj_out, hid, st)?;

            // Router (host-side top-k over a tiny logit vector).
            ke.rmsnorm(cu, self.h, w.post_ln, self.h_norm, 1, hid, c.rms_eps, st)?;
            mat_gemv(ke, cu, &w.router, self.h_norm, self.router_logits, st)?;
            cu.dtoh_async(&mut self.router_host, self.router_logits, st)?;
            cu.sync_stream(st)?;
            let picks = top_k_softmax(
                &f32_from_le(&self.router_host),
                c.top_k,
                c.norm_topk_prob,
            );

            // Paged expert FFNs.
            let ids: Vec<u16> = picks.iter().map(|p| p.0).collect();
            let handles = self.pager.request(l as u16, &ids)?;
            cu.memset_async(self.moe_acc, 0, c.hidden * 4, st)?;
            for (handle, (_, weight)) in handles.iter().zip(&picks) {
                self.pager.wait_stream(handle, st)?;
                let b = handle.dev;
                ke.q4g64_gemv(cu, b + self.gate_off, self.h_norm, self.gate_out, c.moe_inter as i32, hid, st)?;
                ke.q4g64_gemv(cu, b + self.up_off, self.h_norm, self.up_out, c.moe_inter as i32, hid, st)?;
                ke.silu_mul(cu, self.gate_out, self.up_out, self.act_out, c.moe_inter as i32, st)?;
                ke.q4g64_gemv(cu, b + self.down_off, self.act_out, self.expert_out, hid, c.moe_inter as i32, st)?;
                ke.scale_add(cu, self.moe_acc, self.expert_out, *weight, hid, st)?;
            }
            // Handles pin cache slots; a release must not let a new fetch
            // overwrite a slot the enqueued GEMVs haven't read yet. Record
            // an event after this layer's expert kernels and defer the
            // release until the stream has passed it — no sync needed.
            self.defer_release(handles)?;
            ke.add(cu, self.h, self.moe_acc, hid, st)?;
        }

        ke.rmsnorm(cu, self.h, self.core.final_norm, self.h_norm, 1, hid, c.rms_eps, st)?;
        mat_gemv(ke, cu, &self.core.lm_head, self.h_norm, self.logits, st)?;
        cu.dtoh_async(&mut self.logits_host, self.logits, st)?;
        cu.sync_stream(st)?;
        let logits = f32_from_le(&self.logits_host);
        let mut best = 0u32;
        let mut bestv = f32::MIN;
        for (i, v) in logits.iter().enumerate() {
            if *v > bestv {
                bestv = *v;
                best = i as u32;
            }
        }
        Ok(best)
    }

    pub fn pager_metrics(&self) -> llmpager_cuda::pager::Metrics {
        self.pager.metrics()
    }

    /// Queue handles for release once the compute stream passes an event
    /// recorded now. The event ring is small; when it wraps we block on the
    /// oldest entry (bounded pipeline depth, typically never hit).
    fn defer_release(
        &mut self,
        handles: Vec<llmpager_cuda::pager::ExpertHandle>,
    ) -> Result<()> {
        // Drain everything already complete.
        while let Some((ev_idx, _)) = self.pending_release.front() {
            if self.cuda.event_done(self.release_events[*ev_idx])? {
                let (_, done) = self.pending_release.pop_front().unwrap();
                for h in done {
                    self.pager.release(h);
                }
            } else {
                break;
            }
        }
        // If the ring slot we want is still pending, wait it out.
        if self.pending_release.len() == self.release_events.len() {
            let (ev_idx, done) = self.pending_release.pop_front().unwrap();
            self.cuda.sync_event(self.release_events[ev_idx])?;
            for h in done {
                self.pager.release(h);
            }
        }
        let ev = self.next_event;
        self.next_event = (self.next_event + 1) % self.release_events.len();
        self.cuda.record_event(self.release_events[ev], self.stream)?;
        self.pending_release.push_back((ev, handles));
        Ok(())
    }
}

fn f32_from_le(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Softmax over all logits, take top-k, optionally renormalize the picked
/// probabilities to sum to 1 (Qwen3 `norm_topk_prob`).
fn top_k_softmax(logits: &[f32], k: usize, renorm: bool) -> Vec<(u16, f32)> {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    let mut probs: Vec<(u16, f32)> =
        exps.iter().enumerate().map(|(i, e)| (i as u16, e / z)).collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    probs.truncate(k);
    if renorm {
        let s: f32 = probs.iter().map(|p| p.1).sum();
        for p in &mut probs {
            p.1 /= s;
        }
    }
    probs
}

#[cfg(test)]
mod tests {
    use super::top_k_softmax;

    #[test]
    fn topk_renorm() {
        let picks = top_k_softmax(&[0.0, 1.0, 2.0, 3.0], 2, true);
        assert_eq!(picks[0].0, 3);
        assert_eq!(picks[1].0, 2);
        assert!((picks[0].1 + picks[1].1 - 1.0).abs() < 1e-6);
        assert!(picks[0].1 > picks[1].1);
    }
}
