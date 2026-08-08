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
        // Load-time core requantization always uses our native group 64.
        ke.q4g64_gemv(cu, m.dev, x, y, m.rows, m.cols, 64, st)
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
    pager: Option<Pager>,
    pack_path: std::path::PathBuf,
    io_threads: usize,
    direct: bool,
    max_seq: usize,
    /// Speculative prefetch: warm layer L+1 with layer L's expert ids.
    pub prefetch_next: bool,
    // Device buffers.
    h: CUdeviceptr,
    h_norm: CUdeviceptr,
    q: CUdeviceptr,
    k: CUdeviceptr,
    v: CUdeviceptr,
    attn_out: CUdeviceptr,
    proj_out: CUdeviceptr,
    router_logits: CUdeviceptr,
    // Batched MoE buffers: [top_k, inter] / [top_k, hidden] contiguous.
    gate_out: CUdeviceptr,
    up_out: CUdeviceptr,
    act_out: CUdeviceptr,
    expert_out: CUdeviceptr,
    d_expert_ptrs: CUdeviceptr, // [top_k] u64 blob base addresses
    d_expert_wts: CUdeviceptr,  // [top_k] f32 routing weights
    logits: CUdeviceptr,
    attn_scratch: CUdeviceptr,
    kcache: Vec<CUdeviceptr>,
    vcache: Vec<CUdeviceptr>,
    // Blob region offsets (bytes) within an expert blob.
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    /// q4 group size of the expert pack (64 native, 32 repacked QAT).
    expert_group: i32,
    /// KV cache stored f16 (default; halves KV VRAM) or f32.
    kv_f16: bool,
    // Chunked-prefill buffers: whole-chunk hidden states + router logits.
    chunk_cap: usize,
    h_buf: CUdeviceptr,      // [chunk_cap, hidden]
    hn_buf: CUdeviceptr,     // [chunk_cap, hidden]
    router_buf: CUdeviceptr, // [chunk_cap, experts]
    router_chunk_host: Vec<u8>,
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
        direct: bool,
    ) -> Result<Self> {
        let meta = PackReader::open(pack_path)?.meta().clone();
        let cfg = Config::from_json(&meta.config)?;
        let expert_group: i32 = match meta.dtype.as_str() {
            "q4g64-gud" => 64,
            "q4g32-gud" => 32,
            other => bail!("pack dtype {other} unsupported (want q4g64-gud or q4g32-gud)"),
        };

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
            PagerConfig {
                slots_per_layer: slots,
                io_threads,
                decay_interval: 64.max(slots * 4),
                direct,
            },
        )?;

        let f = |n: usize| cuda.alloc_device(n * 4);
        let qkv = cfg.heads * cfg.head_dim;
        let kvd = cfg.kv_heads * cfg.head_dim;
        // f16 KV default; LLMPAGER_KV_F32=1 restores f32 (A/B, debugging).
        let kv_f16 = std::env::var("LLMPAGER_KV_F32").is_err();
        let kv_bytes = cfg.kv_heads * max_seq * cfg.head_dim * if kv_f16 { 2 } else { 4 };
        let mut kcache = Vec::with_capacity(cfg.layers);
        let mut vcache = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            kcache.push(cuda.alloc_device(kv_bytes)?);
            vcache.push(cuda.alloc_device(kv_bytes)?);
        }

        let gate_bytes =
            llmpager_core::quant::q4_bytes(cfg.moe_inter, cfg.hidden, expert_group as usize) as u64;
        let chunk_cap = 64usize;
        Ok(Self {
            h: f(cfg.hidden)?,
            h_norm: f(cfg.hidden)?,
            q: f(qkv)?,
            k: f(kvd)?,
            v: f(kvd)?,
            attn_out: f(qkv)?,
            proj_out: f(cfg.hidden)?,
            router_logits: f(cfg.experts)?,
            gate_out: f(cfg.top_k * cfg.moe_inter)?,
            up_out: f(cfg.top_k * cfg.moe_inter)?,
            act_out: f(cfg.top_k * cfg.moe_inter)?,
            expert_out: f(cfg.top_k * cfg.hidden)?,
            d_expert_ptrs: cuda.alloc_device(cfg.top_k * 8)?,
            d_expert_wts: f(cfg.top_k)?,
            logits: f(cfg.vocab)?,
            attn_scratch: f(cfg.heads * max_seq)?,
            kcache,
            vcache,
            gate_off: 32,
            up_off: 32 + gate_bytes,
            down_off: 32 + 2 * gate_bytes,
            expert_group,
            kv_f16,
            chunk_cap,
            h_buf: f(chunk_cap * cfg.hidden)?,
            hn_buf: f(chunk_cap * cfg.hidden)?,
            router_buf: f(chunk_cap * cfg.experts)?,
            router_chunk_host: vec![0u8; chunk_cap * cfg.experts * 4],
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
            pager: Some(pager),
            pack_path: pack_path.to_path_buf(),
            io_threads,
            direct,
            max_seq,
            prefetch_next: true,
        })
    }

    /// Resize the expert cache (VRAM budgeter): drop the pager (freeing its
    /// arenas) and build a fresh one with `slots` per layer. The cache
    /// restarts cold; the LFU rewarms within a few tokens. Perplexity is
    /// unaffected — paging is lossless at any size.
    pub fn resize_cache(&mut self, slots: u32) -> Result<()> {
        self.cuda.sync()?;
        self.pending_release.clear(); // handles die with the old pager
        self.pager = None; // free old arenas before allocating new ones
        self.pager = Some(Pager::new(
            Arc::clone(&self.cuda),
            &self.pack_path,
            PagerConfig {
                slots_per_layer: slots,
                io_threads: self.io_threads,
                decay_interval: 64.max(slots * 4),
                direct: self.direct,
            },
        )?);
        Ok(())
    }

    /// Run one token at `pos`; returns the argmax over the vocab (greedy).
    /// `want_logits: false` (prefill except the last prompt token) skips the
    /// final norm + lm_head + readback — the KV cache update is the only
    /// side effect needed.
    pub fn step(&mut self, token: u32, pos: usize, want_logits: bool) -> Result<u32> {
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
                c.kv_heads as i32, c.head_dim as i32, pos as i32, self.max_seq as i32,
                self.kv_f16, st,
            )?;
            ke.attn_decode(
                cu, self.q, self.kcache[l], self.vcache[l], self.attn_out, self.attn_scratch,
                c.heads as i32, c.kv_heads as i32, c.head_dim as i32,
                (pos + 1) as i32, self.max_seq as i32,
                1.0 / (c.head_dim as f32).sqrt(), self.kv_f16, st,
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

            // Paged expert FFNs — one batched launch per projection stage.
            let ids: Vec<u16> = picks.iter().map(|p| p.0).collect();
            let handles = self.pager.as_ref().unwrap().request(l as u16, &ids)?;
            if self.prefetch_next && l + 1 < c.layers {
                // Cross-layer id reuse is a heuristic; prefetch is
                // best-effort and never blocks.
                self.pager.as_ref().unwrap().prefetch((l + 1) as u16, &ids);
            }
            for handle in &handles {
                self.pager.as_ref().unwrap().wait_stream(handle, st)?;
            }
            let e = handles.len() as i32;
            let ptrs: Vec<u8> =
                handles.iter().flat_map(|h| h.dev.to_le_bytes()).collect();
            let wts: Vec<u8> =
                picks.iter().flat_map(|p| p.1.to_le_bytes()).collect();
            cu.htod_async(self.d_expert_ptrs, &ptrs, st)?;
            cu.htod_async(self.d_expert_wts, &wts, st)?;
            let inter = c.moe_inter as i32;
            ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.gate_off, self.h_norm, 0, self.gate_out, inter, hid, self.expert_group, e, st)?;
            ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.up_off, self.h_norm, 0, self.up_out, inter, hid, self.expert_group, e, st)?;
            ke.silu_mul(cu, self.gate_out, self.up_out, self.act_out, e * inter, st)?;
            ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.down_off, self.act_out, inter, self.expert_out, hid, inter, self.expert_group, e, st)?;
            ke.moe_reduce(cu, self.expert_out, self.d_expert_wts, self.h, e, hid, st)?;
            // Handles pin cache slots; a release must not let a new fetch
            // overwrite a slot the enqueued GEMVs haven't read yet. Record
            // an event after this layer's expert kernels and defer the
            // release until the stream has passed it — no sync needed.
            self.defer_release(handles)?;
        }

        if !want_logits {
            return Ok(0);
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

    /// Union prefill: run `tokens` (≤ chunk_cap) through all layers as one
    /// chunk. Attention stays token-by-token (causal), but each layer's
    /// routed experts are fetched once as the union over the whole chunk,
    /// in waves small enough to never pin the entire cache. Returns the
    /// greedy argmax of the last token when `want_logits`.
    pub fn step_chunk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Result<u32> {
        let n = tokens.len();
        if n == 0 || n > self.chunk_cap {
            bail!("chunk of {n} tokens (cap {})", self.chunk_cap);
        }
        if start_pos + n > self.max_seq {
            bail!("chunk exceeds max_seq {}", self.max_seq);
        }
        let c = self.cfg.clone();
        let cu = Arc::clone(&self.cuda);
        let cu = &*cu;
        let ke = self.kernels;
        let ke = &ke;
        let st = self.stream;
        let hid = c.hidden as i32;
        let h_at = |b: CUdeviceptr, t: usize| b + (t * c.hidden * 4) as u64;

        for (t, tok) in tokens.iter().enumerate() {
            ke.bf16_row(cu, self.core.embed, *tok as i32, hid, h_at(self.h_buf, t), st)?;
        }

        // Waves release eagerly, so a wave may use every slot in the layer —
        // but never more (request() rejects that outright).
        let wave = (self.pager.as_ref().unwrap().slots_per_layer() as usize).max(1);

        for l in 0..c.layers {
            let w = &self.core.layers[l];

            // Attention, causally in order; KV appended before later tokens
            // attend, so within-chunk attention sees the whole prefix.
            for t in 0..n {
                let pos = start_pos + t;
                let ht = h_at(self.h_buf, t);
                ke.rmsnorm(cu, ht, w.input_ln, self.h_norm, 1, hid, c.rms_eps, st)?;
                mat_gemv(ke, cu, &w.q, self.h_norm, self.q, st)?;
                mat_gemv(ke, cu, &w.k, self.h_norm, self.k, st)?;
                mat_gemv(ke, cu, &w.v, self.h_norm, self.v, st)?;
                ke.rmsnorm(cu, self.q, w.q_norm, self.q, c.heads as i32, c.head_dim as i32, c.rms_eps, st)?;
                ke.rmsnorm(cu, self.k, w.k_norm, self.k, c.kv_heads as i32, c.head_dim as i32, c.rms_eps, st)?;
                ke.rope(cu, self.q, c.heads as i32, c.head_dim as i32, pos as i32, c.rope_theta, st)?;
                ke.rope(cu, self.k, c.kv_heads as i32, c.head_dim as i32, pos as i32, c.rope_theta, st)?;
                ke.kv_append(
                    cu, self.k, self.v, self.kcache[l], self.vcache[l],
                    c.kv_heads as i32, c.head_dim as i32, pos as i32, self.max_seq as i32,
                    self.kv_f16, st,
                )?;
                ke.attn_decode(
                    cu, self.q, self.kcache[l], self.vcache[l], self.attn_out, self.attn_scratch,
                    c.heads as i32, c.kv_heads as i32, c.head_dim as i32,
                    (pos + 1) as i32, self.max_seq as i32,
                    1.0 / (c.head_dim as f32).sqrt(), self.kv_f16, st,
                )?;
                mat_gemv(ke, cu, &w.o, self.attn_out, self.proj_out, st)?;
                ke.add(cu, ht, self.proj_out, hid, st)?;
                // Post-norm now, straight into the chunk buffer the router
                // and expert GEMVs read from.
                ke.rmsnorm(cu, ht, w.post_ln, h_at(self.hn_buf, t), 1, hid, c.rms_eps, st)?;
                mat_gemv(
                    ke, cu, &w.router, h_at(self.hn_buf, t),
                    self.router_buf + (t * c.experts * 4) as u64, st,
                )?;
            }

            // Whole-chunk routing on the host.
            let span = n * c.experts * 4;
            cu.dtoh_async(&mut self.router_chunk_host[..span], self.router_buf, st)?;
            cu.sync_stream(st)?;
            let all = f32_from_le(&self.router_chunk_host[..span]);
            let picks: Vec<Vec<(u16, f32)>> = (0..n)
                .map(|t| {
                    top_k_softmax(
                        &all[t * c.experts..(t + 1) * c.experts],
                        c.top_k,
                        c.norm_topk_prob,
                    )
                })
                .collect();
            let mut union: Vec<u16> = picks.iter().flatten().map(|p| p.0).collect();
            union.sort_unstable();
            union.dedup();

            // Fetch the union in waves; every expert is read at most once
            // per chunk regardless of how many tokens routed to it.
            for wave_ids in union.chunks(wave) {
                let handles = self.pager.as_ref().unwrap().request(l as u16, wave_ids)?;
                for h in &handles {
                    self.pager.as_ref().unwrap().wait_stream(h, st)?;
                }
                let dev_of = |id: u16| {
                    handles
                        .iter()
                        .find(|h| h.expert == id)
                        .map(|h| h.dev)
                        .unwrap()
                };
                let inter = c.moe_inter as i32;
                for t in 0..n {
                    let sub: Vec<&(u16, f32)> = picks[t]
                        .iter()
                        .filter(|p| wave_ids.contains(&p.0))
                        .collect();
                    if sub.is_empty() {
                        continue;
                    }
                    let e = sub.len() as i32;
                    let ptrs: Vec<u8> =
                        sub.iter().flat_map(|p| dev_of(p.0).to_le_bytes()).collect();
                    let wts: Vec<u8> = sub.iter().flat_map(|p| p.1.to_le_bytes()).collect();
                    cu.htod_async(self.d_expert_ptrs, &ptrs, st)?;
                    cu.htod_async(self.d_expert_wts, &wts, st)?;
                    let x = h_at(self.hn_buf, t);
                    ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.gate_off, x, 0, self.gate_out, inter, hid, self.expert_group, e, st)?;
                    ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.up_off, x, 0, self.up_out, inter, hid, self.expert_group, e, st)?;
                    ke.silu_mul(cu, self.gate_out, self.up_out, self.act_out, e * inter, st)?;
                    ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.down_off, self.act_out, inter, self.expert_out, hid, inter, self.expert_group, e, st)?;
                    ke.moe_reduce(cu, self.expert_out, self.d_expert_wts, h_at(self.h_buf, t), e, hid, st)?;
                }
                // Eager release: the next wave's request() blocks on free
                // slots, and nothing else would drain a deferred queue —
                // deferring here deadlocks. One sync per wave is amortized
                // over the whole chunk.
                cu.sync_stream(st)?;
                for h in handles {
                    self.pager.as_ref().unwrap().release(h);
                }
            }
        }

        if !want_logits {
            return Ok(0);
        }
        let last = h_at(self.h_buf, n - 1);
        ke.rmsnorm(cu, last, self.core.final_norm, self.h_norm, 1, hid, c.rms_eps, st)?;
        mat_gemv(ke, cu, &self.core.lm_head, self.h_norm, self.logits, st)?;
        cu.dtoh_async(&mut self.logits_host, self.logits, st)?;
        cu.sync_stream(st)?;
        let logits = f32_from_le(&self.logits_host);
        let (mut best, mut bestv) = (0u32, f32::MIN);
        for (i, v) in logits.iter().enumerate() {
            if *v > bestv {
                bestv = *v;
                best = i as u32;
            }
        }
        Ok(best)
    }

    pub fn chunk_cap(&self) -> usize {
        self.chunk_cap
    }

    pub fn pager_metrics(&self) -> llmpager_cuda::pager::Metrics {
        self.pager.as_ref().unwrap().metrics()
    }

    /// The full logits of the last `step(_, _, true)` call (f32, vocab-sized).
    pub fn last_logits(&self) -> Vec<f32> {
        f32_from_le(&self.logits_host)
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
                    self.pager.as_ref().unwrap().release(h);
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
                self.pager.as_ref().unwrap().release(h);
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

// Safety: the raw CUDA handles are usable from any thread with the context
// current (bind_thread / primary context); Decoder is always used behind a
// Mutex by the server, one thread at a time.
unsafe impl Send for Decoder {}

impl Drop for Decoder {
    fn drop(&mut self) {
        // Model unloading (M5): return every device allocation. The pager
        // frees its own arenas in its Drop (which runs after this body).
        let _ = self.cuda.sync();
        let mut ptrs = vec![
            self.h, self.h_norm, self.q, self.k, self.v, self.attn_out,
            self.proj_out, self.router_logits, self.gate_out, self.up_out,
            self.act_out, self.expert_out, self.d_expert_ptrs,
            self.d_expert_wts, self.logits, self.attn_scratch,
            self.h_buf, self.hn_buf, self.router_buf,
        ];
        ptrs.extend(&self.kcache);
        ptrs.extend(&self.vcache);
        ptrs.extend(self.core.device_ptrs());
        ptrs.sort_unstable();
        ptrs.dedup();
        for p in ptrs {
            self.cuda.free_device(p);
        }
        for e in &self.release_events {
            self.cuda.destroy_event(*e);
        }
    }
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
