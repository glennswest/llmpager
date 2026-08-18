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

use anyhow::{bail, Context, Result};
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
    ram_bytes: u64,
    /// VRAM left free for other processes on the card (bytes).
    reserve_bytes: u64,
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
    /// Concurrent sequence slots (lockstep batch decode). KV caches hold
    /// `batch_cap` independent regions per layer.
    batch_cap: usize,
    // Chunked-prefill buffers: whole-chunk hidden states + router logits.
    chunk_cap: usize,
    h_buf: CUdeviceptr,      // [chunk_cap, hidden]
    hn_buf: CUdeviceptr,     // [chunk_cap, hidden]
    router_buf: CUdeviceptr, // [chunk_cap, experts]
    router_chunk_host: Vec<u8>,
    logits_host: Vec<u8>,
    router_host: Vec<u8>,
    logits_multi: Vec<Vec<f32>>,
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
        ram_bytes: u64,
        reserve_bytes: u64,
        batch_cap: usize,
    ) -> Result<Self> {
        let batch_cap = batch_cap.max(1);
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
                ram_bytes,
                reserve_bytes,
            },
        )?;

        let f = |n: usize| cuda.alloc_device(n * 4);
        let qkv = cfg.heads * cfg.head_dim;
        let kvd = cfg.kv_heads * cfg.head_dim;
        // f16 KV default; LLMPAGER_KV_F32=1 restores f32 (A/B, debugging).
        let kv_f16 = std::env::var("LLMPAGER_KV_F32").is_err();
        let kv_bytes =
            batch_cap * cfg.kv_heads * max_seq * cfg.head_dim * if kv_f16 { 2 } else { 4 };
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
            batch_cap,
            chunk_cap,
            h_buf: f(chunk_cap * cfg.hidden)?,
            hn_buf: f(chunk_cap * cfg.hidden)?,
            router_buf: f(chunk_cap * cfg.experts)?,
            router_chunk_host: vec![0u8; chunk_cap * cfg.experts * 4],
            logits_host: vec![0u8; cfg.vocab * 4],
            router_host: vec![0u8; cfg.experts * 4],
            logits_multi: Vec::new(),
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
            ram_bytes,
            reserve_bytes,
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
        let previous = self.slots_per_layer();
        self.pager = None; // free old arenas before allocating new ones
        let build = |slots: u32, reserve_bytes: u64| {
            Pager::new(
                Arc::clone(&self.cuda),
                &self.pack_path,
                PagerConfig {
                    slots_per_layer: slots,
                    io_threads: self.io_threads,
                    decay_interval: 64.max(slots * 4),
                    direct: self.direct,
                    ram_bytes: self.ram_bytes,
                    reserve_bytes,
                },
            )
        };
        // The old arena is already gone, so a failure here would leave the
        // decoder with no pager at all and panic on the next token. Put the
        // previous size back before reporting the failure.
        match build(slots, self.reserve_bytes) {
            Ok(p) => {
                self.pager = Some(p);
                Ok(())
            }
            Err(e) => {
                // Restore service at the previous size, ignoring the reserve:
                // an unsatisfiable reserve is exactly what tends to land here,
                // and re-applying it would fail again and leave no pager.
                self.pager = Some(build(previous, 0).context(
                    "cache resize failed and the previous size could not be restored",
                )?);
                Err(e.context(format!("cache resize to {slots} slots failed; kept {previous}")))
            }
        }
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
        self.pager.as_ref().unwrap().tick();
        Ok(best)
    }

    /// Union prefill: run `tokens` (≤ chunk_cap) through all layers as one
    /// chunk on sequence slot 0. Returns the greedy argmax of the last
    /// token when `want_logits`.
    pub fn step_chunk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Result<u32> {
        let entries: Vec<(u32, usize, usize)> =
            tokens.iter().enumerate().map(|(i, t)| (*t, start_pos + i, 0)).collect();
        let out = self.step_multi(&entries, want_logits)?;
        Ok(out.last().copied().unwrap_or(0))
    }

    /// General multi-token step: each entry is (token, position, sequence
    /// slot). Within one call, entries on the same slot must be in
    /// ascending position order (prefill); entries on distinct slots are
    /// independent streams decoding in lockstep (batch). Every layer
    /// fetches the union of all entries' experts once, in waves. Returns
    /// the greedy argmax per entry when `want_logits` (else zeros).
    pub fn step_multi(
        &mut self,
        entries: &[(u32, usize, usize)],
        want_logits: bool,
    ) -> Result<Vec<u32>> {
        let n = entries.len();
        if n == 0 || n > self.chunk_cap {
            bail!("{n} entries (cap {})", self.chunk_cap);
        }
        for &(_, pos, seq) in entries {
            if pos >= self.max_seq {
                bail!("position {pos} exceeds max_seq {}", self.max_seq);
            }
            if seq >= self.batch_cap {
                bail!("sequence slot {seq} exceeds batch cap {}", self.batch_cap);
            }
        }
        // Waves below are sized to the whole layer, so this path can only
        // make progress from a clean pin state. A generation that ended in
        // `step` leaves up to `release_events.len()` layers' worth of
        // handles pinned in the deferred ring, and nothing here would ever
        // drain them — the first wave over one of those layers would stall
        // on slots this very thread holds, forever. Drain the ring first.
        self.flush_pending_release()?;
        let c = self.cfg.clone();
        let cu = Arc::clone(&self.cuda);
        let cu = &*cu;
        let ke = self.kernels;
        let ke = &ke;
        let st = self.stream;
        let hid = c.hidden as i32;
        let h_at = |b: CUdeviceptr, t: usize| b + (t * c.hidden * 4) as u64;

        let kv_elem = if self.kv_f16 { 2 } else { 4 };
        let seq_off = |seq: usize| {
            (seq * c.kv_heads * self.max_seq * c.head_dim * kv_elem) as u64
        };
        for (t, &(tok, _, _)) in entries.iter().enumerate() {
            ke.bf16_row(cu, self.core.embed, tok as i32, hid, h_at(self.h_buf, t), st)?;
        }

        // Waves release eagerly, so a wave may use every slot in the layer —
        // but never more (request() rejects that outright).
        let wave = (self.pager.as_ref().unwrap().slots_per_layer() as usize).max(1);

        for l in 0..c.layers {
            let w = &self.core.layers[l];

            // Attention, causally in order; KV appended before later tokens
            // attend, so within-chunk attention sees the whole prefix.
            for (t, &(_, pos, seq)) in entries.iter().enumerate() {
                let ht = h_at(self.h_buf, t);
                let (kc, vc) = (self.kcache[l] + seq_off(seq), self.vcache[l] + seq_off(seq));
                ke.rmsnorm(cu, ht, w.input_ln, self.h_norm, 1, hid, c.rms_eps, st)?;
                mat_gemv(ke, cu, &w.q, self.h_norm, self.q, st)?;
                mat_gemv(ke, cu, &w.k, self.h_norm, self.k, st)?;
                mat_gemv(ke, cu, &w.v, self.h_norm, self.v, st)?;
                ke.rmsnorm(cu, self.q, w.q_norm, self.q, c.heads as i32, c.head_dim as i32, c.rms_eps, st)?;
                ke.rmsnorm(cu, self.k, w.k_norm, self.k, c.kv_heads as i32, c.head_dim as i32, c.rms_eps, st)?;
                ke.rope(cu, self.q, c.heads as i32, c.head_dim as i32, pos as i32, c.rope_theta, st)?;
                ke.rope(cu, self.k, c.kv_heads as i32, c.head_dim as i32, pos as i32, c.rope_theta, st)?;
                ke.kv_append(
                    cu, self.k, self.v, kc, vc,
                    c.kv_heads as i32, c.head_dim as i32, pos as i32, self.max_seq as i32,
                    self.kv_f16, st,
                )?;
                ke.attn_decode(
                    cu, self.q, kc, vc, self.attn_out, self.attn_scratch,
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

        self.pager.as_ref().unwrap().tick();
        if !want_logits {
            return Ok(vec![0; n]);
        }
        // Per-entry logits; the wrapper cases (prefill, single decode) read
        // one, lockstep batch reads all. Distinct sequence slots each need
        // theirs; same-slot prefill only needs the last, so skip the rest.
        self.logits_multi.clear();
        let mut out = Vec::with_capacity(n);
        for (t, &(_, _, seq)) in entries.iter().enumerate() {
            let is_last_for_slot = entries
                .iter()
                .skip(t + 1)
                .all(|&(_, _, s2)| s2 != seq);
            if !is_last_for_slot {
                self.logits_multi.push(Vec::new());
                out.push(0);
                continue;
            }
            ke.rmsnorm(cu, h_at(self.h_buf, t), self.core.final_norm, self.h_norm, 1, hid, c.rms_eps, st)?;
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
            self.logits_multi.push(logits);
            out.push(best);
        }
        Ok(out)
    }

    /// Logits per entry from the last `step_multi(_, true)` call (empty for
    /// entries that were not the last of their sequence slot).
    pub fn last_logits_multi(&self) -> &[Vec<f32>] {
        &self.logits_multi
    }

    pub fn chunk_cap(&self) -> usize {
        self.chunk_cap
    }

    /// Slots per layer actually allocated (a VRAM reserve may clamp it).
    pub fn slots_per_layer(&self) -> u32 {
        self.pager.as_ref().map(|p| p.slots_per_layer()).unwrap_or(0)
    }

    /// Free and total VRAM on the device, across every process.
    pub fn mem_info(&self) -> Result<(u64, u64)> {
        self.cuda.mem_info()
    }

    /// Change the VRAM reserve; takes effect on the next `resize_cache`,
    /// which frees the current arena before the new one is sized.
    pub fn set_reserve_bytes(&mut self, bytes: u64) {
        self.reserve_bytes = bytes;
    }

    pub fn batch_cap(&self) -> usize {
        self.batch_cap
    }

    pub fn pager_metrics(&self) -> llmpager_cuda::pager::Metrics {
        self.pager.as_ref().unwrap().metrics()
    }

    pub fn expert_stats(&self) -> Vec<u64> {
        self.pager.as_ref().unwrap().expert_stats()
    }

    pub fn prewarm(&self, counts: &[u64]) -> Result<usize> {
        self.pager.as_ref().unwrap().prewarm(counts)
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

    /// Bytes of KV state one token occupies across all layers — the host
    /// cost of parking a session's context.
    pub fn kv_bytes_per_token(&self) -> usize {
        let elem = if self.kv_f16 { 2 } else { 4 };
        2 * self.cfg.layers * self.cfg.kv_heads * self.cfg.head_dim * elem
    }

    fn kv_geometry(&self, seq: usize, len: usize) -> Result<(usize, u64, u64)> {
        if seq >= self.batch_cap {
            bail!("sequence slot {seq} exceeds batch cap {}", self.batch_cap);
        }
        if len > self.max_seq {
            bail!("kv length {len} exceeds max_seq {}", self.max_seq);
        }
        let elem = if self.kv_f16 { 2 } else { 4 };
        let run = len * self.cfg.head_dim * elem;
        let head_stride = (self.max_seq * self.cfg.head_dim * elem) as u64;
        let slot_off = seq as u64 * self.cfg.kv_heads as u64 * head_stride;
        Ok((run, head_stride, slot_off))
    }

    /// Copy sequence slot `seq`'s KV for positions [0, len) to host memory.
    /// A slot is [kv_heads, max_seq, head_dim] per layer, so a prefix is
    /// `kv_heads` separate runs rather than one contiguous block.
    pub fn kv_export(&self, seq: usize, len: usize) -> Result<Vec<u8>> {
        let (run, head_stride, slot_off) = self.kv_geometry(seq, len)?;
        let mut out = vec![0u8; self.kv_bytes_per_token() * len];
        let mut at = 0usize;
        for l in 0..self.cfg.layers {
            for base in [self.kcache[l], self.vcache[l]] {
                for h in 0..self.cfg.kv_heads {
                    let src = base + slot_off + h as u64 * head_stride;
                    self.cuda.dtoh_async(&mut out[at..at + run], src, self.stream)?;
                    at += run;
                }
            }
        }
        self.cuda.sync_stream(self.stream)?;
        Ok(out)
    }

    /// Inverse of `kv_export`: restore a parked context into slot `seq`.
    pub fn kv_import(&mut self, seq: usize, len: usize, blob: &[u8]) -> Result<()> {
        let (run, head_stride, slot_off) = self.kv_geometry(seq, len)?;
        let want = self.kv_bytes_per_token() * len;
        if blob.len() != want {
            bail!("kv blob is {} bytes, expected {want} for {len} tokens", blob.len());
        }
        let mut at = 0usize;
        for l in 0..self.cfg.layers {
            for base in [self.kcache[l], self.vcache[l]] {
                for h in 0..self.cfg.kv_heads {
                    let dst = base + slot_off + h as u64 * head_stride;
                    self.cuda.htod_async(dst, &blob[at..at + run], self.stream)?;
                    at += run;
                }
            }
        }
        self.cuda.sync_stream(self.stream)?;
        Ok(())
    }

    /// Release every handle still queued in the deferred ring, waiting on
    /// each recorded event first. Restores the "this decoder pins nothing"
    /// invariant that `step_multi`'s wave fetching depends on.
    fn flush_pending_release(&mut self) -> Result<()> {
        while let Some((ev_idx, _)) = self.pending_release.front() {
            self.cuda.sync_event(self.release_events[*ev_idx])?;
            let (_, done) = self.pending_release.pop_front().unwrap();
            for h in done {
                self.pager.as_ref().unwrap().release(h);
            }
        }
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
