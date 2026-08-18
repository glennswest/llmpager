# llmpager — Project Context

MoE expert-paging inference engine for Linux + NVIDIA. Runs Mixture-of-Experts
LLMs whose weights exceed VRAM by keeping only the shared core (embeddings,
attention, router, norms) plus KV cache resident on the GPU and streaming
routed experts from NVMe on demand, with an LFU cache of hot experts in VRAM.

Inspired by turbo-fieldfare (Apple Silicon / Metal / SSD streaming); this is
the Linux + CUDA equivalent.

## Version

Current: **0.21.0** (pre-1.0, API unstable)

Version locations (must all match):
- `Cargo.toml` — `[workspace.package] version` (all crates inherit it)

## Target Hardware

- **ai.g8.lo** (192.168.8.140) — VM 600 on pve.g8.lo
  - RTX 5060 Ti (Blackwell GB206), 16GB VRAM, driver 610.43.02, CUDA 13.3
  - 12 vCPU, 64GB RAM, 200GB virtio-scsi disk (NVMe-backed, ~130GB free)
  - Debian 13, Python 3.13.5, user `glenn` (ssh key auth)

## Architecture (summary — details in docs/DESIGN.md)

- **Expert pack** (`.llmpk`): on-disk format; per-(layer, expert) weight blobs,
  4096-byte aligned for O_DIRECT reads. Converter builds it from a HF checkpoint.
- **ExpertCache**: per-layer slotted VRAM cache with aged-LFU eviction
  (turbo-fieldfare uses 16 slots/layer; ours is configurable).
- **Pager**: I/O thread pool doing O_DIRECT preads into a pinned host ring
  buffer, then `cudaMemcpyAsync` on a dedicated copy stream; CUDA events gate
  compute. CPU fallback path (no CUDA) for development/tests on the Mac.
- **Runtime**: decode loop where router top-k for a layer triggers cache
  lookups + miss fetches, overlapped with attention compute.
- Reference model target: Qwen3-30B-A3B (4-bit) — experts dominate weights;
  core + cache + KV fit comfortably in 16GB.

## Work Plan

### M0 — Scaffold & environment validation
- [x] Probe ai.g8.lo (GPU, driver, Python, disk)
- [x] Repo scaffold: CLAUDE.md, README, CHANGELOG, .gitignore, Cargo workspace
- [x] Create GitHub repo, first push (github.com/glennswest/llmpager)
- [x] Core data structures: ExpertCache (aged LFU), pack format read/write
- [x] Unit tests (llmpager-core has no GPU deps; 8/8 green on macOS + Linux)
- [x] Bench binary: O_DIRECT read bandwidth, pinned H2D bandwidth,
      end-to-end paged-fetch latency (libcuda loaded at runtime, no toolkit)
- [x] Run M0 bench on ai.g8.lo, record numbers in docs/BENCHMARKS.md

### M1 — Paging core, GPU-proven
- [x] `llmpager-cuda` crate: driver wrapper (moved from bench) + CUDA events
- [x] Async Pager: io worker pool, miss → pread → H2D → event; hits and
      in-flight slots share per-slot event readiness; condvar on stall
- [x] Prefetch hook (fire-and-forget fetch, pin released after fill)
- [x] Pager-based bench on ai.g8.lo: prefetch=1 +42% over sync loop;
      48 slots + prefetch → 113 tok/s ceiling, 8.8 ms/token wait
- [x] Metrics: hit rate, bytes, fetch latency histogram

### M2 — Real model end-to-end (in progress)
- [x] `q4g64` quantization (symmetric 4-bit, group 64) in llmpager-core
- [x] Converter: HF Qwen3-MoE safetensors → .llmpk expert pack (q4g64)
      + resident core as a separate pageable safetensors file (unit-tested
      on synthetic checkpoint; real Qwen3-30B-A3B run pending download)
- [ ] Run converter on Qwen3-30B-A3B on ai.g8.lo (checkpoint downloading
      to ~/models/qwen3-30b-a3b; auto-convert watcher armed)
- [x] Kernel toolchain: nvcc 13.3 on ai.g8.lo; build.rs .cu→PTX (compute_80,
      driver JIT to sm_120); module load/launch via runtime-loaded libcuda
- [x] `q4g64_gemv` kernel verified on GPU (rel err <6e-6; 35 GB/s naive —
      M3: batch per-layer launches + vectorize)
- [x] Decode kernel set verified on GPU (worst rel err 1.9e-6): rmsnorm,
      bf16 GEMV, silu-mul, add, NeoX RoPE, GQA decode attention (f32 KV,
      two-pass softmax), bf16 embed-row gather
- [x] Runtime crate `llmpager-run`: core loader (safetensors → VRAM), KV
      cache, host-side router top-k + greedy sampling, tokenizer, decode
      loop wiring kernels + pager
- [x] End-to-end synthetic smoke test on GPU: gen-test checkpoint →
      convert → decode produces deterministic tokens (61% cache hit on
      8-expert toy)
- [x] Real tokens on ai.g8.lo: Qwen3-30B-A3B converted (21s; 15.4GB pack
      + 3.1GB core) and decoding coherently at ~19 tok/s greedy, 83%
      expert-cache hit (numbers in docs/BENCHMARKS.md)
- [x] Qwen3-Coder-30B-A3B pack converted (20s) and decoding idiomatic
      Rust at 18.6 tok/s cold / 77.6% hit — second model, same engine
- [x] Perplexity validation (`--ppl`): 8.62 on English text; identical
      across cache sizes (paging lossless); q4-core +4.5% PPL quantified

### M3 — Performance (in progress)
- [x] Vectorized GEMV kernels (16B loads): q4g64 35→122 GB/s, bf16 no
      longer the limiter; decode 19.9 → 31.1 tok/s
- [x] Event-based deferred expert-handle release (per-layer stream sync
      removed): 31.1 → 33.1 tok/s
- [ ] Reduce host round-trips: GPU router top-k (or batched dtoh), fewer
      per-layer syncs; profile launch count
- [ ] Batched per-layer expert GEMV (one launch for all top-k experts)
- [ ] Speculative expert prefetch (reuse-distance / gate-estimate heuristics)
- [ ] Quantize resident core to q4 (core streaming is ~4.3GB/token bf16 —
      biggest remaining bandwidth term)
- [ ] Optional GPUDirect Storage (cuFile) path
- [ ] tokens/sec + hit-rate benchmarks vs cache size

### M4 — Serving
- [x] OpenAI-compatible HTTP endpoint (`llmpager-serve`: /v1/models,
      /v1/completions, /v1/chat/completions with Qwen3 ChatML template,
      EOS stopping, usage + tok/s in responses)
- [x] Deployment unit: systemd `llmpager.service` on ai.g8.lo, port 8090,
      base model + RAM tier; verified over the network from the Mac

### M5 — Multi-model (in progress)
- [x] Model registry: serve.json config, `model:` routing, lazy warm-up,
      LRU eviction with full VRAM reclaim (Pager/Decoder Drop free all
      device memory); evict-all retry on VRAM pressure
- [x] Both 30B packs warm simultaneously on 16GB (24 slots each,
      13-14 tok/s each); verified live on ai.g8.lo:8090
- [x] VRAM budgeter v1: solo model gets slots_solo (48), residents shrink
      to slots (24) when another model warms; resize = pager rebuild
      (~0.8s model load measured live; journal-verified rebalance)
- [ ] Budgeter v2: activity-based sizing (request-rate weighted), not
      just warm-count
- [ ] Disk-bandwidth arbitration between concurrently faulting models

### M6 — Kimi K2.6 (1T / 32B active) — in progress
Target: moonshotai/Kimi-K2.6 (downloading to /data/models/kimi-k2.6;
/data grown 800G→2TB). Text stack is DeepseekV3ForCausalLM inside a
multimodal kimi_k25 wrapper (vision tower ignored). 61 layers (layer 0
dense, 18432 inter), 384 routed experts top-8 + 1 shared (2048 inter),
MLA (q_lora 1536, kv_lora 512, nope 128 + rope 64, v 128, 64 heads),
YaRN (factor 64, orig 4096, theta 50000), sigmoid router with
noaux_tc bias + routed_scaling 2.827, vocab 163840. Checkpoint is QAT
**int4 group-32 symmetric** (compressed-tensors: weight_packed int32 +
weight_scale), 1.1TB — repack, don't requantize.

- [x] Core: parametric group size (q4g32 alongside q4g64) — quant fns,
      q4_store_group for bit-exact repack; 11 tests green
- [x] Kernels: q4 GEMV + batch generalized to group param; verified on
      GPU (g32 @ 2048x7168: 1.4e-5, 192.5 GB/s; g64 no regression);
      decoder reads group from pack dtype (q4g32-gud)
- [x] Converter: kimi_k25 auto-detected; compressed-tensors int4 g32
      repacked bit-exactly (I32 [rows, cols/8], value k at bits 4k;
      scales bf16→f16); core = all non-expert language_model tensors,
      prefix stripped; vision dropped; moe_layer_offset=1 for the dense
      first layer. Round-trip unit test vs reference dequant green.
      Attention/shared/dense stay bf16 in the core file — the runtime
      requantizes to q4g64 at load (bf16 core is 13.4GB, over VRAM).
- [x] Real checkpoint converted (568GB int4 ckpt -> 570.9GB pack +
      23.4GB core, ~25min); tokenizer.json built from tiktoken
      (special-id mismatch in tokenizer_config found and fixed)
- [x] **FIRST REAL TOKENS 2026-08-08**: coherent MoE explanation from
      Kimi K2.6 (1T params) on the 16GB card — 0.35 tok/s decode,
      2.1% hit, 829GB streamed (slots=4, O_DIRECT, cold cache).
      Root-caused garbage-output bug: compressed-tensors int4 is
      offset-binary (value+8), not two's complement; repack is now a
      verbatim nibble copy, verified vs official unpack_from_int32
- [x] Infra opts: VM RAM 64→128GB (host has 187GB); disk iothreads —
      8-thread O_DIRECT at 25MB blobs now 6.94 GB/s (was 3.6)
- [x] Kernels: MLA set verified on GPU (worst rel err 3e-7) —
      mla_rope (interleaved pairs, host freq table), mla_attn_decode
      (MQA over compressed [max_seq, 576] cache), bf16_gemv_batch
      (strided w/x/y), strided_copy
- [x] Runtime: KimiDecoder — absorbed MLA decode (kv_b reordered to
      kt/vw at load), whole core q4g64-requantized, embed table
      host-resident (row gather per token), sigmoid+bias router
      (weights from unbiased scores, renorm × routed_scaling), shared
      expert, dense layer 0, YaRN inv_freq + softmax mscale;
      AnyDecoder CLI dispatch on kv_lora_rank; --min-expert-weight
      expert-dropping knob
- [x] Synthetic end-to-end on GPU: gen-test-kimi → convert → decode:
      deterministic tokens, bit-identical across cache sizes
- [ ] Serve: register kimi (AnyDecoder in llmpager-serve); Kimi chat
      template; expect 0.5-1.5 tok/s single-stream (batch use)
- [ ] Union prefill (chunked, per-layer expert union) — required for
      usable prompt processing at Kimi scale

### M7 — Throughput & serving quality (in progress)
Approved 2026-08-08: items 1-5.
- [x] 1. Union prefill (v0.12.0): step_chunk in both decoders; waves with
      eager release (deferred release deadlocked — fixed). Measured on
      Qwen3-30B, 1215-tok prompt, O_DIRECT under download contention:
      71.9s → 47.2s prefill, 166 → 108GB streamed; chunk union averaged
      ~46 of 128 experts/layer. CLI --chunk=1 for A/B.
- [x] 2. Sampling (v0.12.0): temp/top-p/top-k/repetition penalty/seed,
      CLI + serve request fields; greedy default; 5 unit tests.
- [x] 3. Kimi serving (v0.12.0): AnyDecoder registry (auto-detect),
      im_user/im_middle chat template (matches checkpoint jinja);
      deployed live, sampled haiku verified via API.
- [x] 4. Batched decode (v0.15.0) — step_multi on both engines,
      serve prompt-array/n>1 API, selftests PASS; Qwen batch 2 =
      1.94x aggregate (97% efficiency), batch 4 = 2.32x. Expert
      dropping validated free at 0.05 (+0.10% PPL), kimi default.
      Original staged plan:
      (a) [x] f16 KV cache (default; LLMPAGER_KV_F32=1 reverts) — PPL
          12.9575 (f32) vs 12.9671 (f16), +0.07% = noise; and +14%
          decode (26.9 -> 30.6 tok/s teacher-forced, less cache BW),
      (b) step_multi(entries: (token, pos, seq_slot)) — step_chunk's
          union machinery with per-slot KV caches and per-entry seq_len
          in attention (grid.y = entry),
      (c) serve: OpenAI prompt-array / n>1 requests decode in lockstep.
- [x] 5. Profiling + pre-warm shipped; first A/B RETRACTED (ran on a
      stale box binary that ignored the flags — fail-loud deploy.sh
      added). Real A/B rerunning in the zstd chain. GPU router top-k:
      skipped — irrelevant at disk-bound speeds.

### M8 — Data density & memory organization (planned 2026-08-08)
Deep-dive conclusions; ranked. Items 1-3 are no-quality-risk memory
reorganization; 4 is a measured quality/VRAM trade; 5-7 are further out.
Already optimal, do not touch: bit-exact int4 QAT repack, 4096-aligned
blobs, layer-major pack ordering (union prefill reads a layer as one
near-sequential ~9.6GB sweep), uniform blob size (no slot fragmentation).
- [x] 1. Managed RAM tier (v0.14.0): anonymous NORESERVE mapping +
      global-pool LFU bookkeeping, read-through in the io workers,
      write-allocate on disk reads; --ram-gb / serve ram_gb (kimi 80GB
      = ~3,200 experts). First 120-token run: 0.35 -> 0.57 tok/s (+63%)
      with the tier still cold; converges higher on long runs.
- [x] 2. Global VRAM slot pool (v0.20.0/v0.21.0, opt-in and OFF):
      **better cache, slower engine.** Hit 51.3 -> 54.3% (qwen3-30b) and
      53.3 -> 59.3% (coder) at slots=24, 6-13% fewer bytes, PPL identical —
      all reproducible. But under `--direct=1`, the only stable timing
      instrument on this box, it is ~10% *slower* (12.04/11.93 vs
      10.45/10.93 tok/s). Something costs more than the fetches it saves;
      unprofiled candidates are the eviction scan growing 24 -> 1152 slots
      inside the I/O workers' mutex, and one large residency map replacing
      48 small ones. Next step is a profile, then a sampled or bucketed
      victim search. Full write-up in docs/PERFORMANCE.md.
- [ ] 3. f16 for Kimi attention state: MLA cache rows f32 -> f16
      (2.3KB -> 1.15KB/token; 256K ctx: 590 -> 295MB/seq) and absorbed
      kv_b (kt/vw) bf16 -> f16/fp8 (~0.5GB VRAM back to expert slots).
      Qwen f16-KV gate already passed (+0.07% PPL, +14% speed).
- [ ] 4. fp8 core A/B (needs real-model PPL baseline): attention core at
      fp8 (~6.7GB, native on Blackwell) vs q4g64 (~3.4GB) vs bf16
      (13.4GB, doesn't fit). Per-tensor choice — attention fp8 +
      shared/dense q4 is the likely sweet spot. Requires fp8 GEMV kernel.
- [x] 5. Entropy-coded pack (v0.16.0): spike measured 13.2% ceiling,
      zstd -3 achieves it; shipped as CPU-decode-in-workers (no GPU
      stage needed). Verdict from real A/B: 496GB (-13.0%) but decode
      0.32 vs 0.58 tok/s — decompress latency rides the miss critical
      path. SPACE feature (convert with LLMPAGER_COMPRESS=zstd), not a
      speed lever. Pre-warm rerun on the fixed binary: REAL +24%
      decode / +67% prefill (tier 42.6->61.4% hits) — serving
      self-profiling now pays off directly.
- [ ] 6. Blob phase split (gate+up | down): stream each expert in two
      phases, halving staging footprint and overlapping finer. Small win.
- [ ] 7. Cold-expert lower-bit tier (int3 for rarely-routed experts):
      only behind a PPL gate — breaks the QAT guarantee; last resort.
- [ ] 8. Core allocation packing: Kimi core load makes ~700 cuMemAllocs
      (norms etc. round up to ~2MB granularity => ~1-1.5GB waste; caused
      first-run OOM at slots=4). Pack per-layer weights into one arena
      alloc each, like the pager slot arenas.

### M9 — Unsloth Dynamic-quant import — DEFERRED 2026-08-11
Parked at user request (higher-priority work). State at parking: GGUF v3
reader + --gguf-info landed (commit in tree); the UD-Q2 download and ALL
Kimi artifacts (checkpoint, pack, core) were removed from /data during
an infra cleanup — /data now has 1.7TB free. To resume M9: re-download
unsloth/Kimi-K2.6-GGUF UD-Q2_K_XL (350GB), gguf-info to confirm layout,
then the repack converter + K-quant kernels per the checklist below.
Note: kimi-k2.6 was de-listed from deploy/serve.json in 1647437, so the
serving config is clean; re-adding it requires re-converting the pack.
New artifacts on /data not from this plan (parallel work): qwen3-235b-
a22b (downloading?) and qwen3-30b-a3b-x12.llmpk.

### (deferred) M9 original assessment (2026-08-09)
Deep dive on unsloth.ai: their fine-tuning stack/Studio are not relevant
to us, but **Unsloth Dynamic 2.0 GGUFs are the remaining single-stream
speed lever**. They publish calibrated, per-tensor mixed-precision quants
of the exact models we run:
- Kimi K2.6 UD-Q2_K_XL: **350GB (-39% vs our 571GB int4 pack)** with
  calibration-tuned quality (attention kept higher-bit); Q4_K_XL ~585GB.
- Disk-bound math: 350GB pack => ~7.3GB/token worst case vs 12 =>
  ~+60% decode on top of pre-warm (0.72 -> ~1.15 tok/s est.), RAM tier
  coverage 14% -> 23% of experts, and 220GB disk back.
- Kimi K3 (2.8T) exists at 594GB 1-bit-ish / 861GB UD-Q2 — a future
  option once disk allows; quality at 1-bit needs our PPL gate.
Adoption plan (when started):
- [ ] Converter mode: GGUF -> .llmpk (parse GGUF, repack expert tensors
      preserving GGML K-quant superblocks; per-region dtype tag in the
      blob header)
- [ ] Kernels: q2_K / q3_K / q4_K GEMV (port llama.cpp dequant math
      into our warp-per-row + batched skeleton; verify vs CPU reference)
- [ ] Core from the same GGUF (their scheme keeps attention high-bit —
      replaces our blind q4g64 requant with calibrated choices)
- [ ] PPL gate vs our int4 pack before switching serving
Also worth imitating cheaply: calibration-guided per-tensor precision
for our own core requant (their Calibration_v3/v5 insight: chat-template
data, not wikitext).

### M10 — Multi-session serving (KV store) — shipped 2026-08-17 (v0.18.0)
One warm model, many independent contexts. Today every request re-prefills
from position 0 on sequence slot 0, so the same model cannot hold two
purposes at once and every chat turn re-pages the whole conversation.
The kernels already isolate sequences (`step_multi` takes a seq slot and
per-slot KV regions); what is missing is identity, persistence, and reuse.
- [x] 1. Decoder KV export/import: `kv_export(slot, len)` / `kv_import`
      on both engines. Qwen slot layout is [kv_heads, max_seq, head_dim]
      (strided prefix copy); Kimi's MLA cache is [max_seq, qk] (one
      contiguous run). ~96KB/token for qwen3-30b, ~192KB/token for 235B.
- [x] 2. `SessionStore` per warm model: session id -> {tokens, VRAM seq
      slot | host-RAM blob}. LRU over the `batch_cap` VRAM slots, then
      LRU over a host-RAM budget (`session_ram_gb`). Restoring a session
      is an H2D copy (~10ms) versus seconds of re-prefill paging.
- [x] 3. Prefix reuse: longest common prefix between the session's stored
      tokens and the new prompt is kept; only the divergent suffix is
      prefilled. Serves both continuation (chat) and shared-prefix fan-out
      (slidemaker sends every slide behind one system prompt).
- [x] 4. API: `session` field on completions/chat; GET/DELETE /v1/sessions.
- [x] 5. `--session-selftest`: prefix-reused output must be bit-identical
      to a full prefill, and KV export/import must round-trip.
- [ ] 6. (deferred) Continuous batching so sessions decode *concurrently*
      in lockstep, sharing every expert fetch — the throughput lever on
      H2D-bound models. Needs a scheduler thread owning the decoder; the
      session slot machinery above is its prerequisite.

### M11 — VRAM co-tenancy — shipped 2026-08-18 (v0.19.0), closes #6
Blocking slidemaker: IndexTTS-2 could not start alongside llmpager.
- [x] `cuMemGetInfo` in the driver; `reserve_bytes` in `PagerConfig`
- [x] `Pager::new` clamps slots to leave the reserve free (warm-up + resize)
- [x] `reserve_mb` in serve.json (per model or global), `--reserve-mb` CLI
- [x] `POST /v1/admin/slots` {target|reserve_mb}, `GET /v1/admin/vram`
- [x] resize_cache made failure-safe (it frees the arena before rebuilding,
      so a failed rebuild used to leave pager: None and panic the server)
- [ ] Automatic yielding on observed pressure — deliberately not done:
      CUDA gives no signal about *who* needs memory, so polling free VRAM
      would shrink the cache for any transient allocation. Explicit asks
      are the honest interface until something better exists.
- [ ] KV sequence slots do not resize; only the expert cache does. Yielding
      those would mean dropping live session contexts.

## Session Log

- 2026-08-18 (later still): **The global pool, measured properly (v0.21.0)
  — and my own +14% claim withdrawn.** Two corrections, both mine.
  - Aging was counted in *insertions*, so one constant meant a different
    real cadence per layer count, cache size and miss rate — hence the
    cliff (54.9% at slots=24, 2.1% at slots=8, same setting). `Pager::tick`
    now ages the shared pool every N forward passes (default 4); the
    response curve is smooth and unimodal at both sizes. When a tuning
    constant is cliff-edged, suspect the unit.
  - The "+14% decode" I put in the v0.20.0 changelog was noise. Undirected
    generation timing here spans ±20% (the same baseline gave 9.29-12.33
    tok/s across batches), and the winner flipped between measurement
    batches. Under `--direct=1`, stable to ~1%, the shared pool is ~10%
    *slower* while moving 6% fewer bytes. So: cache gains real, throughput
    loss real, feature stays off. **Deterministic counters for tuning;
    `--direct=1` and repeats for wall-clock.**

- 2026-08-18 (later): **Flat expert ids and an optional global slot pool
  (v0.20.0), closes #1.** `ExpertCache` now keys on a flat `u32` and draws
  slots from partitions; one partition per layer is the old behaviour, one
  partition is the global pool M8 wanted, and a flat expert population (what
  neuro-tcore asked for in #1) is one partition with no folding.
  - The global pool helps, but less than the work plan assumed, and only at
    the right decay cadence: +3.6pp hit, -7.5% bytes, ~+14% decode at
    slots=24. PPL identical, so it is lossless.
  - **Two measurement lessons worth keeping.** First, my initial A/B said the
    pool was clearly *worse* — it was running the x48 decay default I had
    just guessed at; the parameter, not the idea, was wrong. Second,
    wall-clock decode on this box varies ~20% run to run (the same baseline
    gave 9.53-12.36 tok/s), so single-run timings decide nothing. Hit rate
    and bytes streamed are exactly reproducible and should drive tuning,
    with paired repeats for timing.
  - Kept opt-in: the failure mode of a mistuned decay constant is a 2% hit
    rate, not a small regression. Deriving it from the observed miss rate is
    the fix that would make it safe to default on.

- 2026-08-18: **VRAM co-tenancy shipped (v0.19.0), closes #6.** The expert
  arena is now sized against *free* VRAM, not just against llmpager's own
  warm set, so another process on the card is no longer starved by cache we
  can give up. Verified with the actual neighbour: at 48 slots IndexTTS-2's
  torch OOMs asking for 7GB (5.43 GiB free); after a `{"reserve_mb": 9000}`
  admin call llmpager drops to 23 slots, 8990 MB free, the allocation
  succeeds, and llmpager keeps answering.
  - Two hazards found by testing the failure path rather than the happy one.
    `resize_cache` frees the old arena *before* building the new one, so any
    failure after that point left `pager: None` and panicked the next token
    — testing `reserve_mb=15000` on a 16GB card killed the server outright.
    The fallback rebuild also has to ignore the reserve, since an
    unsatisfiable reserve is precisely what lands there; the first version
    of the fix inherited it and failed twice.
  - Deliberately not automatic: CUDA exposes no cross-process pressure
    signal, so polling free VRAM would shrink the cache for any transient
    allocation by anyone. An explicit ask from the neighbour is honest and
    matches how slidemaker actually runs (LLM pass, then TTS pass).

- 2026-08-17 (later): **Multi-session serving shipped (v0.18.0).** One warm
  model now holds many named KV contexts; a request naming a session decodes
  on that session's own sequence slot and prefills only what its KV does not
  already cover. Slot 0 stays the anonymous lane, so sessionless traffic
  cannot clobber a session. Parking is D2H, restoring is H2D (96KB/token on
  qwen3-30b), against seconds of expert paging to rebuild the same context.
  - Measured: chat turn 2 reused 34 of 53 prompt tokens; the slidemaker
    shape (one system prompt, many slides) went 212 tokens/4.28s cold to
    195 reused / 11 prefilled at 0.42s; a parked context restored from host
    RAM reused 19 of 21.
  - **Finding worth keeping: chunked prefill is not bit-reproducible.** A
    layer fetches its chunk's expert *union* in waves, so regrouping tokens
    regroups the partial sums. Changing chunk size alone moves next-token
    logits 0.093 (scale 22.2); prefix reuse moves them 0.039. So `--chunk=N`
    already changes outputs today, and "bit-identical across reuse" was the
    wrong test criterion — the right one is "inside the engine's own spread,
    with the KV copy itself exact", which is what --session-selftest checks.
  - Still serial: sessions make each turn cheap but requests do not decode
    concurrently. Continuous batching over session slots (M10 item 6) is the
    throughput lever, and matters most on H2D-bound models like the 235B.

- 2026-08-17: **Serving deadlock fixed (v0.17.0).** Reported as "llmpager
  hangs when using it"; reproduced as *one request per warm model, then
  silence*. Discriminating measurement: while wedged the process burned
  **2 CPU ticks in 20s** (0.1% of a core) with GPU 0% and `read_bytes`
  flat, and 9 threads parked in `futex_wait_queue` — a deadlock, not a
  spin. (Cumulative `systemd` CPU accounting looks like spinning and is
  not evidence either way.) Sharp test: the *identical* 194-token prompt
  succeeds as request 1 and hangs as request 2, so the bug is carried
  state, not input.
  - Root cause: `Decoder::step` defers expert-handle release through an
    8-event ring, and only `step` drains it — so a generation that ends
    in decode leaves up to 8 layers' worth of slots pinned indefinitely.
    The next request opens with union prefill, which fetches each layer's
    expert union in waves sized to the *whole* layer; the first wave over
    a still-pinned layer stalls in `Pager::request` on slots held by the
    very thread doing the waiting. Tiny prompts survived because their
    per-layer union fit in the slots that were left.
  - Fix: `step_multi` flushes the deferred ring before fetching. Kimi's
    decoder was never affected (eager release on every path).
  - Guard: `Pager::request` waits a 10s grace, then errors if the layer
    is fully pinned with nothing in flight — this class of bug can no
    longer present as a silent hang.
  - Ops note: `cargo build --release` on the box builds **nothing
    relevant** — `default-members` excludes llmpager-run/-serve (they
    need nvcc), and it exits "Finished" in 1.4s. Always deploy with
    `~/deploy.sh llmpager-run llmpager-serve`, then verify the running
    binary (`strings /proc/$(systemctl show llmpager -p MainPID --value)/exe`).

- 2026-08-12: **Qwen3-235B-A22B-Instruct-2507 running on the 16GB card.**
  Converted 470.2GB bf16 (118 shards) → 121GB pack + 15.99GB core in 364s
  (94 layers × 128 experts top-8, max quant err 0.21094). `Qwen3MoeForCausalLM`,
  so the generic converter and decoder handled it with no new model code.
  - **`core_q4` had to be plumbed into `serve.json`** (`e365b21`): the runtime
    and CLI already had `--core-dtype=q4`, but `llmpager-serve` passed a
    hardcoded `false` in that position of `AnyDecoder::new`, so the HTTP path
    could not serve any model whose bf16 core exceeds VRAM. The 235B core is
    15.99GB on a 16GB card and OOMed at load.
  - **`slots=8` is a hard floor, not a tuning choice** — fewer slots than
    experts-per-token errors out (`requested 8 experts but layer has only 6
    slots`). With 94 layers the cache costs ~9.8MB × 94 per slot, so `slots=16`
    OOMs outright; there is almost no headroom above the floor.
  - Measured at a 1282-token prompt: **prefill 6.17 tok/s, decode 2.30 tok/s**,
    expert cache 6-10% hit, RAM tier 92.8% hit at `ram_gb=100` (pack is 121GB,
    box has 125GB). ~5 minutes per case for a 200-token answer.
  - **Expert-drop is inert on this model** — 0.05 and 0.10 both streamed
    byte-identical 1219.62GB. Independently reproduces the M8 finding.
  - **Diagnosis: host-to-device bandwidth bound, not disk bound.** ~7.4GB moves
    per token (94 layers × 8 experts × ~9.8MB); against the M0-measured 25.3GB/s
    pinned H2D that is a ~3.4 tok/s ceiling, and 2.30 is 68% of it. Cache and
    routing knobs cannot help; only fewer bytes per token can — lower-bit expert
    quantization (M9 K-quants) or batching, which amortizes each transfer across
    N sequences. **Batch scaling on this model was never measured** (first
    attempt passed `--batch-selftest` without `--batch`, so the cap stayed 1;
    the rerun was stopped when work was parked). That remains the open lever.
  - Site note: the WAN is ~100 Mbit and shared, so the 470GB pull took 9h54m.
    Source precision is a bandwidth decision here, not only a quality one — FP8
    (236GB) or AWQ (124GB) would cut that 2-4x if the converter learns them.

- 2026-08-06 (evening): M3 continued — RAM tier (`--direct=0`, biggest
  single win: cold prompts 20→32 tok/s, peak 37.6), batched MoE, prefill
  lm_head skip; core-q4 and cross-layer prefetch measured and rejected
  (PERFORMANCE.md). M4 shipped: llmpager-serve + systemd on ai.g8.lo:8090,
  verified over the network. Coder pack converted — two models ready.
  Releases: v0.4.0, v0.5.0, v0.6.0. Next: M5 multi-model (registry,
  budgeter), perplexity check, q4-kernel ALU work.
- 2026-08-06 (19:00): **First real tokens.** Qwen3-30B-A3B decodes coherently at
  ~19 tok/s greedy (83% cache hit) with experts paged from NVMe on the
  16GB card — M2's core goal. Fixed VRAM allocation-granularity waste
  (per-layer slot arenas). Released v0.2.0. Remaining M2: coder pack
  (downloading), perplexity check. M3 next: batched/vectorized GEMVs
  (compute-bound at 19 of ~113 tok/s paging ceiling), event-based handle
  release, prefetch in the decode loop.
- 2026-08-06: Recovered GPU passthrough on pve.g8.lo (Blackwell D3cold vfio
  bug — `vfio_pci.disable_idle_d3=1`), fixed guest DKMS kernel/header drift.
  ai.g8.lo operational. Language pivoted Python→Rust at user request before
  first commit. M0 complete: cache + pack + bench green on ai.g8.lo; results
  in docs/BENCHMARKS.md (H2D 25.3 GB/s, disk ~4 GB/s, 93% hit rate @ 48
  slots → 104 tok/s paging ceiling). M1 complete same day: llmpager-cuda
  async pager, prefetch=1 gives +42% (113 tok/s ceiling @ 48 slots). Next:
  M2 — converter (HF Qwen3-30B-A3B 4-bit → .llmpk + resident core) and the
  model runtime. Open decision for M2: custom kernels vs candle for the
  resident (non-expert) path — evaluate candle's quantized MoE support
  first; the pager API (request/wait_stream/release) is runtime-agnostic.
