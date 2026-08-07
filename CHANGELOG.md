# Changelog

## [v0.11.0] — 2026-08-07

### Added
- Packaging: `.deb` (Debian/Ubuntu) and `.rpm` (Fedora/RHEL) built by
  `deploy/packaging/build-packages.sh` — /usr/bin binaries, systemd unit,
  `/etc/llmpager/serve.json` conffile, docs; glibc is the only hard
  dependency (libcuda loads at runtime). Attached to GitHub releases.
- Presentation: plain-language vocabulary slide + "in plain terms"
  callouts; PDF export (docs/presentation/llmpager.pdf)
- README: Debian/Fedora install instructions

## [v0.10.0] — 2026-08-07

### Added
- SSE streaming (`"stream": true`) on /v1/completions and
  /v1/chat/completions — OpenAI chunk format, token-by-token deltas,
  final usage + tok/s chunk, [DONE] terminator. Generation runs on a
  worker thread feeding a channel-backed chunked response.

## [v0.9.1] — 2026-08-07

### Changed
- **perf:** fp16 magic-number nibble unpack in both q4 GEMV kernels
  (122 → 137.7 GB/s); decode 37.6 → 41.0 tok/s (+106% total vs M2
  baseline). Core-q4 re-tested with the faster kernel: still rejected
  (35.2 vs 41.0)
- docs: presentation and PERFORMANCE.md updated with new numbers

## [v0.9.0] — 2026-08-07

M5 budgeter: warm-count-driven expert-cache sizing. Solo models run the
big cache; loading a second model shrinks residents automatically.
Model load into serving rotation measured at 0.8s.

### Added
- `Decoder::resize_cache` — rebuild the pager at a new slot count
- Registry rebalancing on warm-set changes; `slots_solo` config field;
  `/v1/models` reports live per-model slot allocation

## [v0.8.0] — 2026-08-07

Perplexity validation: the pipeline is numerically healthy and paging is
provably lossless.

### Added
- `llmpager-run --ppl=FILE` — teacher-forced perplexity. Measured:
  PPL 8.62 on English text, bit-identical across cache sizes (24 vs 48
  slots), q4-core quality cost quantified at +4.5% PPL
- `Decoder::last_logits()` accessor

## [v0.7.0] — 2026-08-07

M5: multi-model serving. Two 30B MoE models warm simultaneously on one
16GB GPU, routed by the OpenAI `model:` field.

### Added
- Multi-model registry in `llmpager-serve`: JSON config (`deploy/serve.json`),
  lazy warm-up, LRU eviction, evict-all retry under VRAM pressure
- Full VRAM reclaim on model unload: Pager frees its slot arenas and
  events, Decoder frees buffers/KV/core (tied-embedding safe)
- Both Qwen3 packs served at 24 slots each (13-14 tok/s per model);
  `/v1/models` reports per-model warm state

## [v0.6.0] — 2026-08-07

M4: serving. llmpager is now a network service on ai.g8.lo.

### Added
- `llmpager-serve`: OpenAI-compatible HTTP server — GET /v1/models,
  POST /v1/completions, POST /v1/chat/completions (Qwen3 ChatML
  template); greedy decode, serial requests, usage + tok/s in responses
- EOS stopping from model config (`eos_token_id`), used by CLI and server
- `deploy/llmpager.service` — systemd unit, deployed and verified
  (chat completion over the network at ~20 tok/s)
- `llmpager-run` split into lib + bin so the server reuses the Decoder

## [v0.5.0] — 2026-08-07

The RAM tier: `--direct=0` serves the pack from the OS page cache when it
fits in host RAM — VRAM misses cost a memory copy, not a disk read.
Decode 34.8 → 37.6 tok/s warm-routing, 20.4 → 32.2 tok/s cold-routing,
prefill ~2x. Slot-sweep results on the real model recorded.

### Added
- `PagerConfig.direct` / `llmpager-run --direct=0` — page-cache RAM tier
- Real-model cache sweep (24/32/48/64 slots x warm/cold prompts) in docs
- README status refreshed with current headline numbers

## [v0.4.0] — 2026-08-07

Decode 33.1 → ~34.8 tok/s; two heuristics measured and rejected with data.

### Added
- Batched MoE: one q4 GEMV launch per projection stage for all top-k
  experts (device blob-pointer array), all-expert silu-mul, weighted
  `moe_reduce` into the residual
- Prefill skips final norm / lm_head / logits readback for non-final
  prompt tokens
- Flags: `--core-dtype=q4|bf16`, `--prefetch-next=1|0` (both default to
  the measured-faster setting)
- docs/PERFORMANCE.md — the performance journey, technique by technique,
  including a session-transcript timeline appendix

### Changed
- Core q4 experiment rejected: 25.9 vs 33.1 tok/s (q4 GEMV is unpack-ALU
  bound) plus greedy drift; core stays bf16 by default
- Cross-layer speculative prefetch rejected: 18.5 → 10.8 tok/s on cold
  prompts — Qwen3 routing is uncorrelated across layers, wrong prefetch
  pollutes the cache and triples disk traffic

## [v0.3.0] — 2026-08-07

Decode 19.9 → 33.1 tok/s (+66%) on Qwen3-30B-A3B.

### Changed
- Vectorized GEMV kernels: q4g64 uses uint nibble-words + float4 x loads
  (35 → 122 GB/s); bf16 uses uint4 loads with a scalar tail (no longer
  the bandwidth limiter)
- Expert handles released via a small CUDA-event ring instead of a
  per-layer stream sync — the GPU pipeline stays deep across layers
- `gemv` bench also measures bf16 GEMV throughput

## [v0.2.0] — 2026-08-07

First working release: a real MoE model (Qwen3-30B-A3B, 18.5GB of weights)
decodes coherently on a 16GB GPU at ~19 tok/s greedy, with routed experts
streamed from NVMe through a VRAM LFU cache (83% hit rate).

### Added
- `llmpager-core`: `.llmpk` expert pack format (4096-aligned blobs,
  O_DIRECT reads); per-layer aged-LFU ExpertCache with pin/refcount
  semantics; q4g64 symmetric 4-bit groupwise quantization; minimal
  safetensors reader
- `llmpager-cuda`: runtime-loaded libcuda driver wrapper (no toolkit needed
  at runtime); async Pager — io worker pool, O_DIRECT→pinned→VRAM pipeline,
  CUDA-event readiness, condvar stall handling, prefetch, metrics; CUDA
  kernel set (q4g64 GEMV, bf16 GEMV, rmsnorm, silu-mul, add/scale-add,
  NeoX RoPE, GQA decode attention, KV append, embed gather) compiled
  .cu→PTX at build time, all verified vs CPU references
- `llmpager-convert`: HF Qwen3-MoE checkpoint → q4g64 pack + pageable
  bf16 resident core; direct safetensors parsing; parallel per-layer
  quantization (21s for the 30B model); synthetic-checkpoint generator
- `llmpager-run`: greedy decode CLI — resident core in VRAM, host-side
  router top-k, paged expert FFNs, HF tokenizer, streaming output
- `llmpager-bench`: disk/H2D/paged-fetch/pager/GEMV/kernel benchmarks
- docs: DESIGN.md (architecture, multi-model M5, Kimi-class sizing),
  BENCHMARKS.md (all measured results)

### Fixed
- Pager slot buffers allocated as per-layer arenas — individual ~2.5MB
  device allocations round up to allocation granularity and wasted ~60%
  of the cache budget (OOM at 64 slots/layer)

### Infrastructure
- ai.g8.lo: RTX 5060 Ti passthrough recovered (Blackwell D3cold vfio bug,
  `vfio_pci.disable_idle_d3=1`), nvcc 13.3, 800GB /data model store

## [Unreleased]
<!-- New unreleased changes go here -->

### 2026-08-07
- **docs:** docs/PERFORMANCE.md — running performance journey (technique → measurement), presentation source material
- **docs:** PERFORMANCE.md appendix — one-day timeline and details mined from the session transcript
- **perf:** Core q4 experiment measured and rejected (25.9 vs 33.1 tok/s; greedy drift) — kept behind --core-dtype=q4; findings in PERFORMANCE.md
- **perf:** Batched MoE launches (33.1→34.8 tok/s); prefill skips lm_head for non-final tokens; cross-layer prefetch measured and rejected (cache pollution) — details in PERFORMANCE.md

- **feat:** Qwen3-Coder-30B-A3B converted and verified — second model running through the same engine (18.6 tok/s cold-cache)

### 2026-08-07
- **docs:** Red Hat-styled HTML presentation (docs/presentation/llmpager.html) — overview, source, results, lessons learned
- **fix:** timeline corrected — ai.g8.lo was on UTC; the whole project ran in one ~5-hour session (first tokens at hour 3); VM timezone set to America/Chicago
