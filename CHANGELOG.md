# Changelog

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
