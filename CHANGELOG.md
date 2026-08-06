# Changelog

## [Unreleased]

### 2026-08-06
- **chore:** Project scaffold — CLAUDE.md work plan, README, CHANGELOG,
  .gitignore, Cargo workspace (v0.1.0, Rust)
- **docs:** docs/DESIGN.md — architecture for expert paging on Linux/CUDA
- **feat:** `llmpager-core::cache` — per-layer aged-LFU ExpertCache with
  pin/refcount semantics and stall signaling
- **feat:** `llmpager-core::pack` — `.llmpk` expert pack writer/reader,
  4096-aligned blobs, O_DIRECT open path, aligned buffer helper
- **feat:** `llmpager-bench` — M0 microbenchmarks: synthetic pack generator,
  multi-threaded O_DIRECT random-read benchmark, and (behind `--features
  cuda`, via runtime-loaded libcuda driver API) pinned H2D bandwidth and an
  end-to-end paged expert-fetch benchmark with skewed routing
- **test:** Unit tests for cache eviction/pinning/decay and pack round-trip
- **docs:** docs/BENCHMARKS.md — M0 results from ai.g8.lo (disk 4.3 GB/s O_DIRECT, pinned H2D 25.3 GB/s, paged-fetch hit-rate/slots scaling table); CLAUDE.md M0 checked off
- **feat:** `llmpager-cuda` crate — CUDA driver wrapper (moved from bench, + events/stream-wait) and async Pager: io worker pool, O_DIRECT→pinned→VRAM fetch pipeline, per-slot CUDA-event readiness, condvar stall handling, best-effort prefetch, latency-histogram metrics
- **feat:** `llmpager-bench pager` subcommand — M1 benchmark of the async pager with layer-ahead prefetch
- **docs:** M1 pager benchmark results in docs/BENCHMARKS.md; M1 checked off in work plan
- **docs:** Multi-model design section (DESIGN.md) + M5 milestone; M2 marked in progress
- **feat:** `llmpager-core::quant` — q4g64 symmetric 4-bit groupwise quantization (scales-then-nibbles layout for GEMV streaming), reference dequant, error-bound tests
- **feat:** PackMeta grows optional `config` JSON for runtime model params
- **feat:** `llmpager-convert` — HF Qwen3-MoE checkpoint → q4g64 .llmpk pack + pageable resident-core safetensors; direct safetensors parsing (pread, no whole-shard loads), per-layer parallel quantization, end-to-end test with synthetic checkpoint
- **docs:** Kimi-class (1T MoE) feasibility sizing in DESIGN.md; ai.g8.lo gained 800GB /data model store
- **docs:** GEMV kernel bring-up results (correctness 5e-6, 35 GB/s naive) in BENCHMARKS.md
- **feat:** Full decode kernel set (rmsnorm, bf16 GEMV, silu-mul, add, RoPE, GQA attention, embed gather) verified on GPU vs CPU references
- **feat:** `llmpager-run` — decode runtime: core loader (bf16 matrices + f32 norms to VRAM), per-layer KV caches, host-side router top-k + greedy sampling, tokenizer via HF tokenizers, decode loop wiring kernels + pager
- **feat:** kernels: multi-row rmsnorm (per-head q/k norm), kv_append, scale_add; driver memset
- **feat:** `llmpager-convert --gen-test=DIR` — synthetic full checkpoint for GPU smoke tests
- **feat:** `llmpager-core::st` — shared minimal safetensors reader
