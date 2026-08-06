# llmpager — Project Context

MoE expert-paging inference engine for Linux + NVIDIA. Runs Mixture-of-Experts
LLMs whose weights exceed VRAM by keeping only the shared core (embeddings,
attention, router, norms) plus KV cache resident on the GPU and streaming
routed experts from NVMe on demand, with an LFU cache of hot experts in VRAM.

Inspired by turbo-fieldfare (Apple Silicon / Metal / SSD streaming); this is
the Linux + CUDA equivalent.

## Version

Current: **0.1.0** (pre-1.0, API unstable)

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
- [ ] Remaining kernels: rmsnorm, rope, attention, silu-mul, router topk,
      bf16 GEMV for core; decode loop wiring core + pager
- [ ] Model runtime: attention/router resident, expert FFN via pager
- [ ] Greedy decode CLI producing real tokens on ai.g8.lo
- [ ] Perplexity sanity check vs reference

### M3 — Performance
- [ ] Overlap tuning (double-buffering, per-layer prefetch depth)
- [ ] Optional GPUDirect Storage (cuFile) path
- [ ] Speculative expert prefetch (reuse-distance / gate-estimate heuristics)
- [ ] tokens/sec + hit-rate benchmarks vs cache size

### M4 — Serving
- [ ] OpenAI-compatible HTTP endpoint
- [ ] Deployment unit (systemd) on ai.g8.lo

### M5 — Multi-model
- [ ] Global VRAM budgeter: per-model expert-cache autosizing by activity
      (busy models grow slots, idle models shrink toward zero)
- [ ] Pageable resident cores: load/unload a model's core on demand
      (~0.5s switch at 4 GB/s; N models installed, K warm)
- [ ] Model registry; serving routes `model:` to the right pager instance
- [ ] Disk-bandwidth arbitration between concurrently faulting models

## Session Log

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
