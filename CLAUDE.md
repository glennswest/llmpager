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
- [ ] Repo scaffold: CLAUDE.md, README, CHANGELOG, .gitignore, pyproject
- [ ] Create GitHub repo, first push
- [ ] Core data structures: ExpertCache (aged LFU), pack format read/write
- [ ] Unit tests (pure Python/numpy, runnable on Mac without CUDA)
- [ ] Bench script: NVMe O_DIRECT read bandwidth, pinned H2D bandwidth,
      end-to-end paged-fetch latency on ai.g8.lo
- [ ] Run M0 bench on ai.g8.lo, record numbers in docs/BENCHMARKS.md

### M1 — Paging core, GPU-proven
- [ ] Pinned host ring buffer + copy-stream pipeline (torch CUDA)
- [ ] Async Pager: miss → pread → H2D → event; hit → tensor handle
- [ ] Prefetch hook (issue fetches for layer L+1 while L computes)
- [ ] Cache/pager integration test on ai.g8.lo with synthetic experts
- [ ] Metrics: hit rate, bytes/token, fetch latency histogram

### M2 — Real model end-to-end
- [ ] Converter: HF Qwen3-30B-A3B (4-bit) → .llmpk expert pack + resident core
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

## Session Log

- 2026-08-06: Recovered GPU passthrough on pve.g8.lo (Blackwell D3cold vfio
  bug — `vfio_pci.disable_idle_d3=1`), fixed guest DKMS kernel/header drift.
  ai.g8.lo operational. M0 in progress: scaffolding repo, then cache + pack +
  tests, then bench on ai.g8.lo.
