# llmpager — Design

## Problem

MoE models activate a small fraction of their weights per token (Gemma
26B-A4B: ~4B of 26B; Qwen3-30B-A3B: ~3B of 30B), but conventional engines
require *all* expert weights resident in GPU (or CPU) memory. turbo-fieldfare
showed on Apple Silicon that you can instead keep a small resident core in
memory and stream routed experts from SSD, caching the hot ones — because
expert selection is heavily skewed in practice.

llmpager applies the same shape to Linux + NVIDIA:

| turbo-fieldfare (macOS) | llmpager (Linux) |
|---|---|
| Metal kernels, unified memory | CUDA, discrete VRAM |
| `pread` from APFS SSD | O_DIRECT `pread` (io_uring later) from NVMe |
| 16-slot LFU per layer | configurable slots, aged LFU per layer |
| `.gturbo` pack | `.llmpk` pack |
| CPU orchestrates loads | I/O thread pool + dedicated CUDA copy stream |

## Memory budget (RTX 5060 Ti, 16GB)

Resident in VRAM: embeddings + attention + router + norms (~1-3GB for
30B-class MoE at 4-bit), KV cache (FP16, size ∝ context), and the expert
cache (slots/layer × layers × expert size). Everything else stays on NVMe.
Host RAM holds only the pinned staging ring (a few hundred MB), *not* a
second copy of the weights — O_DIRECT bypasses the page cache deliberately.

## Components

### Expert pack (`.llmpk`) — `llmpager-core::pack`

Header (4KB, JSON meta) + index (16B per expert) + 4096-aligned blobs.
Row-major (layer, expert). Alignment lets every read be O_DIRECT-legal:
aligned offset, aligned pinned buffer, aligned span. The converter (M2)
builds packs from HF checkpoints, quantizing expert FFN weights to 4-bit
groupwise; the resident core ships as a separate ordinary safetensors file.

### Expert cache — `llmpager-core::cache`

Pure bookkeeping (no GPU types): maps (layer, expert) → slot, aged-LFU
eviction, pin/refcount semantics so in-flight fetches and running forward
passes are never evicted. `Stalled` is surfaced to the caller when all slots
are pinned; the pager then blocks on an event instead of thrashing.
Frequency counters halve every N insertions — an expert hot early in a long
generation decays away instead of squatting.

### Pager (M1) — CUDA side

- One device buffer per (layer, slot): the cache's backing store.
- Pinned host ring buffer (cuMemHostAlloc → page-aligned, so directly usable
  as the O_DIRECT read target — disk → pinned is a single copy).
- I/O worker threads: `pread(O_DIRECT)` into a ring entry, then
  `cuMemcpyHtoDAsync` on the copy stream, then record a CUDA event.
- Compute stream waits on the event for each miss; hits proceed immediately.
- Prefetch: during layer L's expert FFN, layer L+1's router output is not
  yet known, but (a) sequential-layer heuristics and (b) previous-token
  reuse give useful predictions; measured in M3.

### Runtime (M2)

Decode loop with resident attention/router (either custom kernels or an
existing Rust inference stack for the non-expert path — decided in M2 after
measuring; candidate: candle). Expert FFN pulls weights through the pager.

## Benchmarks drive the design

M0's `llmpager-bench` measures the three legs on real hardware *before* any
model code exists:

1. `disk` — random-expert O_DIRECT read throughput vs thread count
2. `paged` h2d warm-up — pinned H2D bandwidth (PCIe ceiling)
3. `paged` — full miss path (cache → pread → pinned → VRAM) under a skewed
   routing distribution, reporting hit rate, ms/token, effective GB/s

Rule of thumb targets on this hardware: NVMe ~3-7 GB/s, PCIe 5.0 x8 H2D
~25 GB/s pinned. At 8 active experts × ~3MB and an 80% hit rate, a token
needs ~5MB from disk → sub-2ms fetch overhead per token if overlap works.

## Later

- io_uring instead of thread pool preads
- GPUDirect Storage (cuFile) NVMe→VRAM, skipping the host bounce
- Speculative prefetch from router logit trends
- Multi-GPU expert sharding
