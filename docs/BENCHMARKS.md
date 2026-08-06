# Benchmarks

## M0 — 2026-08-06, ai.g8.lo

Hardware: RTX 5060 Ti 16GB (vfio passthrough), 12 vCPU, 64GB RAM, virtio-scsi
disk on NVMe-backed LVM. Driver 610.43.02 / CUDA 13.3, Debian 13, Rust 1.96.

Synthetic pack: 24 layers × 64 experts × 3.0MB blobs (~4.6GB) — Qwen3-30B-A3B-ish
expert geometry. All reads O_DIRECT.

### Raw legs

| Leg | Result |
|---|---|
| Pack write (buffered) | 2.75 GB/s |
| Disk random expert read, 1 thread | 4.34 GB/s (0.69 ms/blob) |
| Disk random expert read, 8 threads | 3.60 GB/s (aggregate) |
| Pinned H2D copy (256MB × 20) | **25.32 GB/s** |

Notes: single-threaded O_DIRECT already saturates the virtio-scsi path;
8 threads slightly degrade aggregate throughput (queue contention through
the single virtio-scsi device). H2D at 25.3 GB/s is near PCIe line rate for
this slot — passthrough costs us nothing on the copy leg.

### End-to-end paged fetch (cache → O_DIRECT pread → pinned → VRAM)

500 tokens, top-8 of 64 experts/layer × 24 layers, routing skew: 80% of picks
from a 24-expert hot set. `tok/s` is the decode-loop ceiling from paging alone
(no model compute); real decode overlaps compute with fetch.

| Slots/layer | VRAM cache | Hit rate | ms/token fetch | tok/s ceiling |
|---|---|---|---|---|
| 16 | 1.2 GB | 47.5% | 55.8 | 17.9 |
| 32 | 2.3 GB | 87.9% | 17.5 | 57.0 |
| 48 | 3.5 GB | 93.4% | 9.6 | 104.5 |

0 stalls in all runs.

## M1 — 2026-08-06, ai.g8.lo (async pager)

Same pack and routing skew as M0, through `llmpager-cuda`'s async Pager
(4 io threads). `pager --prefetch=N` fetches layer L+N's experts while layer
L runs, using whole-token routing as a perfect predictor (real models will
use router heuristics, M3). Hit-rate rows with prefetch are inflated by the
prefetch probes themselves — compare tok/s and wait ms/token, not hit rate.

| Config | Wait ms/token | tok/s ceiling |
|---|---|---|
| 32 slots, prefetch 0 | 18.0 | 55.4 |
| 32 slots, prefetch 1 | 12.7 | 78.4 |
| 32 slots, prefetch 2 | 11.9 | 83.2 |
| 48 slots, prefetch 1 | **8.8** | **113.2** |

- prefetch 0 matches the M0 synchronous loop (55 vs 57 tok/s) — the pager's
  bookkeeping adds nothing measurable.
- One layer of prefetch buys +42% with zero extra bandwidth; depth 2 adds
  little more. In a real decode the gain grows: compute per layer widens the
  overlap window and prefetch waits vanish entirely once fetch < compute.
- Fetch latency distribution is tight: >99% of fetches complete under 2ms
  at prefetch 0 (0.7ms read + 0.12ms H2D); no tail beyond 5ms in any run.

## M2 kernel bring-up — 2026-08-06, ai.g8.lo

`q4g64_gemv` (warp-per-row, correctness-first), compute_80 PTX JIT'd by the
driver onto Blackwell (sm_120). Verified against the CPU dequant reference.

| Shape (rows×cols) | Worst rel err | us/launch | Weight throughput |
|---|---|---|---|
| 768×2048 (gate/up) | 5.2e-6 | 23.6 | 35.4 GB/s |
| 2048×768 (down) | 1.8e-6 | 22.3 | 37.5 GB/s |

Read-through: 35 GB/s is ~8% of this card's VRAM bandwidth and the launch
cost (~23us) dominates at expert-sized matrices — a naive-kernel result,
as expected. Decode implications: 48 layers × 8 experts × 3 GEMVs = 1,152
launches ≈ 27ms/token if launched individually. M3 priorities are therefore
(1) one launch per layer batching all top-k experts' three projections and
(2) vectorized loads / half2 math to approach memory bandwidth. Correctness
and the PTX toolchain (nvcc → PTX → driver JIT via runtime-loaded libcuda)
are proven.

### Read-through for the design

- The miss path costs ~0.7-1ms per 3MB expert (read) + ~0.12ms (H2D) —
  fetch is disk-bound, as expected. Priorities for M1/M3: hit rate and
  read parallelism, not the copy engine.
- Cache size is the dominant lever: tripling slots turned a 47% hit rate
  into 93% and a 6× better ceiling. On 16GB VRAM there is room for a
  40-50-slot cache per layer at this expert size alongside core + KV.
- 8 io threads through one virtio-scsi queue don't beat 1 thread on raw
  bandwidth; M1 should keep io parallelism modest and consider io_uring.
  If disk becomes the wall later, NVMe passthrough or multiqueue
  virtio would recover host-native performance.
