# Benchmarks

## Kimi K2.6 (1T / 32B active) first light — 2026-08-08, ai.g8.lo

570.9GB expert pack (int4 g32, bit-exact QAT repack) + 23.4GB core
(q4g64-requantized at load, embeddings host-side). 16GB RTX 5060 Ti,
slots=4/layer (~400MB more VRAM would not fit), O_DIRECT, cold cache.

| Metric | Value |
|---|---|
| Decode | 0.35 tok/s |
| Prefill (22 tok, union) | 0.74 tok/s |
| Expert cache hit | 2.1% |
| Streamed for 60 tokens | 829 GB |

Output is coherent and on-topic. At a 2.1% hit rate this is the
disk-bound floor; the M8 RAM tier (~17% of experts in host RAM) and
profiling-driven pre-warm are the planned levers toward 1-2 tok/s.

### + 80GB managed RAM tier (same day)

| Config | Decode | Notes |
|---|---|---|
| No tier (baseline) | 0.35 tok/s | 60 tokens |
| --ram-gb=80, first run | **0.57 tok/s** (+63%) | 120 tokens, tier still warming |

The tier held ~3,200 experts (13%) by run end; every disk read
write-allocates, so steady-state agent workloads keep climbing toward
the 1-2 tok/s target. (The test prompt asked the model to explain
virtual memory paging. It did, correctly, while running on it.)

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

## M2 — first real-model decode — 2026-08-07, ai.g8.lo

Qwen3-30B-A3B (48 layers, 128 experts top-8), converted in 21s to a
15.43GB q4g64 pack + 3.08GB bf16 resident core. Greedy decode via
`llmpager-run`, all weights beyond the core paged from NVMe.

> "The capital of France is **Paris. The capital of the United Kingdom is
> London. The capital of Germany is Berlin. ..."**

| Metric | 48 slots | 64 slots |
|---|---|---|
| Decode tok/s | 19.9 | 19.0 |
| Expert cache hit | 83.4% | 83.9% |
| Prefill tok/s | 8.0 | 7.6 |

Notes:
- Coherent completions on factual and code prompts — converter, pack
  format, kernels, router, and pager agree end-to-end.
- Real routing is flatter than the synthetic 80/20 benchmark: going from
  48 to 64 slots buys almost no hit rate. The M3 lever is prefetch +
  faster GEMVs, not more cache.
- ~19 tok/s with naive (35 GB/s) GEMV kernels and a per-layer stream
  sync; the M1 paging ceiling at this hit rate was ~113 tok/s, so compute,
  not paging, is the current bottleneck — as expected pre-M3.
- Pager slot buffers must be per-layer arenas: individual ~2.5MB
  cuMemAllocs round up to allocation granularity and waste ~60% VRAM
  (found as OOM at 64 slots; fixed).

## M3 progress — 2026-08-07, ai.g8.lo

Same Qwen3-30B-A3B greedy decode (48 slots):

| Change | Decode tok/s |
|---|---|
| M2 baseline (scalar kernels, per-layer sync) | 19.9 |
| + vectorized GEMVs (uint/float4, uint4 bf16) | 31.1 |
| + event-based deferred handle release | **33.1** |

Kernel microbenchmarks after vectorization: q4g64 GEMV 121.7 GB/s
(was 35.4); bf16 GEMV 1.7 TB/s on an L2-resident 16.8MB matrix — i.e.
VRAM-bandwidth-bound in real use, no longer the bottleneck.

Remaining per-token costs (~30ms): 48 router dtoh round-trips, expert
misses (~10% at 48 slots), core bf16 streaming (~4.3GB/token — quantizing
the core is the biggest open win), per-expert launch overhead.

## Real-model cache sweep — 2026-08-07, ai.g8.lo (v0.4.0, O_DIRECT)

Qwen3-30B-A3B, 64 generated tokens. "capital" = warm routing (repetitive),
"haiku" = cold routing (diverse). Decode tok/s (hit rate):

| Slots/layer | VRAM cache | capital | haiku |
|---|---|---|---|
| 24 | 2.9 GB | 14.8 (71.8%) | 11.2 (57.9%) |
| 32 | 3.9 GB | 21.0 (81.4%) | 13.2 (66.6%) |
| 48 | 5.8 GB | 33.3 (89.4%) | 20.4 (78.9%) |
| 64 | 7.7 GB | 34.6 (90.4%) | 25.5 (84.6%) |

Diminishing returns above 48 slots on warm prompts, but cold prompts keep
gaining — more slots mostly help the miss-heavy workloads. The stronger
fix for misses is making them cheaper, not fewer: see the RAM tier below.

## RAM tier — 2026-08-07 (v0.5.0, `--direct=0`)

Pack (15.4GB) fully page-cache resident in 64GB host RAM; misses become
~25 GB/s memory copies instead of 4 GB/s disk reads. 48 slots:

| Prompt | O_DIRECT | RAM tier warm |
|---|---|---|
| capital | 33.3 tok/s | **37.6 tok/s** |
| haiku | 20.4 tok/s | **32.2 tok/s** |
| prefill | 7-9 tok/s | 12-17 tok/s |

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
