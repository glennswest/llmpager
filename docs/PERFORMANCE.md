# Performance Journey

The running story of how llmpager got faster, one technique at a time.
Every number was measured on the same hardware; each section explains the
technique, why it works, and what it bought. BENCHMARKS.md holds the raw
result tables; this file is the narrative.

## Hardware and its limits (ai.g8.lo)

| Resource | Measured / spec | Why it matters |
|---|---|---|
| RTX 5060 Ti VRAM bandwidth | ~448 GB/s spec | ceiling for GEMV-bound decode |
| PCIe (pinned H2D) | **25.3 GB/s** measured | expert upload leg |
| NVMe via virtio-scsi (O_DIRECT) | **4.3 GB/s** measured | expert fetch leg |
| VRAM | 16 GB | must hold core + KV + expert cache |

The model that drives everything: **Qwen3-30B-A3B** — 18.5GB of q4 weights
(15.4GB experts + 3.1GB core), which does not fit in 16GB of VRAM. llmpager
runs it anyway by keeping the core resident and paging the 8-of-128 routed
experts per layer from NVMe through a VRAM LFU cache.

## The scoreboard

| Date | Change | Metric | Value |
|---|---|---|---|
| 08-06 | M0: synchronous paged fetch, 48 slots | paging ceiling (no compute) | 104 tok/s |
| 08-06 | M1: async pager + 1-layer prefetch | paging ceiling (no compute) | **113 tok/s** |
| 08-07 | M2: first real decode (scalar kernels) | real decode | 19.9 tok/s |
| 08-07 | M3: vectorized GEMV kernels | real decode | 31.1 tok/s |
| 08-07 | M3: event-based deferred handle release | real decode | **33.1 tok/s** |

The gap between 33 tok/s real decode and the 113 tok/s paging ceiling is
the remaining M3 headroom — the pager can already feed experts 3× faster
than compute consumes them.

## Techniques

### 1. O_DIRECT + aligned everything (M0)

Expert reads bypass the page cache: weights already cached in VRAM must
not be cached again in host RAM. The `.llmpk` pack aligns every blob to
4096B, and pinned staging buffers are page-aligned by construction, so
disk → pinned → VRAM is two copies with zero rebuffering.
**Result: 4.3 GB/s from NVMe, single reader thread.**

### 2. Aged-LFU expert cache (M0)

Real MoE routing is skewed but drifts. Pure LFU pins yesterday's hot
experts forever; aging (halving counters every N insertions) lets the
cache track the workload. Cache size is the dominant paging lever:

| slots/layer | hit rate (synthetic 80/20) | paging ceiling |
|---|---|---|
| 16 | 47.5% | 17.9 tok/s |
| 32 | 87.9% | 57.0 tok/s |
| 48 | 93.4% | 104.5 tok/s |

Real-model routing turned out flatter: 48 slots → 83-89%, and 64 slots
buys almost nothing more (83.9%). Cache alone can't close the gap — the
rest must come from overlap and compute speed.

### 3. Async pager + prefetch (M1)

I/O workers own O_DIRECT fds and pinned buffers; misses flow
disk → pinned → `cuMemcpyHtoDAsync` on worker streams; readiness is a CUDA
event per slot, so compute never host-blocks on a hit. Prefetching just
one layer ahead overlapped fetch with "compute" for **+42%** on the paging
ceiling (55 → 78 tok/s at 32 slots) with zero extra bandwidth.

### 4. Getting real tokens before optimizing (M2)

Correctness-first scalar kernels, verified against CPU references
(worst rel err < 6e-6), per-layer stream syncs everywhere. 19.9 tok/s —
slow, but *provably correct*, which made every later optimization a pure
perf diff against known-good output.

### 5. Vectorized GEMV (M3) — +56%

Both GEMV kernels were bandwidth-bound on scalar loads (~35 GB/s).
Rewriting the inner loops to 16-byte vector loads — `uint` nibble-words +
`float4` activations for q4g64, `uint4` (8 weights/load) for bf16 —
took q4g64 to **122 GB/s** and bf16 to effective VRAM bandwidth.
Decode: 19.9 → 31.1 tok/s. Lesson: on modern GPUs the memory system,
not the ALUs, sets GEMV speed; issue the widest loads you can.

### 6. Event-ring handle release (M3) — +6%

Expert cache slots stay pinned while kernels read them, and releasing
used to cost a full stream sync per layer (48 pipeline drains per token).
Now a CUDA event is recorded after each layer's expert kernels and
handles are released lazily once the stream passes it — the pipeline
stays deep. Decode: 31.1 → 33.1 tok/s.

### Fixed along the way

- **VRAM allocation granularity**: 3,072 individual ~2.5MB slot buffers
  rounded up to ~4MB each inside the driver — ~60% of the cache budget
  wasted, discovered as an OOM. Per-layer arena allocations fixed it.

## Next levers (measured, not guessed)

Per-token cost model at 33 tok/s (~30ms/token):

| Cost | Estimate | Planned fix |
|---|---|---|
| Core bf16 streaming (~4.3GB/token) | ~12-15ms | quantize core to q4 (→ ~1.1GB) |
| Router host round-trips (48/token) | ~3-5ms | GPU top-k or batched readback |
| Expert misses (~11-17% of 384 fetches) | ~3-8ms | speculative prefetch |
| Per-expert launch overhead (~1.2k launches) | ~3-6ms | batched per-layer MoE launches |
