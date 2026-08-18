# Performance Journey

The running story of how llmpager got faster, one technique at a time.
Every number was measured on the same hardware in one ~5-hour session; each section explains the
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
| 08-06 | M2: first real decode (scalar kernels) | real decode | 19.9 tok/s |
| 08-06 | M3: vectorized GEMV kernels | real decode | 31.1 tok/s |
| 08-06 | M3: event-based deferred handle release | real decode | 33.1 tok/s |
| 08-06 | M3: batched MoE launches | real decode | 34.8 tok/s |
| 08-06 | (rejected) core q4 | real decode | 25.9 tok/s ✗ |
| 08-06 | (rejected) cross-layer prefetch | real decode (cold prompt) | 10.8 vs 18.5 tok/s ✗ |
| 08-06 | M3: RAM tier (page cache, `--direct=0`) | real decode | 37.6 tok/s (32.2 on cold prompts) |
| 08-06 | M3: fp16 magic-number nibble unpack | real decode | **41.0 tok/s** |

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

### 7. Core q4 quantization (M3) — rejected, kept the data

Hypothesis: the resident core streams ~4.3GB bf16 per token; re-quantizing
it to q4 at load (→ ~1.1GB) should be the biggest remaining win.

Measured: **25.9 tok/s — slower than the 33.1 bf16 baseline**, and the
greedy decode path drifted (first token changed) from double quantization.
Two lessons:

1. Bandwidth ratios only convert to speed if the kernels are equally
   efficient: our q4 GEMV sustains 122 GB/s (nibble-unpack ALU bound)
   while the bf16 GEMV runs at near-VRAM bandwidth — 1.1GB at 122 GB/s
   loses to 4.3GB at ~400 GB/s.
2. Core weights are quality-critical (attention + lm_head touch every
   logit); experts tolerate q4 far better than the shared trunk does.

Kept behind `--core-dtype=q4`; becomes interesting again only if the q4
kernel reaches ~300+ GB/s (half2 math, dual-issue unpack — future work).

### 8. Batched MoE launches (M3) — +5-9%

Each layer's 8 experts ran as 24+ serialized small GEMVs (768 rows each —
a fraction of the GPU). Now one launch per projection stage covers all
top-k experts (`grid.y` = expert, device array of blob addresses), the
silu-mul runs over all experts at once, and a `moe_reduce` kernel does the
weighted accumulate. 33.1 → **34.8 tok/s**, byte-identical output.

### 9. Prefill lm_head skip

Prefill computed the full 620MB lm_head projection for every prompt token
and discarded it; now only the last prompt token pays. Measured effect was
small — which itself was the finding: prefill is **cold-cache-bound**
(every early token faults most of its experts), not compute-bound. Prefill
optimization is a paging problem, not a kernel problem.

### 10. Cross-layer speculative prefetch — rejected, kept the data

Heuristic: prefetch layer L+1's experts using layer L's routed ids.
Measured on a cold prompt: **18.5 → 10.8 tok/s**, hit rate 77% → 67%,
disk traffic 12.6 → 35.3 GB. Qwen3-MoE's expert choice is effectively
uncorrelated across layers, so ~90% of prefetches were wrong — and wrong
prefetches are worse than nothing because they evict genuinely hot
entries and saturate the disk. Lesson: in a paging system, *bad prefetch
is cache pollution plus bandwidth theft*; the temporal reuse the LFU
already captures (same layer, recent tokens) is the signal that works.
`--prefetch-next=1` keeps the experiment reproducible.

### 11. The RAM tier (M3/M5) — +13-58%, biggest single decode win

"Do we need the memory tier?" Measured answer: yes, decisively — *when
the pack fits in host RAM*. The 30B pack is 15.4GB; the box has 64GB.
O_DIRECT (correct for huge packs) was deliberately starving a resource we
had: with `--direct=0` the OS page cache becomes a full RAM tier, and a
VRAM miss costs a ~25 GB/s memory copy instead of a 4 GB/s disk read.

| Prompt class | disk-backed | RAM tier (warm) |
|---|---|---|
| cold routing (haiku) | 20.4 tok/s | **32.2 tok/s** |
| warm routing (capitals) | 33.3 tok/s | **37.6 tok/s** |
| prefill | 7-9 tok/s | **12-17 tok/s** |

The hierarchy is now VRAM cache (hits, free) → RAM (misses, ~1ms) →
NVMe (first touch only). Cold prompts — precisely the ones the VRAM
cache handles worst — gain the most, because every miss got 6× cheaper.
For Kimi-class packs (550GB ≫ RAM) this becomes the explicit pinned
partial RAM tier in the M5 design; for anything that fits in RAM, the
page cache already does the job with zero code.

### 12. Perplexity validation — paging is provably lossless

Teacher-forced NLL over a 137-token English paragraph (`--ppl`):

| Config | Perplexity |
|---|---|
| 48 slots, bf16 core | **8.6199** |
| 24 slots, bf16 core | **8.6199** (identical to 4 decimals) |
| 48 slots, q4 core | 9.0038 (+4.5%) |

The cache-size invariance is the architecture's correctness proof: the
expert cache changes *when* weights are fetched, never *what* they are.
The q4-core row puts a number on the earlier rejection (+4.5% PPL) — and
adds a nuance: in teacher-forced mode q4-core is *faster* (37.5 vs 33.6
tok/s) because the full lm_head runs every token and its 4x smaller
matrix dominates; in normal decode the trade reverses. Same weights,
different workload, opposite conclusion — measure the workload you ship.

### 13. Multi-model + VRAM budgeter (M5)

Two 30B-A3B models (36.9GB of weights combined) serve simultaneously from
16GB of VRAM, routed by the OpenAI `model:` field. The budgeter divides
the expert-cache budget by warm count: a solo model runs 48 slots
(~33 tok/s class), and when a second model warms, residents shrink to
24/24 (~14 tok/s each). Cache resize is a pager rebuild — measured 0.8s
to load a whole 18.5GB model into serving rotation, because only the
3.1GB core actually moves (experts page in on demand). Perplexity
invariance (technique 12) is what makes resizing free of quality risk.

### 14. fp16 magic-number nibble unpack (M3) — +9%

The q4 GEMV was ALU-bound on dequantization (~5 instructions per weight:
shift, mask, cast, subtract, FMA). The classic trick: OR each nibble into
an fp16's mantissa at exponent 1024 (`| 0x6400`), subtract 1032
(1024 + the zero-point 8) in `half2` — two exact dequantized values per
three ALU ops. Kernel: 122 → 137.7 GB/s; decode: 37.6 → **41.0 tok/s**.
Re-testing core-q4 with the faster kernel: still loses (35.2 vs 41.0) —
the rejection survives its own fix. Total arc: 19.9 → 41.0 (+106%).

### 15. Global VRAM slot pool (M8 item 2) — better cache, slower engine

One shared slot pool for every layer instead of a private array each.
Blob size is uniform, so any slot fits any expert and layers with diffuse
routing can claim more of them. `LLMPAGER_GLOBAL_POOL=1`; **opt-in, and
staying that way.**

The cache half of the premise holds, and reproducibly. qwen3-30b-a3b and
qwen3-coder-30b-a3b, 100-token prompt, 60 tokens generated:

| model | slots | hit (per-layer -> global) | streamed |
|-------|-------|---------------------------|----------|
| 30b | 24 | 51.3% -> **54.3%** | 39.70 -> 37.21 GB |
| 30b | 8 | 27.4% -> 27.6% | 59.14 -> 58.98 GB |
| coder | 24 | 53.3% -> **59.3%** | 36.70 -> 32.02 GB |
| coder | 8 | 24.8% -> 25.6% | 59.15 -> 58.47 GB |

Perplexity is identical (18.7649), so it is lossless.

**And it is still slower.** Under `--direct=1`, where wall-clock on this
box is stable to about 1%, the engine loses roughly 10% despite moving
6% fewer bytes:

| run | per-layer | global |
|-----|-----------|--------|
| 1 | 12.04 tok/s | 10.45 tok/s |
| 2 | 11.93 tok/s | 10.93 tok/s |

Something in the shared pool costs more than the fetches it saves. Two
candidates, neither confirmed: eviction scans the whole pool for the
lowest-frequency unpinned slot — 1152 slots instead of 24, inside the
mutex the I/O workers need to publish fills — and one large residency
map replaces 48 small ones. Profile before touching it again; a sampled
or bucketed victim search would test the first directly.

**Two measurement lessons, both learned the hard way here.**

*The first A/B said the pool was clearly worse* (45.2% vs 51.3% hit). It
was running a decay constant picked by guess. The idea was fine; the
parameter was wrong — worth remembering before writing off a design.

*Then three separate timing batches disagreed about the winner.* Without
`--direct`, generation timing on this box spans ±20% run to run: the same
per-layer baseline measured 9.53, 9.97, 11.47, 9.29, 11.50 and 12.33
tok/s. A 6% effect is invisible in that. **v0.20.0 shipped a "+14%
decode" claim built on three paired runs of exactly this noise; it did
not survive being measured properly and is withdrawn.** Hit rate and
bytes streamed are exactly reproducible and are what tuning should use;
for wall-clock, use `--direct=1` and repeat.

**The decay cadence had a cliff, and the unit was why.** Aging counted
insertions, so one constant meant a different real cadence for every
layer count, cache size and miss rate — the setting that gave 54.9% at
slots=24 collapsed to 2.1% at slots=8. `Pager::tick()` now ages the pool
every N forward passes (default 4, `LLMPAGER_POOL_DECAY_TOKENS`), and
the response curve is smooth and unimodal at both sizes:

| decay | hit @ slots=24 | hit @ slots=8 |
|-------|----------------|---------------|
| 1 token | 36.2% | 25.7% |
| 2 tokens | 50.7% | **28.0%** |
| **4 tokens** | **54.3%** | 27.6% |
| 8 tokens | 52.7% | 21.5% |
| 16 tokens | 41.6% | 15.5% |

Worth keeping even though the feature is off: when a tuning constant is
cliff-edged, suspect the unit before adding a lookup table.

### Fixed along the way

- **VRAM allocation granularity**: 3,072 individual ~2.5MB slot buffers
  rounded up to ~4MB each inside the driver — ~60% of the cache budget
  wasted, discovered as an OOM. Per-layer arena allocations fixed it.

## Appendix: the one-day timeline

Wall-clock milestones mined from the session transcript (2026-08-06,
times local). The project went from a wedged GPU and an empty repo to a
released paging engine with real tokens in under three hours:

| Time | Event |
|---|---|
| 16:04 | Session start: empty repo, ai.g8.lo unreachable |
| 16:11 | VM start fails: `vfio ... error getting device from group 13` — GPU wedged, then falls off the PCI bus entirely during recovery attempts |
| 16:19 | Host reboot #1 — failure reproduces from clean boot (not a transient) |
| 16:24 | Community search matches the exact signature: Blackwell D3cold vfio bug (Proxmox #7374); fix is `vfio_pci.disable_idle_d3=1` |
| 16:30 | Reboot #2 with fix baked into initramfs: GPU holds D0 |
| 16:33 | `nvidia-smi` healthy inside the guest (after fixing DKMS kernel/header drift) |
| 16:57 | M0 measured: disk 4.34 GB/s O_DIRECT, pinned H2D 25.3 GB/s, 104 tok/s paging ceiling @ 48 slots |
| 17:11 | M1 async pager: prefetch=1 → +42%, 113 tok/s ceiling |
| 17:45 | First custom CUDA kernel verified on Blackwell (q4g64 GEMV, rel err 5.2e-6, 35 GB/s scalar) |
| 18:02 | Full decode kernel set verified (7 kernels, worst rel err 1.9e-6) |
| 19:00 | Qwen3-30B-A3B converted in 21s; **first real tokens: "…Paris."** at 19.9 tok/s |
| later that evening | Vectorized GEMVs → 31.1 tok/s; event-ring release → 33.1 tok/s; v0.2.0 and v0.3.0 tagged |

Presentation-worthy details preserved in the transcript:
- The GPU "pending transaction" FLR timeouts that *looked* like a wedged
  card were actually the D3cold power state making config space
  unreachable — the same fix cured both symptoms.
- M0's paged-fetch latency histogram: >99% of 3MB expert fetches
  complete in <2ms; no tail beyond 5ms in any run all day.
- 8 io threads through one virtio-scsi queue measured *slower* than 1
  thread (3.60 vs 4.34 GB/s) — queue contention, which shaped the
  pager's "few deep readers" design.

## Next levers (measured, not guessed)

Per-token cost model at 33 tok/s (~30ms/token):

| Cost | Estimate | Planned fix |
|---|---|---|
| Core bf16 streaming (~4.3GB/token) | ~12-15ms | quantize core to q4 (→ ~1.1GB) |
| Router host round-trips (48/token) | ~3-5ms | GPU top-k or batched readback |
| Expert misses (~11-17% of 384 fetches) | ~3-8ms | speculative prefetch |
| Per-expert launch overhead (~1.2k launches) | ~3-6ms | batched per-layer MoE launches |
