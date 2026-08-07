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
| 08-07 | M3: event-based deferred handle release | real decode | 33.1 tok/s |
| 08-07 | M3: batched MoE launches | real decode | **34.8 tok/s** |
| 08-07 | (rejected) core q4 | real decode | 25.9 tok/s ✗ |
| 08-07 | (rejected) cross-layer prefetch | real decode (cold prompt) | 10.8 vs 18.5 tok/s ✗ |

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

### Fixed along the way

- **VRAM allocation granularity**: 3,072 individual ~2.5MB slot buffers
  rounded up to ~4MB each inside the driver — ~60% of the cache budget
  wasted, discovered as an OOM. Per-layer arena allocations fixed it.

## Appendix: the one-day timeline

Wall-clock milestones mined from the session transcript (2026-08-06,
times local). The project went from a wedged GPU and an empty repo to a
released paging engine with real tokens in about eight hours:

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
| +next session | Vectorized GEMVs → 31.1 tok/s; event-ring release → 33.1 tok/s; v0.2.0 and v0.3.0 tagged |

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
