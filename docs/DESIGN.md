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

**Pin lifetime invariant.** A `Stalled` layer can only be unblocked by
someone releasing a pin, so a caller must never enter a fetch that needs
more slots than it can free itself. Decode defers release through a small
event ring (a handle stays pinned until the compute stream passes it);
union prefill fetches in waves sized to the *whole* layer and releases
eagerly. Mixing the two is the trap: a decode-ended generation leaves ring
entries pinned, and the next prefill's first wave over one of those layers
waits on slots its own thread holds — a single-threaded deadlock (fixed in
v0.17.0 by flushing the ring at `step_multi` entry). `Pager::request` now
treats "fully pinned with nothing in flight" as an error rather than
waiting forever, so any future violation reports itself instead of hanging.

### VRAM co-tenancy (M11) — `reserve_bytes`

`Pager::new` is the only place that sizes the expert arena, so it is the
only place that needs to know about the rest of the card. Given a
reserve it calls `cuMemGetInfo` and clamps `slots_per_layer` to whatever
still leaves that much free, erroring only if not even one slot per layer
fits. The clamp applies identically at warm-up and on every resize, and
`resize_cache` frees the old arena *before* building the new one, so the
free-memory reading is against genuinely free memory rather than memory
we are about to release.

Because the arena is dropped first, a failure to build the replacement
would leave the decoder with no pager at all. Both engines therefore
rebuild at the previous size — with the reserve ignored, since an
unsatisfiable reserve is exactly what tends to fail — and report the
original error. The serving layer additionally rolls the reserve back so
a bad value is not left armed for the next rebalance.

What does *not* resize: the resident core, and the KV sequence slots
(`batch_cap × max_seq`). Yielding those would mean dropping session
contexts, so the expert cache — which is reconstructible from the pack by
definition — absorbs the whole adjustment.

### Sessions (M10) — `llmpager-serve::SessionStore`

The decoder allocates `batch_cap` independent KV regions per layer and
`step_multi` takes a sequence slot per entry, so isolated contexts were
already possible; sessions add identity, persistence, and reuse on top.

- A session is a name, the tokens whose KV it holds, and where that KV
  lives. Position i of the cache holds state for `tokens[..=i]`, so any
  prompt sharing a prefix with `tokens` can skip prefilling that prefix.
- Slot 0 stays the anonymous lane: sessionless requests decode there as
  they always have, and cannot clobber a session's state.
- Two eviction tiers, matching the expert path: LRU over VRAM sequence
  slots, then LRU over a host-RAM budget. Parking is `kv_export` (D2H)
  and restoring is `kv_import` (H2D) — ~96KB/token for qwen3-30b, so a
  1000-token context is ~98MB and moves in milliseconds, against seconds
  of expert paging to prefill it again.
- Reuse is capped at `prompt.len() - 1`: the forward pass over at least
  one token is what produces the logits the reply is sampled from.
- Dropping a parked context also clears its token list, so the store can
  never claim reuse it cannot honour.

Prefill is not bit-reproducible across chunk groupings — a layer fetches
its chunk's expert *union* in waves, so regrouping the tokens regroups
the partial sums. Measured on a 130-token prompt: changing the chunk size
alone moves the next-token logits by 0.093 (scale 22.2), while prefix
reuse moves them 0.039. The KV copy itself is exact — same boundaries,
same bits — which is what `--session-selftest` asserts.

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

## Multi-model (M5)

Nothing in the pager is a singleton: each `Pager` owns its pack, VRAM slots,
cache, and io workers, so N models already coexist within the VRAM budget.
Making that first-class:

- **VRAM budgeter** — slot counts become dynamic. A control loop grows the
  active model's cache (hit rate rises with slots; see BENCHMARKS.md) and
  shrinks idle models' caches by evicting lowest-frequency unpinned slots —
  the aged-LFU structure already supports this.
- **Pageable cores** — the resident core is stored as its own artifact (the
  converter emits it separately from the pack, exactly so this works) and
  can be unloaded for cold models. Reloading ~2GB of core at NVMe speed is
  ~0.5s: model switch costs half a second, not a restart.
- **Registry + routing** — serving maps `model:` in the request to a pager
  instance; the budgeter decides which cores stay warm.
- **Shared-disk discipline** — concurrent faulting models split NVMe
  bandwidth; the io pool stays global (few, deep readers) so one model's
  miss storm can't convoy-starve another's.

## Kimi-class models (stretch)

Could a 1T-class MoE (Kimi K2: 1T total / 32B active, 61 layers, 384
experts/layer, 8+1 active) run this way on ai.g8.lo? Sizing:

- **Pack**: ~550-600GB at q4 — fits the 800GB `/data` volume (added
  2026-08-06 from the host's 990 EVO Plus; ~1.4TB still free on that NVMe
  for a K3-scale ~1.4TB pack if ever needed).
- **Per-expert blob**: ~22MB (3 × 2048×7168 q4). Worst-case per-token
  traffic: 61 layers × 9 experts × 22MB ≈ 12GB → ~3s/token at measured
  4 GB/s with zero cache hits.
- **Caching**: 16GB VRAM affords only ~300 expert slots (~1.3% of 23k
  experts); a **RAM tier** matters here — 64GB holds ~2,700 experts, and a
  RAM hit costs a memcpy + H2D (~ms) instead of a disk read. Realistic
  expectation with both tiers and routing locality: **0.5-1.5 tok/s** —
  batch/offline territory, same class as the published
  Kimi-on-tiny-GPU demos, not interactive chat.
- **Code gaps**: shared-expert support, MLA attention, the RAM mid-tier
  (VRAM → RAM → NVMe), and sparse embedding lookup (embed rows fetched
  by token id rather than resident — at 160k vocab the table alone is
  ~0.6GB q4).

Qwen3-30B-A3B remains the proving model; Kimi-class is what the M5
multi-tier work unlocks.

## Later

- io_uring instead of thread pool preads
- GPUDirect Storage (cuFile) NVMe→VRAM, skipping the host bounce
- Speculative prefetch from router logit trends
- Multi-GPU expert sharding
