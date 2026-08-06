# llmpager

**MoE expert-paging inference engine for Linux + NVIDIA.**

Run Mixture-of-Experts LLMs whose weights don't fit in VRAM. llmpager keeps
only the model's shared core — embeddings, attention, router, norms — and the
KV cache resident on the GPU, and streams the routed experts from NVMe on
demand. A configurable per-layer LFU cache holds the hottest experts in VRAM,
so most tokens hit cache and the SSD only sees the misses.

The same idea as [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare)
(which does this on Apple Silicon with Metal + SSD streaming), rebuilt for
Linux hosts with NVIDIA GPUs and CUDA.

## How it works

```
 token → router (resident, VRAM)
            │ top-k expert ids for layer L
            ▼
      ExpertCache[L]  ──hit──► expert weights already in VRAM
            │miss
            ▼
      Pager: O_DIRECT pread (NVMe) → pinned host ring → cudaMemcpyAsync
            │                                   (copy stream, overlapped
            ▼                                    with attention compute)
      cache slot filled, CUDA event signals compute stream
```

- **Expert pack (`.llmpk`)** — on-disk format with 4096-byte-aligned blobs per
  (layer, expert), built once from a Hugging Face checkpoint.
- **Aged-LFU cache** — per-layer slots in VRAM; frequency counters decay so
  yesterday's hot experts don't pin the cache.
- **Async pager** — I/O thread pool + pinned ring buffer + dedicated CUDA copy
  stream; fetches for layer L+1 overlap compute of layer L.

## Status

Early development (`0.x`). See `CLAUDE.md` for the work plan and
`docs/DESIGN.md` for the architecture.

## Requirements

- Linux, NVIDIA GPU (tested: RTX 5060 Ti 16GB, driver 610.x, CUDA 13)
- Rust 1.80+
- NVMe SSD for the expert pack

The CUDA path loads `libcuda.so.1` at runtime via the stable driver API — no
CUDA toolkit install needed, just the GPU driver. `llmpager-core` (pack format,
cache) has no GPU dependency at all, so its tests run on any machine,
including macOS.

## Development

```bash
cargo test                 # core: cache + pack format (any OS)
```

Benchmarks (on the GPU host):

```bash
cargo build --release -p llmpager-bench --features cuda
target/release/llmpager-bench gen  --path=/data/bench.llmpk --layers=24 --experts=64 --bytes=3000000
target/release/llmpager-bench disk --path=/data/bench.llmpk --threads=8
target/release/llmpager-bench paged --path=/data/bench.llmpk --slots=16 --topk=8
```

## License

Apache-2.0
