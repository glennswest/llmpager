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

Working (`0.x`, API unstable). On an RTX 5060 Ti 16GB, Qwen3-30B-A3B
(18.5GB of q4 weights) decodes coherently at **~41 tok/s** greedy with a
48-slot/layer expert cache (89% hit rate) and the pack served from a RAM
tier; Qwen3-Coder-30B-A3B runs as a second model through the same engine.
The performance journey — every technique, measurement, and rejected
experiment — is in `docs/PERFORMANCE.md`; architecture in
`docs/DESIGN.md`; work plan in `CLAUDE.md`.

## Requirements

- Linux, NVIDIA GPU (tested: RTX 5060 Ti 16GB, driver 610.x, CUDA 13)
- Rust 1.80+
- NVMe SSD for the expert pack

The CUDA path loads `libcuda.so.1` at runtime via the stable driver API — no
CUDA toolkit install needed, just the GPU driver. `llmpager-core` (pack format,
cache) has no GPU dependency at all, so its tests run on any machine,
including macOS.

## Install (Debian / Fedora)

Prebuilt x86_64 packages ship with each GitHub release — the binaries load
`libcuda.so.1` at runtime, so one build serves both families and no CUDA
toolkit is needed:

```bash
# Debian / Ubuntu
sudo apt install ./llmpager_<version>_amd64.deb

# Fedora / RHEL
sudo dnf install ./llmpager-<version>-1.x86_64.rpm
```

Then convert a model, point the config at it, and start the service:

```bash
llmpager-convert --model-dir=<hf-checkpoint> \
  --out-pack=/var/lib/llmpager/m.llmpk \
  --out-core=/var/lib/llmpager/m.core.safetensors
sudo edit /etc/llmpager/serve.json     # name, pack, core, tokenizer paths
sudo systemctl enable --now llmpager   # OpenAI-compatible API on :8090
```

Packages are reproducible from source: `deploy/packaging/build-packages.sh`.

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
