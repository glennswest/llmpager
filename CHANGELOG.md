# Changelog

## [Unreleased]

### 2026-08-06
- **chore:** Project scaffold — CLAUDE.md work plan, README, CHANGELOG,
  .gitignore, Cargo workspace (v0.1.0, Rust)
- **docs:** docs/DESIGN.md — architecture for expert paging on Linux/CUDA
- **feat:** `llmpager-core::cache` — per-layer aged-LFU ExpertCache with
  pin/refcount semantics and stall signaling
- **feat:** `llmpager-core::pack` — `.llmpk` expert pack writer/reader,
  4096-aligned blobs, O_DIRECT open path, aligned buffer helper
- **feat:** `llmpager-bench` — M0 microbenchmarks: synthetic pack generator,
  multi-threaded O_DIRECT random-read benchmark, and (behind `--features
  cuda`, via runtime-loaded libcuda driver API) pinned H2D bandwidth and an
  end-to-end paged expert-fetch benchmark with skewed routing
- **test:** Unit tests for cache eviction/pinning/decay and pack round-trip
