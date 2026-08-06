//! llmpager-core: device-agnostic building blocks for MoE expert paging.
//!
//! This crate owns no GPU memory. [`cache::ExpertCache`] is pure bookkeeping
//! that maps (layer, expert) to cache slots and picks eviction victims; the
//! CUDA layer owns the actual per-slot device buffers. [`pack`] implements the
//! `.llmpk` on-disk expert pack: 4096-byte-aligned blobs so reads can bypass
//! the page cache with O_DIRECT.

pub mod cache;
pub mod pack;
pub mod quant;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
