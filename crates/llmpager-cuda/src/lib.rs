//! CUDA side of llmpager.
//!
//! [`driver`] is a minimal wrapper over the CUDA driver API, loaded from
//! `libcuda.so.1` at runtime — the driver ABI is stable and ships with the
//! GPU driver, so no CUDA toolkit is needed on the machine. [`pager`] is the
//! async expert pager: an I/O worker pool reads expert blobs from an
//! O_DIRECT-opened pack into pinned staging buffers, copies them to per-slot
//! VRAM buffers on worker streams, and publishes readiness through CUDA
//! events, so compute can wait per-expert (or make a stream wait) without
//! host round-trips.

pub mod driver;
#[cfg(feature = "kernels")]
pub mod kernels;
pub mod pager;
