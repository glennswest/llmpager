//! Minimal CUDA driver API wrapper, loaded from libcuda.so.1 at runtime.
//!
//! Covers exactly what the pager needs: primary context, device / pinned
//! allocation, async H2D copies, streams, and events.

use anyhow::{bail, Result};
use libloading::Library;

type CUresult = i32;
pub type CUdeviceptr = u64;
pub type CUstream = *mut std::ffi::c_void;
pub type CUevent = *mut std::ffi::c_void;
type CUcontext = *mut std::ffi::c_void;

/// Skip cross-device timing bookkeeping; we use events purely for ordering.
const CU_EVENT_DISABLE_TIMING: u32 = 0x2;

macro_rules! cu {
    ($name:ident, $call:expr) => {{
        let rc: CUresult = $call;
        if rc != 0 {
            bail!("{} failed: CUDA error {rc}", stringify!($name));
        }
    }};
}

pub struct Cuda {
    lib: &'static Library,
    ctx: CUcontext,
    mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    mem_host_alloc: unsafe extern "C" fn(*mut *mut u8, usize, u32) -> CUresult,
    mem_host_free: unsafe extern "C" fn(*mut u8) -> CUresult,
    memcpy_htod_async: unsafe extern "C" fn(CUdeviceptr, *const u8, usize, CUstream) -> CUresult,
    stream_create: unsafe extern "C" fn(*mut CUstream, u32) -> CUresult,
    stream_sync: unsafe extern "C" fn(CUstream) -> CUresult,
    stream_wait_event: unsafe extern "C" fn(CUstream, CUevent, u32) -> CUresult,
    event_create: unsafe extern "C" fn(*mut CUevent, u32) -> CUresult,
    event_record: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    event_sync: unsafe extern "C" fn(CUevent) -> CUresult,
    ctx_set: unsafe extern "C" fn(CUcontext) -> CUresult,
    ctx_sync: unsafe extern "C" fn() -> CUresult,
}

impl Cuda {
    pub fn init() -> Result<Self> {
        let lib: &'static Library = Box::leak(Box::new(unsafe {
            Library::new("libcuda.so.1").or_else(|_| Library::new("libcuda.so"))?
        }));
        unsafe {
            let cu_init: unsafe extern "C" fn(u32) -> CUresult = *lib.get(b"cuInit")?;
            let device_get: unsafe extern "C" fn(*mut i32, i32) -> CUresult =
                *lib.get(b"cuDeviceGet")?;
            let ctx_retain: unsafe extern "C" fn(*mut CUcontext, i32) -> CUresult =
                *lib.get(b"cuDevicePrimaryCtxRetain")?;
            let ctx_set: unsafe extern "C" fn(CUcontext) -> CUresult =
                *lib.get(b"cuCtxSetCurrent")?;

            cu!(cuInit, cu_init(0));
            let mut dev = 0i32;
            cu!(cuDeviceGet, device_get(&mut dev, 0));
            let mut ctx: CUcontext = std::ptr::null_mut();
            cu!(cuDevicePrimaryCtxRetain, ctx_retain(&mut ctx, dev));
            cu!(cuCtxSetCurrent, ctx_set(ctx));

            Ok(Self {
                ctx,
                mem_alloc: *lib.get(b"cuMemAlloc_v2")?,
                mem_free: *lib.get(b"cuMemFree_v2")?,
                mem_host_alloc: *lib.get(b"cuMemHostAlloc")?,
                mem_host_free: *lib.get(b"cuMemFreeHost")?,
                memcpy_htod_async: *lib.get(b"cuMemcpyHtoDAsync_v2")?,
                stream_create: *lib.get(b"cuStreamCreate")?,
                stream_sync: *lib.get(b"cuStreamSynchronize")?,
                stream_wait_event: *lib.get(b"cuStreamWaitEvent")?,
                event_create: *lib.get(b"cuEventCreate")?,
                event_record: *lib.get(b"cuEventRecord")?,
                event_sync: *lib.get(b"cuEventSynchronize")?,
                ctx_set,
                ctx_sync: *lib.get(b"cuCtxSynchronize")?,
                lib,
            })
        }
    }

    /// Make the primary context current on the calling thread. Worker threads
    /// must call this once before issuing CUDA calls.
    pub fn bind_thread(&self) -> Result<()> {
        unsafe { cu!(cuCtxSetCurrent, (self.ctx_set)(self.ctx)) };
        let _ = self.lib;
        Ok(())
    }

    pub fn alloc_device(&self, len: usize) -> Result<CUdeviceptr> {
        let mut p: CUdeviceptr = 0;
        unsafe { cu!(cuMemAlloc, (self.mem_alloc)(&mut p, len)) };
        Ok(p)
    }

    pub fn free_device(&self, p: CUdeviceptr) {
        unsafe { (self.mem_free)(p) };
    }

    /// Page-locked host memory. Page-aligned, hence also O_DIRECT-legal as a
    /// read target — disk to pinned staging is a single copy.
    pub fn alloc_pinned(&self, len: usize) -> Result<Pinned> {
        let mut p: *mut u8 = std::ptr::null_mut();
        unsafe { cu!(cuMemHostAlloc, (self.mem_host_alloc)(&mut p, len, 0)) };
        Ok(Pinned { ptr: p, len })
    }

    pub fn free_pinned(&self, b: &Pinned) {
        unsafe { (self.mem_host_free)(b.ptr) };
    }

    pub fn stream(&self) -> Result<CUstream> {
        let mut s: CUstream = std::ptr::null_mut();
        unsafe { cu!(cuStreamCreate, (self.stream_create)(&mut s, 0)) };
        Ok(s)
    }

    pub fn event(&self) -> Result<CUevent> {
        let mut e: CUevent = std::ptr::null_mut();
        unsafe { cu!(cuEventCreate, (self.event_create)(&mut e, CU_EVENT_DISABLE_TIMING)) };
        Ok(e)
    }

    pub fn htod_async(&self, dst: CUdeviceptr, src: &[u8], stream: CUstream) -> Result<()> {
        unsafe {
            cu!(
                cuMemcpyHtoDAsync,
                (self.memcpy_htod_async)(dst, src.as_ptr(), src.len(), stream)
            )
        };
        Ok(())
    }

    pub fn record_event(&self, e: CUevent, s: CUstream) -> Result<()> {
        unsafe { cu!(cuEventRecord, (self.event_record)(e, s)) };
        Ok(())
    }

    /// Host blocks until the event has completed.
    pub fn sync_event(&self, e: CUevent) -> Result<()> {
        unsafe { cu!(cuEventSynchronize, (self.event_sync)(e)) };
        Ok(())
    }

    /// All future work on `s` waits for `e` — device-side, no host block.
    pub fn stream_wait_event(&self, s: CUstream, e: CUevent) -> Result<()> {
        unsafe { cu!(cuStreamWaitEvent, (self.stream_wait_event)(s, e, 0)) };
        Ok(())
    }

    pub fn sync_stream(&self, s: CUstream) -> Result<()> {
        unsafe { cu!(cuStreamSynchronize, (self.stream_sync)(s)) };
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        unsafe { cu!(cuCtxSynchronize, (self.ctx_sync)()) };
        Ok(())
    }
}

pub struct Pinned {
    ptr: *mut u8,
    len: usize,
}

impl Pinned {
    pub fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    pub fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

// Safety: raw handles are plain values; the driver API is thread-safe once a
// context is current on the calling thread (bind_thread).
unsafe impl Send for Pinned {}
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}
