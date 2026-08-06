//! Minimal CUDA driver API wrapper, loaded from libcuda.so.1 at runtime.
//!
//! The driver ABI is stable and ships with the GPU driver itself, so this
//! needs no CUDA toolkit on the machine. Only what the M0 bench requires:
//! context setup, device/pinned allocation, async H2D copies, streams.

use anyhow::{bail, Result};
use libloading::Library;

type CUresult = i32;
type CUdeviceptr = u64;
pub type CUstream = *mut std::ffi::c_void;
type CUcontext = *mut std::ffi::c_void;

macro_rules! cu {
    ($api:expr, $name:ident, $call:expr) => {{
        let rc: CUresult = $call;
        if rc != 0 {
            bail!("{} failed: CUDA error {rc}", stringify!($name));
        }
    }};
}

pub struct Cuda {
    _lib: &'static Library,
    ctx: CUcontext,
    mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    mem_host_alloc: unsafe extern "C" fn(*mut *mut u8, usize, u32) -> CUresult,
    mem_host_free: unsafe extern "C" fn(*mut u8) -> CUresult,
    memcpy_htod_async:
        unsafe extern "C" fn(CUdeviceptr, *const u8, usize, CUstream) -> CUresult,
    stream_create: unsafe extern "C" fn(*mut CUstream, u32) -> CUresult,
    stream_sync: unsafe extern "C" fn(CUstream) -> CUresult,
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

            cu!(lib, cuInit, cu_init(0));
            let mut dev = 0i32;
            cu!(lib, cuDeviceGet, device_get(&mut dev, 0));
            let mut ctx: CUcontext = std::ptr::null_mut();
            cu!(lib, cuDevicePrimaryCtxRetain, ctx_retain(&mut ctx, dev));
            cu!(lib, cuCtxSetCurrent, ctx_set(ctx));

            Ok(Self {
                ctx,
                mem_alloc: *lib.get(b"cuMemAlloc_v2")?,
                mem_free: *lib.get(b"cuMemFree_v2")?,
                mem_host_alloc: *lib.get(b"cuMemHostAlloc")?,
                mem_host_free: *lib.get(b"cuMemFreeHost")?,
                memcpy_htod_async: *lib.get(b"cuMemcpyHtoDAsync_v2")?,
                stream_create: *lib.get(b"cuStreamCreate")?,
                stream_sync: *lib.get(b"cuStreamSynchronize")?,
                ctx_sync: *lib.get(b"cuCtxSynchronize")?,
                _lib: lib,
            })
        }
    }

    /// Make this thread use the primary context (I/O worker threads need it
    /// before issuing copies; cheap to call).
    pub fn bind_thread(&self, lib_ctx_set: bool) -> Result<()> {
        if lib_ctx_set {
            unsafe {
                let ctx_set: unsafe extern "C" fn(CUcontext) -> CUresult =
                    *self._lib.get(b"cuCtxSetCurrent")?;
                cu!(self._lib, cuCtxSetCurrent, ctx_set(self.ctx));
            }
        }
        Ok(())
    }

    pub fn alloc_device(&self, len: usize) -> Result<CUdeviceptr> {
        let mut p: CUdeviceptr = 0;
        unsafe { cu!(l, cuMemAlloc, (self.mem_alloc)(&mut p, len)) };
        Ok(p)
    }

    pub fn free_device(&self, p: CUdeviceptr) {
        unsafe { (self.mem_free)(p) };
    }

    /// Page-locked host memory; page-aligned, so also valid for O_DIRECT.
    pub fn alloc_pinned(&self, len: usize) -> Result<Pinned> {
        let mut p: *mut u8 = std::ptr::null_mut();
        unsafe { cu!(l, cuMemHostAlloc, (self.mem_host_alloc)(&mut p, len, 0)) };
        Ok(Pinned { ptr: p, len })
    }

    pub fn free_pinned(&self, b: &Pinned) {
        unsafe { (self.mem_host_free)(b.ptr) };
    }

    pub fn stream(&self) -> Result<CUstream> {
        let mut s: CUstream = std::ptr::null_mut();
        unsafe { cu!(l, cuStreamCreate, (self.stream_create)(&mut s, 0)) };
        Ok(s)
    }

    pub fn htod_async(
        &self,
        dst: CUdeviceptr,
        src: &[u8],
        stream: CUstream,
    ) -> Result<()> {
        unsafe {
            cu!(
                l,
                cuMemcpyHtoDAsync,
                (self.memcpy_htod_async)(dst, src.as_ptr(), src.len(), stream)
            )
        };
        Ok(())
    }

    pub fn sync_stream(&self, s: CUstream) -> Result<()> {
        unsafe { cu!(l, cuStreamSynchronize, (self.stream_sync)(s)) };
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        unsafe { cu!(l, cuCtxSynchronize, (self.ctx_sync)()) };
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

unsafe impl Send for Pinned {}
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}
