//! Compiled-in PTX kernels (built by build.rs with nvcc; `kernels` feature).

use anyhow::Result;

use crate::driver::{CUdeviceptr, CUfunction, CUstream, Cuda};

pub const Q4G64_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/q4g64.ptx"));

pub struct Kernels {
    q4g64_gemv: CUfunction,
}

impl Kernels {
    pub fn load(cuda: &Cuda) -> Result<Self> {
        let module = cuda.module_from_ptx(Q4G64_PTX)?;
        Ok(Self { q4g64_gemv: cuda.function(module, "q4g64_gemv")? })
    }

    /// y[rows] = W x, W a q4g64 blob region (scales then nibbles).
    pub fn q4g64_gemv(
        &self,
        cuda: &Cuda,
        blob: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        rows: i32,
        cols: i32,
        stream: CUstream,
    ) -> Result<()> {
        const WARPS_PER_BLOCK: u32 = 4;
        let grid = (rows as u32).div_ceil(WARPS_PER_BLOCK);
        let mut a_blob = blob;
        let mut a_x = x;
        let mut a_y = y;
        let mut a_rows = rows;
        let mut a_cols = cols;
        let mut params = [
            &mut a_blob as *mut _ as *mut std::ffi::c_void,
            &mut a_x as *mut _ as *mut std::ffi::c_void,
            &mut a_y as *mut _ as *mut std::ffi::c_void,
            &mut a_rows as *mut _ as *mut std::ffi::c_void,
            &mut a_cols as *mut _ as *mut std::ffi::c_void,
        ];
        cuda.launch(
            self.q4g64_gemv,
            (grid, 1, 1),
            (WARPS_PER_BLOCK * 32, 1, 1),
            &mut params,
            stream,
        )
    }
}
