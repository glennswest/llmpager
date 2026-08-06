//! q4g64 GEMV kernel bring-up: quantize a random matrix on the CPU, run the
//! GPU kernel, compare against the CPU dequant reference, then measure
//! sustained throughput (weight bytes/s — the number that bounds expert FFN
//! decode speed).

use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Result};
use llmpager_core::quant::{q4g64_bytes, q4g64_dequantize, q4g64_quantize};
use llmpager_cuda::driver::Cuda;
use llmpager_cuda::kernels::Kernels;

use crate::{Flags, Rng};

pub fn run(f: &Flags) -> Result<()> {
    let rows = f.num("rows", 768) as usize;
    let cols = f.num("cols", 2048) as usize;
    let iters = f.num("iters", 2000) as usize;

    println!("gemv: {rows}x{cols} q4g64, {iters} iters");
    let mut rng = Rng::new(11);
    let w: Vec<f32> = (0..rows * cols).map(|_| (rng.unit() as f32 - 0.5) * 0.2).collect();
    let x: Vec<f32> = (0..cols).map(|_| (rng.unit() as f32 - 0.5) * 2.0).collect();

    let mut blob = vec![0u8; q4g64_bytes(rows, cols)];
    q4g64_quantize(&w, rows, cols, &mut blob)?;

    // CPU reference from the dequantized weights (so quantization error
    // itself doesn't count against the kernel).
    let mut wd = vec![0f32; rows * cols];
    q4g64_dequantize(&blob, rows, cols, &mut wd)?;
    let mut y_ref = vec![0f32; rows];
    for r in 0..rows {
        y_ref[r] = wd[r * cols..(r + 1) * cols]
            .iter()
            .zip(&x)
            .map(|(a, b)| a * b)
            .sum();
    }

    let cuda = Arc::new(Cuda::init()?);
    let kernels = Kernels::load(&cuda)?;
    let stream = cuda.stream()?;

    let d_blob = cuda.alloc_device(blob.len())?;
    let d_x = cuda.alloc_device(cols * 4)?;
    let d_y = cuda.alloc_device(rows * 4)?;
    cuda.htod_async(d_blob, &blob, stream)?;
    cuda.htod_async(d_x, bytemuck_f32(&x), stream)?;
    kernels.q4g64_gemv(&cuda, d_blob, d_x, d_y, rows as i32, cols as i32, stream)?;
    let mut y_gpu_bytes = vec![0u8; rows * 4];
    cuda.dtoh_async(&mut y_gpu_bytes, d_y, stream)?;
    cuda.sync_stream(stream)?;
    let y_gpu: Vec<f32> = y_gpu_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut worst = 0f32;
    for (a, b) in y_ref.iter().zip(&y_gpu) {
        let denom = a.abs().max(1.0);
        worst = worst.max((a - b).abs() / denom);
    }
    // fp32 accumulation order differs between CPU and warp reduction; allow
    // small relative drift.
    if worst > 1e-3 {
        bail!("kernel mismatch: worst relative error {worst}");
    }
    println!("gemv: correctness OK (worst rel err {worst:.2e})");

    let t0 = Instant::now();
    for _ in 0..iters {
        kernels.q4g64_gemv(&cuda, d_blob, d_x, d_y, rows as i32, cols as i32, stream)?;
    }
    cuda.sync_stream(stream)?;
    let secs = t0.elapsed().as_secs_f64();
    let bytes = blob.len() as f64 * iters as f64;
    println!(
        "gemv: {:.1} us/launch, {:.1} GB/s weight throughput",
        secs * 1e6 / iters as f64,
        bytes / 1e9 / secs
    );
    Ok(())
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
