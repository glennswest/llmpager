//! Decode-kernel verification: each GPU kernel vs a CPU reference on small
//! shapes. This is the correctness gate for M2's runtime building blocks.

use std::sync::Arc;

use anyhow::{bail, Result};
use llmpager_cuda::driver::{CUdeviceptr, CUstream, Cuda};
use llmpager_cuda::kernels::Kernels;

use crate::{Flags, Rng};

fn f32s(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bf16_round(x: f32) -> u16 {
    (((x.to_bits()) + 0x8000) >> 16) as u16
}

fn bf16_val(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

struct Gpu {
    cuda: Arc<Cuda>,
    kernels: Kernels,
    stream: CUstream,
}

impl Gpu {
    fn up(&self, bytes: &[u8]) -> Result<CUdeviceptr> {
        let d = self.cuda.alloc_device(bytes.len().max(4))?;
        self.cuda.htod_async(d, bytes, self.stream)?;
        Ok(d)
    }

    fn down_f32(&self, d: CUdeviceptr, n: usize) -> Result<Vec<f32>> {
        let mut raw = vec![0u8; n * 4];
        self.cuda.dtoh_async(&mut raw, d, self.stream)?;
        self.cuda.sync_stream(self.stream)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) -> Result<()> {
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(want) {
        worst = worst.max((g - w).abs() / w.abs().max(1.0));
    }
    if worst > tol || got.len() != want.len() {
        bail!("{name}: FAIL (worst rel err {worst:.2e}, tol {tol:.0e})");
    }
    println!("{name}: OK (worst rel err {worst:.2e})");
    Ok(())
}

pub fn run(_f: &Flags) -> Result<()> {
    let cuda = Arc::new(Cuda::init()?);
    let kernels = Kernels::load(&cuda)?;
    let stream = cuda.stream()?;
    let g = Gpu { cuda: Arc::clone(&cuda), kernels, stream };
    let mut rng = Rng::new(23);
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n).map(|_| (rng.unit() as f32 - 0.5) * 2.0).collect()
    };

    // rmsnorm
    {
        let n = 2048;
        let (x, w) = (rnd(n), rnd(n));
        let eps = 1e-6f32;
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let want: Vec<f32> = x.iter().zip(&w).map(|(a, b)| a * inv * b).collect();
        let (dx, dw) = (g.up(f32s(&x))?, g.up(f32s(&w))?);
        let dy = g.cuda.alloc_device(n * 4)?;
        g.kernels.rmsnorm(&cuda, dx, dw, dy, n as i32, eps, stream)?;
        check("rmsnorm", &g.down_f32(dy, n)?, &want, 1e-4)?;
    }

    // bf16_gemv
    {
        let (rows, cols) = (64, 128);
        let wf = rnd(rows * cols);
        let x = rnd(cols);
        let wb: Vec<u16> = wf.iter().map(|v| bf16_round(*v)).collect();
        let want: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|c| bf16_val(wb[r * cols + c]) * x[c]).sum())
            .collect();
        let wb_bytes: Vec<u8> = wb.iter().flat_map(|h| h.to_le_bytes()).collect();
        let (dw, dx) = (g.up(&wb_bytes)?, g.up(f32s(&x))?);
        let dy = g.cuda.alloc_device(rows * 4)?;
        g.kernels.bf16_gemv(&cuda, dw, dx, dy, rows as i32, cols as i32, stream)?;
        check("bf16_gemv", &g.down_f32(dy, rows)?, &want, 1e-4)?;
    }

    // silu_mul + add
    {
        let n = 1000;
        let (a, b) = (rnd(n), rnd(n));
        let want: Vec<f32> =
            a.iter().zip(&b).map(|(x, y)| x / (1.0 + (-x).exp()) * y).collect();
        let (da, db) = (g.up(f32s(&a))?, g.up(f32s(&b))?);
        let dout = g.cuda.alloc_device(n * 4)?;
        g.kernels.silu_mul(&cuda, da, db, dout, n as i32, stream)?;
        check("silu_mul", &g.down_f32(dout, n)?, &want, 1e-4)?;

        let want2: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        g.kernels.add(&cuda, da, db, n as i32, stream)?;
        check("add", &g.down_f32(da, n)?, &want2, 1e-5)?;
    }

    // rope
    {
        let (heads, hd, pos, base) = (4usize, 16usize, 7i32, 1e6f32);
        let x = rnd(heads * hd);
        let mut want = x.clone();
        for h in 0..heads {
            for i in 0..hd / 2 {
                let freq = base.powf(-2.0 * i as f32 / hd as f32);
                let ang = pos as f32 * freq;
                let (s, c) = ang.sin_cos();
                let x1 = x[h * hd + i];
                let x2 = x[h * hd + i + hd / 2];
                want[h * hd + i] = x1 * c - x2 * s;
                want[h * hd + i + hd / 2] = x1 * s + x2 * c;
            }
        }
        let dx = g.up(f32s(&x))?;
        g.kernels.rope(&cuda, dx, heads as i32, hd as i32, pos, base, stream)?;
        // __powf/__sinf are fast-math approximations; allow more slack.
        check("rope", &g.down_f32(dx, heads * hd)?, &want, 5e-3)?;
    }

    // attn_decode (GQA 4 heads over 2 kv heads)
    {
        let (heads, kv_heads, hd, seq, max_seq) = (4usize, 2usize, 16usize, 9usize, 16usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let q = rnd(heads * hd);
        let k = rnd(kv_heads * max_seq * hd);
        let v = rnd(kv_heads * max_seq * hd);
        let mut want = vec![0f32; heads * hd];
        for h in 0..heads {
            let kvh = h / (heads / kv_heads);
            let scores: Vec<f32> = (0..seq)
                .map(|p| {
                    (0..hd)
                        .map(|d| q[h * hd + d] * k[(kvh * max_seq + p) * hd + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            for d in 0..hd {
                want[h * hd + d] = (0..seq)
                    .map(|p| exps[p] / z * v[(kvh * max_seq + p) * hd + d])
                    .sum();
            }
        }
        let (dq, dk, dv) = (g.up(f32s(&q))?, g.up(f32s(&k))?, g.up(f32s(&v))?);
        let dout = g.cuda.alloc_device(heads * hd * 4)?;
        let dscratch = g.cuda.alloc_device(heads * max_seq * 4)?;
        g.kernels.attn_decode(
            &cuda, dq, dk, dv, dout, dscratch,
            heads as i32, kv_heads as i32, hd as i32,
            seq as i32, max_seq as i32, scale, stream,
        )?;
        check("attn_decode", &g.down_f32(dout, heads * hd)?, &want, 1e-3)?;
    }

    // bf16_row
    {
        let (vocab, n, row) = (10usize, 64usize, 3usize);
        let table = rnd(vocab * n);
        let tb: Vec<u16> = table.iter().map(|v| bf16_round(*v)).collect();
        let want: Vec<f32> = (0..n).map(|i| bf16_val(tb[row * n + i])).collect();
        let tb_bytes: Vec<u8> = tb.iter().flat_map(|h| h.to_le_bytes()).collect();
        let dt = g.up(&tb_bytes)?;
        let dout = g.cuda.alloc_device(n * 4)?;
        g.kernels.bf16_row(&cuda, dt, row as i32, n as i32, dout, stream)?;
        check("bf16_row", &g.down_f32(dout, n)?, &want, 1e-6)?;
    }

    println!("kernels: all decode kernels verified");
    Ok(())
}
