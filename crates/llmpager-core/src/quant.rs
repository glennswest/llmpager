//! `q4g64`: symmetric 4-bit groupwise quantization, group size 64.
//!
//! Weights are row-major `[rows, cols]`; groups run along the input (col)
//! axis, so a GEMV kernel walking one row streams scales and nibbles
//! sequentially. Per group of 64 values: one f16 scale, then 32 bytes of
//! nibbles (value `i` lives in byte `i/2`; even `i` = low nibble). Nibbles
//! store `q + 8` where `q = clamp(round(x/scale), -8, 7)`.
//!
//! Layout per tensor: all scales first (`rows * cols/64` f16, row-major),
//! then all nibble bytes (`rows * cols/2`). Keeping the two regions separate
//! (rather than interleaved) lets the kernel vectorize loads of each.
//!
//! `cols` must be a multiple of 64.

use anyhow::{bail, Result};

pub const GROUP: usize = 64;

/// Bytes a quantized `[rows, cols]` tensor occupies: scales then data.
pub fn q4g64_bytes(rows: usize, cols: usize) -> usize {
    rows * (cols / GROUP) * 2 + rows * cols / 2
}

/// Quantize `src` (row-major f32, `rows * cols`) into `dst`, which must be
/// exactly [`q4g64_bytes`] long. Returns the maximum absolute error.
pub fn q4g64_quantize(src: &[f32], rows: usize, cols: usize, dst: &mut [u8]) -> Result<f32> {
    if cols % GROUP != 0 {
        bail!("cols {cols} not a multiple of {GROUP}");
    }
    if src.len() != rows * cols {
        bail!("src length {} != {rows}x{cols}", src.len());
    }
    if dst.len() != q4g64_bytes(rows, cols) {
        bail!("dst length {} != {}", dst.len(), q4g64_bytes(rows, cols));
    }
    let groups_per_row = cols / GROUP;
    let scales_len = rows * groups_per_row * 2;
    let (scales, data) = dst.split_at_mut(scales_len);

    let mut max_err = 0.0f32;
    for r in 0..rows {
        for g in 0..groups_per_row {
            let vals = &src[r * cols + g * GROUP..r * cols + (g + 1) * GROUP];
            let amax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // Denormal-small scales round-trip badly through f16; clamp up.
            let scale = f16_round(if amax > 0.0 { amax / 7.0 } else { 1e-8 }).max(1e-7);
            let sidx = (r * groups_per_row + g) * 2;
            scales[sidx..sidx + 2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());

            let base = r * cols / 2 + g * GROUP / 2;
            for i in 0..GROUP / 2 {
                let q0 = quantize_one(vals[2 * i], scale, &mut max_err);
                let q1 = quantize_one(vals[2 * i + 1], scale, &mut max_err);
                data[base + i] = (q0 as u8 & 0x0F) | ((q1 as u8 & 0x0F) << 4);
            }
        }
    }
    Ok(max_err)
}

fn quantize_one(x: f32, scale: f32, max_err: &mut f32) -> i32 {
    let q = (x / scale).round().clamp(-8.0, 7.0) as i32;
    let err = (x - q as f32 * scale).abs();
    if err > *max_err {
        *max_err = err;
    }
    q + 8
}

/// Dequantize back to f32 (reference implementation; the GPU kernel is the
/// production consumer of this format).
pub fn q4g64_dequantize(buf: &[u8], rows: usize, cols: usize, out: &mut [f32]) -> Result<()> {
    if cols % GROUP != 0 || buf.len() != q4g64_bytes(rows, cols) || out.len() != rows * cols {
        bail!("q4g64_dequantize: bad dimensions");
    }
    let groups_per_row = cols / GROUP;
    let scales_len = rows * groups_per_row * 2;
    let (scales, data) = buf.split_at(scales_len);
    for r in 0..rows {
        for g in 0..groups_per_row {
            let sidx = (r * groups_per_row + g) * 2;
            let scale = f16_bits_to_f32(u16::from_le_bytes([scales[sidx], scales[sidx + 1]]));
            let base = r * cols / 2 + g * GROUP / 2;
            for i in 0..GROUP / 2 {
                let b = data[base + i];
                out[r * cols + g * GROUP + 2 * i] = ((b & 0x0F) as i32 - 8) as f32 * scale;
                out[r * cols + g * GROUP + 2 * i + 1] = ((b >> 4) as i32 - 8) as f32 * scale;
            }
        }
    }
    Ok(())
}

// f16 helpers (scales only) — kept dependency-free.

fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let frac = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> 0
        }
        let m = (frac | 0x80_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    sign | ((e as u32) << 10) as u16 | ((frac + 0x1000) >> 13) as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // subnormal: normalize
            let shift = f.leading_zeros() - 21;
            sign | ((127 - 15 - shift + 1) << 23) | ((f << (shift + 13)) & 0x7F_FFFF)
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, f) => sign | 0x7F80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

fn f16_round(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * 0.1
            })
            .collect()
    }

    #[test]
    fn round_trip_error_bounded() -> Result<()> {
        let (rows, cols) = (8, 256);
        let src = pseudo(rows * cols, 3);
        let mut buf = vec![0u8; q4g64_bytes(rows, cols)];
        let max_err = q4g64_quantize(&src, rows, cols, &mut buf)?;
        let mut back = vec![0f32; rows * cols];
        q4g64_dequantize(&buf, rows, cols, &mut back)?;
        // 4-bit symmetric: error <= scale/2 ~= amax/14 per group.
        for (a, b) in src.iter().zip(back.iter()) {
            assert!((a - b).abs() <= max_err + 1e-6);
        }
        assert!(max_err < 0.1 / 14.0 * 1.1, "max_err {max_err}");
        Ok(())
    }

    #[test]
    fn f16_helpers() {
        for x in [0.0f32, 1.0, -1.0, 0.5, 65504.0, 1e-8, 3.14159] {
            let y = f16_bits_to_f32(f32_to_f16_bits(x));
            assert!((x - y).abs() <= (x.abs() * 0.001).max(1e-7), "{x} -> {y}");
        }
    }

    #[test]
    fn rejects_bad_dims() {
        let src = vec![0f32; 10];
        let mut buf = vec![0u8; 10];
        assert!(q4g64_quantize(&src, 1, 10, &mut buf).is_err());
    }
}
