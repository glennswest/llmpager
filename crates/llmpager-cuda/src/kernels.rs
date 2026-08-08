//! Compiled-in PTX kernels (built by build.rs with nvcc; `kernels` feature).

use anyhow::Result;

use crate::driver::{CUdeviceptr, CUfunction, CUstream, Cuda};

pub const Q4G64_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/q4g64.ptx"));
pub const DECODE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/decode.ptx"));

macro_rules! params {
    ($($arg:expr),* $(,)?) => {{
        [$((&raw mut $arg) as *mut std::ffi::c_void),*]
    }};
}

#[derive(Clone, Copy)]
pub struct Kernels {
    q4g64_gemv: CUfunction,
    q4g64_gemv_batch: CUfunction,
    moe_reduce_f32: CUfunction,
    rmsnorm_f32: CUfunction,
    bf16_gemv: CUfunction,
    silu_mul_f32: CUfunction,
    add_f32: CUfunction,
    scale_add_f32: CUfunction,
    kv_append_f32: CUfunction,
    rope_f32: CUfunction,
    attn_decode_f32: CUfunction,
    bf16_row_to_f32: CUfunction,
}

fn grid_1d(n: usize, block: u32) -> u32 {
    ((n as u32).div_ceil(block)).min(1024).max(1)
}

impl Kernels {
    pub fn load(cuda: &Cuda) -> Result<Self> {
        let q4 = cuda.module_from_ptx(Q4G64_PTX)?;
        let de = cuda.module_from_ptx(DECODE_PTX)?;
        Ok(Self {
            q4g64_gemv: cuda.function(q4, "q4g64_gemv")?,
            q4g64_gemv_batch: cuda.function(q4, "q4g64_gemv_batch")?,
            moe_reduce_f32: cuda.function(q4, "moe_reduce_f32")?,
            rmsnorm_f32: cuda.function(de, "rmsnorm_f32")?,
            bf16_gemv: cuda.function(de, "bf16_gemv")?,
            silu_mul_f32: cuda.function(de, "silu_mul_f32")?,
            add_f32: cuda.function(de, "add_f32")?,
            scale_add_f32: cuda.function(de, "scale_add_f32")?,
            kv_append_f32: cuda.function(de, "kv_append_f32")?,
            rope_f32: cuda.function(de, "rope_f32")?,
            attn_decode_f32: cuda.function(de, "attn_decode_f32")?,
            bf16_row_to_f32: cuda.function(de, "bf16_row_to_f32")?,
        })
    }

    /// y[rows] = W x, W a q4 blob region (scales then nibbles), `group`
    /// values per scale (64 for our packs, 32 for repacked QAT int4).
    #[allow(clippy::too_many_arguments)]
    pub fn q4g64_gemv(
        &self,
        cuda: &Cuda,
        blob: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        rows: i32,
        cols: i32,
        group: i32,
        stream: CUstream,
    ) -> Result<()> {
        const WARPS: u32 = 4;
        let (mut blob, mut x, mut y, mut rows_a, mut cols_a, mut group_a) =
            (blob, x, y, rows, cols, group);
        let mut p = params![blob, x, y, rows_a, cols_a, group_a];
        cuda.launch(
            self.q4g64_gemv,
            ((rows as u32).div_ceil(WARPS), 1, 1),
            (WARPS * 32, 1, 1),
            &mut p,
            stream,
        )
    }

    /// Row-wise RMSNorm over `rows` rows of length `n`; `w` shared per row.
    #[allow(clippy::too_many_arguments)]
    /// Batched q4g64 GEMV over `experts` blobs (device array of base
    /// addresses) at a shared byte offset; x_stride 0 shares the input.
    #[allow(clippy::too_many_arguments)]
    pub fn q4g64_gemv_batch(
        &self,
        cuda: &Cuda,
        blobs: CUdeviceptr,
        region_off: u64,
        x: CUdeviceptr,
        x_stride: i32,
        y: CUdeviceptr,
        rows: i32,
        cols: i32,
        group: i32,
        experts: i32,
        stream: CUstream,
    ) -> Result<()> {
        const WARPS: u32 = 4;
        let (mut blobs, mut off, mut x, mut xs, mut y) = (blobs, region_off, x, x_stride, y);
        let (mut rows_a, mut cols_a, mut group_a) = (rows, cols, group);
        let mut p = params![blobs, off, x, xs, y, rows_a, cols_a, group_a];
        cuda.launch(
            self.q4g64_gemv_batch,
            ((rows as u32).div_ceil(WARPS), experts as u32, 1),
            (WARPS * 32, 1, 1),
            &mut p,
            stream,
        )
    }

    /// out[n] += Σ_e wts[e] * eouts[e, n]
    #[allow(clippy::too_many_arguments)]
    pub fn moe_reduce(
        &self,
        cuda: &Cuda,
        eouts: CUdeviceptr,
        wts: CUdeviceptr,
        out: CUdeviceptr,
        experts: i32,
        n: i32,
        stream: CUstream,
    ) -> Result<()> {
        let (mut eo, mut w, mut o, mut e_a, mut n_a) = (eouts, wts, out, experts, n);
        let mut p = params![eo, w, o, e_a, n_a];
        cuda.launch(self.moe_reduce_f32, (grid_1d(n as usize, 256), 1, 1), (256, 1, 1), &mut p, stream)
    }

    pub fn rmsnorm(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        w: CUdeviceptr,
        y: CUdeviceptr,
        rows: i32,
        n: i32,
        eps: f32,
        stream: CUstream,
    ) -> Result<()> {
        let (mut x, mut w, mut y, mut n, mut eps) = (x, w, y, n, eps);
        let mut p = params![x, w, y, n, eps];
        cuda.launch(self.rmsnorm_f32, (rows as u32, 1, 1), (256, 1, 1), &mut p, stream)
    }

    /// a += s * b
    pub fn scale_add(
        &self,
        cuda: &Cuda,
        a: CUdeviceptr,
        b: CUdeviceptr,
        s: f32,
        n: i32,
        stream: CUstream,
    ) -> Result<()> {
        let (mut a, mut b, mut s_a, mut n_a) = (a, b, s, n);
        let mut p = params![a, b, s_a, n_a];
        cuda.launch(self.scale_add_f32, (grid_1d(n as usize, 256), 1, 1), (256, 1, 1), &mut p, stream)
    }

    /// Append this token's k/v ([kv_heads, head_dim]) to the caches at `pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(
        &self,
        cuda: &Cuda,
        k: CUdeviceptr,
        v: CUdeviceptr,
        kcache: CUdeviceptr,
        vcache: CUdeviceptr,
        kv_heads: i32,
        head_dim: i32,
        pos: i32,
        max_seq: i32,
        stream: CUstream,
    ) -> Result<()> {
        let n = (kv_heads * head_dim) as usize;
        let (mut k, mut v, mut kc, mut vc) = (k, v, kcache, vcache);
        let (mut kvh, mut hd, mut pos_a, mut ms) = (kv_heads, head_dim, pos, max_seq);
        let mut p = params![k, v, kc, vc, kvh, hd, pos_a, ms];
        cuda.launch(self.kv_append_f32, (grid_1d(n, 256), 1, 1), (256, 1, 1), &mut p, stream)
    }

    /// y[rows] = W x, W bf16 [rows, cols].
    pub fn bf16_gemv(
        &self,
        cuda: &Cuda,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        rows: i32,
        cols: i32,
        stream: CUstream,
    ) -> Result<()> {
        const WARPS: u32 = 4;
        let (mut w, mut x, mut y, mut rows_a, mut cols_a) = (w, x, y, rows, cols);
        let mut p = params![w, x, y, rows_a, cols_a];
        cuda.launch(
            self.bf16_gemv,
            ((rows as u32).div_ceil(WARPS), 1, 1),
            (WARPS * 32, 1, 1),
            &mut p,
            stream,
        )
    }

    pub fn silu_mul(
        &self,
        cuda: &Cuda,
        gate: CUdeviceptr,
        up: CUdeviceptr,
        out: CUdeviceptr,
        n: i32,
        stream: CUstream,
    ) -> Result<()> {
        let (mut gate, mut up, mut out, mut n_a) = (gate, up, out, n);
        let mut p = params![gate, up, out, n_a];
        cuda.launch(self.silu_mul_f32, (grid_1d(n as usize, 256), 1, 1), (256, 1, 1), &mut p, stream)
    }

    /// a += b
    pub fn add(
        &self,
        cuda: &Cuda,
        a: CUdeviceptr,
        b: CUdeviceptr,
        n: i32,
        stream: CUstream,
    ) -> Result<()> {
        let (mut a, mut b, mut n_a) = (a, b, n);
        let mut p = params![a, b, n_a];
        cuda.launch(self.add_f32, (grid_1d(n as usize, 256), 1, 1), (256, 1, 1), &mut p, stream)
    }

    /// In-place NeoX RoPE over [heads, head_dim] at position `pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        heads: i32,
        head_dim: i32,
        pos: i32,
        base: f32,
        stream: CUstream,
    ) -> Result<()> {
        let n = (heads * head_dim / 2) as usize;
        let (mut x, mut h, mut hd, mut pos_a, mut base_a) = (x, heads, head_dim, pos, base);
        let mut p = params![x, h, hd, pos_a, base_a];
        cuda.launch(self.rope_f32, (grid_1d(n, 256), 1, 1), (256, 1, 1), &mut p, stream)
    }

    /// Decode attention over f32 KV caches; `scratch` is [heads, max_seq].
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        kcache: CUdeviceptr,
        vcache: CUdeviceptr,
        out: CUdeviceptr,
        scratch: CUdeviceptr,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        max_seq: i32,
        scale: f32,
        stream: CUstream,
    ) -> Result<()> {
        let (mut q, mut k, mut v, mut o, mut s) = (q, kcache, vcache, out, scratch);
        let (mut h, mut kvh, mut hd, mut sl, mut ms, mut sc) =
            (heads, kv_heads, head_dim, seq_len, max_seq, scale);
        let mut p = params![q, k, v, o, s, h, kvh, hd, sl, ms, sc];
        cuda.launch(self.attn_decode_f32, (heads as u32, 1, 1), (256, 1, 1), &mut p, stream)
    }

    /// out[n] = f32(table[row, :n]) — embedding row gather.
    pub fn bf16_row(
        &self,
        cuda: &Cuda,
        table: CUdeviceptr,
        row: i32,
        n: i32,
        out: CUdeviceptr,
        stream: CUstream,
    ) -> Result<()> {
        let (mut t, mut r, mut n_a, mut o) = (table, row, n, out);
        let mut p = params![t, r, n_a, o];
        cuda.launch(self.bf16_row_to_f32, (grid_1d(n as usize, 256), 1, 1), (256, 1, 1), &mut p, stream)
    }
}
