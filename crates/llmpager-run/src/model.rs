//! Model config (from the pack's embedded HF config.json) and the resident
//! core: every non-expert weight, uploaded to VRAM once at startup.
//! Matrices stay bf16 (the bf16_gemv kernel reads them directly); norm
//! weights are converted to f32 on the host (they're vectors — tiny).

use std::path::Path;

use anyhow::{bail, Context, Result};
use llmpager_core::st::SafeTensors;
use llmpager_cuda::driver::{CUdeviceptr, CUstream, Cuda};

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub moe_inter: usize,
    pub experts: usize,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub vocab: usize,
}

impl Config {
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let u = |k: &str| -> Result<usize> {
            v[k].as_u64().map(|x| x as usize).with_context(|| format!("config: missing {k}"))
        };
        let heads = u("num_attention_heads")?;
        let hidden = u("hidden_size")?;
        Ok(Self {
            hidden,
            layers: u("num_hidden_layers")?,
            heads,
            kv_heads: u("num_key_value_heads")?,
            head_dim: v["head_dim"].as_u64().map(|x| x as usize).unwrap_or(hidden / heads),
            moe_inter: u("moe_intermediate_size")?,
            experts: u("num_experts")?,
            top_k: u("num_experts_per_tok")?,
            norm_topk_prob: v["norm_topk_prob"].as_bool().unwrap_or(false),
            rms_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: v["rope_theta"].as_f64().unwrap_or(1e6) as f32,
            vocab: u("vocab_size")?,
        })
    }
}

/// A core matrix on the device: bf16 verbatim from the checkpoint, or
/// re-quantized to q4g64 at load time (~4x less bandwidth per GEMV).
#[derive(Clone, Copy)]
pub struct Mat {
    pub dev: CUdeviceptr,
    pub rows: i32,
    pub cols: i32,
    pub q4: bool,
}

pub struct LayerWeights {
    pub input_ln: CUdeviceptr,   // f32 [hidden]
    pub q: Mat,                  // [heads*hd, hidden]
    pub k: Mat,                  // [kv*hd, hidden]
    pub v: Mat,                  // [kv*hd, hidden]
    pub o: Mat,                  // [hidden, heads*hd]
    pub q_norm: CUdeviceptr,     // f32 [hd]
    pub k_norm: CUdeviceptr,     // f32 [hd]
    pub post_ln: CUdeviceptr,    // f32 [hidden]
    pub router: Mat,             // [experts, hidden]
}

pub struct CoreWeights {
    pub embed: CUdeviceptr,      // bf16 [vocab, hidden] (row gather only)
    pub final_norm: CUdeviceptr, // f32 [hidden]
    pub lm_head: Mat,            // [vocab, hidden]
    pub layers: Vec<LayerWeights>,
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

impl CoreWeights {
    pub fn load(
        cuda: &Cuda,
        core_path: &Path,
        cfg: &Config,
        core_q4: bool,
        stream: CUstream,
    ) -> Result<Self> {
        let st = SafeTensors::open(core_path)?;

        let up_bf16 = |name: &str, want_elems: usize| -> Result<CUdeviceptr> {
            let (raw, info) = st.raw(name)?;
            if info.dtype != "BF16" {
                bail!("{name}: expected BF16 core weight, got {}", info.dtype);
            }
            if raw.len() != want_elems * 2 {
                bail!("{name}: {} bytes, expected {}", raw.len(), want_elems * 2);
            }
            let d = cuda.alloc_device(raw.len())?;
            cuda.htod_async(d, &raw, stream)?;
            cuda.sync_stream(stream)?; // raw dropped at return; copy must be done
            Ok(d)
        };

        // A GEMV matrix: q4-quantize on the host at load time when enabled
        // and the shape allows it (cols % 64), else upload bf16 verbatim.
        let up_mat = |name: &str, rows: usize, cols: usize| -> Result<Mat> {
            if core_q4 && cols % 64 == 0 {
                let (vals, shape) = st.f32(name)?;
                if shape != vec![rows, cols] {
                    bail!("{name}: shape {shape:?}, expected [{rows}, {cols}]");
                }
                let mut blob = vec![0u8; llmpager_core::quant::q4g64_bytes(rows, cols)];
                llmpager_core::quant::q4g64_quantize(&vals, rows, cols, &mut blob)?;
                let d = cuda.alloc_device(blob.len())?;
                cuda.htod_async(d, &blob, stream)?;
                cuda.sync_stream(stream)?;
                Ok(Mat { dev: d, rows: rows as i32, cols: cols as i32, q4: true })
            } else {
                Ok(Mat {
                    dev: up_bf16(name, rows * cols)?,
                    rows: rows as i32,
                    cols: cols as i32,
                    q4: false,
                })
            }
        };
        let up_norm = |name: &str, want_elems: usize| -> Result<CUdeviceptr> {
            let (vals, _) = st.f32(name)?;
            if vals.len() != want_elems {
                bail!("{name}: {} elems, expected {want_elems}", vals.len());
            }
            let d = cuda.alloc_device(vals.len() * 4)?;
            cuda.htod_async(d, f32_bytes(&vals), stream)?;
            cuda.sync_stream(stream)?;
            Ok(d)
        };

        let qkv = cfg.heads * cfg.head_dim;
        let kv = cfg.kv_heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}");
            layers.push(LayerWeights {
                input_ln: up_norm(&format!("{p}.input_layernorm.weight"), cfg.hidden)?,
                q: up_mat(&format!("{p}.self_attn.q_proj.weight"), qkv, cfg.hidden)?,
                k: up_mat(&format!("{p}.self_attn.k_proj.weight"), kv, cfg.hidden)?,
                v: up_mat(&format!("{p}.self_attn.v_proj.weight"), kv, cfg.hidden)?,
                o: up_mat(&format!("{p}.self_attn.o_proj.weight"), cfg.hidden, qkv)?,
                q_norm: up_norm(&format!("{p}.self_attn.q_norm.weight"), cfg.head_dim)?,
                k_norm: up_norm(&format!("{p}.self_attn.k_norm.weight"), cfg.head_dim)?,
                post_ln: up_norm(&format!("{p}.post_attention_layernorm.weight"), cfg.hidden)?,
                router: up_mat(&format!("{p}.mlp.gate.weight"), cfg.experts, cfg.hidden)?,
            });
        }

        let embed = up_bf16("model.embed_tokens.weight", cfg.vocab * cfg.hidden)?;
        let lm_head = if st.names().any(|n| n == "lm_head.weight") {
            up_mat("lm_head.weight", cfg.vocab, cfg.hidden)?
        } else {
            // Tied embeddings: reuse the bf16 table as the output projection.
            Mat { dev: embed, rows: cfg.vocab as i32, cols: cfg.hidden as i32, q4: false }
        };
        Ok(Self {
            embed,
            final_norm: up_norm("model.norm.weight", cfg.hidden)?,
            lm_head,
            layers,
        })
    }
}
