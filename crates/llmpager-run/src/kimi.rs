//! Kimi K2.x / DeepSeek-V3 decode path: MLA attention (absorbed decode over
//! the compressed KV cache), sigmoid router with bias-corrected top-k,
//! shared expert, dense first layer(s), YaRN RoPE.
//!
//! Differences from the Qwen path (`decode.rs`):
//! - Attention caches one 576-float row per token (c_kv 512 + k_rope 64),
//!   shared by all 64 heads (MQA form). Per-head K/V never materialize:
//!   queries are absorbed through W_kvb_k^T and outputs through W_kvb_v.
//! - The core is far larger (MLA projections), so every core matrix is
//!   requantized to q4g64 at load; the embedding table stays in host RAM
//!   (row gather per token), only lm_head lives in VRAM.
//! - MoE: sigmoid scores, top-k chosen by score + per-expert bias, weights
//!   are the unbiased scores renormalized then scaled by
//!   `routed_scaling_factor`; a shared expert always runs.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use llmpager_core::pack::PackReader;
use llmpager_core::st::SafeTensors;
use llmpager_cuda::driver::{CUdeviceptr, CUevent, CUstream, Cuda};
use llmpager_cuda::kernels::Kernels;
use llmpager_cuda::pager::{ExpertHandle, Pager, PagerConfig};

use crate::model::Mat;

#[derive(Debug, Clone)]
pub struct KimiConfig {
    pub hidden: usize,
    pub layers: usize,
    pub dense_layers: usize,
    pub heads: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_head: usize,
    pub moe_inter: usize,
    pub dense_inter: usize,
    pub experts: usize,
    pub top_k: usize,
    pub routed_scaling: f32,
    pub norm_topk_prob: bool,
    pub rms_eps: f32,
    pub vocab: usize,
    pub eos: Vec<u32>,
    pub softmax_scale: f32,
    /// Per-pair RoPE inverse frequencies (rope/2 entries; YaRN-blended).
    pub inv_freq: Vec<f32>,
}

impl KimiConfig {
    pub fn is_kimi(cfg: &serde_json::Value) -> bool {
        !cfg["kv_lora_rank"].is_null()
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let u = |k: &str| -> Result<usize> {
            v[k].as_u64().map(|x| x as usize).with_context(|| format!("config: missing {k}"))
        };
        let nope = u("qk_nope_head_dim")?;
        let rope = u("qk_rope_head_dim")?;
        let theta = v["rope_theta"].as_f64().unwrap_or(1e4) as f32;

        // YaRN frequency blend + softmax-scale correction (HF DeepseekV3).
        let half = rope / 2;
        let extrap: Vec<f32> =
            (0..half).map(|i| theta.powf(-2.0 * i as f32 / rope as f32)).collect();
        let (inv_freq, mscale_sq) = match v["rope_scaling"].as_object() {
            Some(rs) if rs.get("type").and_then(|t| t.as_str()) == Some("yarn") => {
                let f = |k: &str, d: f64| rs.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
                let factor = f("factor", 1.0) as f32;
                let beta_fast = f("beta_fast", 32.0) as f32;
                let beta_slow = f("beta_slow", 1.0) as f32;
                let orig = f("original_max_position_embeddings", 4096.0) as f32;
                let mscale_all_dim = f("mscale_all_dim", 0.0) as f32;
                let dim = rope as f32;
                let corr = |rot: f32| -> f32 {
                    dim * (orig / (rot * 2.0 * std::f32::consts::PI)).ln()
                        / (2.0 * theta.ln())
                };
                let low = corr(beta_fast).floor().max(0.0);
                let high = corr(beta_slow).ceil().min(dim - 1.0);
                let blended: Vec<f32> = (0..half)
                    .map(|i| {
                        let ramp =
                            (((i as f32) - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
                        let mask = 1.0 - ramp; // 1 => extrapolate (high-freq dims)
                        (extrap[i] / factor) * (1.0 - mask) + extrap[i] * mask
                    })
                    .collect();
                let m = if mscale_all_dim > 0.0 {
                    0.1 * mscale_all_dim * factor.ln() + 1.0
                } else {
                    1.0
                };
                (blended, m * m)
            }
            _ => (extrap, 1.0),
        };
        let softmax_scale = ((nope + rope) as f32).powf(-0.5) * mscale_sq;

        Ok(Self {
            hidden: u("hidden_size")?,
            layers: u("num_hidden_layers")?,
            dense_layers: v["moe_layer_offset"]
                .as_u64()
                .or_else(|| v["first_k_dense_replace"].as_u64())
                .unwrap_or(0) as usize,
            heads: u("num_attention_heads")?,
            q_lora: u("q_lora_rank")?,
            kv_lora: u("kv_lora_rank")?,
            nope,
            rope,
            v_head: u("v_head_dim")?,
            moe_inter: u("moe_intermediate_size")?,
            dense_inter: u("intermediate_size")?,
            experts: u("n_routed_experts")?,
            top_k: u("num_experts_per_tok")?,
            routed_scaling: v["routed_scaling_factor"].as_f64().unwrap_or(1.0) as f32,
            norm_topk_prob: v["norm_topk_prob"].as_bool().unwrap_or(true),
            rms_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            vocab: u("vocab_size")?,
            eos: match &v["eos_token_id"] {
                serde_json::Value::Number(n) => n.as_u64().map(|x| x as u32).into_iter().collect(),
                serde_json::Value::Array(a) => {
                    a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()
                }
                _ => Vec::new(),
            },
            softmax_scale,
            inv_freq,
        })
    }
}

struct KimiLayer {
    input_ln: CUdeviceptr,
    post_ln: CUdeviceptr,
    q_a: Mat,
    q_a_ln: CUdeviceptr,
    q_b: Mat,
    kv_a: Mat,
    kv_a_ln: CUdeviceptr,
    /// Absorbed W_kvb_k^T, bf16 [heads][kv_lora, nope] contiguous.
    kt: CUdeviceptr,
    /// W_kvb_v, bf16 [heads][v_head, kv_lora] contiguous.
    vw: CUdeviceptr,
    o: Mat,
    /// MoE layers only.
    router: Option<Mat>,
    router_bias: Option<Vec<f32>>,
    shared: Option<[Mat; 3]>, // gate, up, down
    /// Dense layers only.
    dense: Option<[Mat; 3]>,
}

struct KimiCore {
    /// bf16 embedding table, host-resident (row gather per token).
    embed_host: Vec<u8>,
    final_norm: CUdeviceptr,
    lm_head: Mat,
    layers: Vec<KimiLayer>,
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn f32_from_le(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn f32_to_bf16(x: f32) -> u16 {
    ((x.to_bits() + 0x8000) >> 16) as u16
}

impl KimiCore {
    fn load(cuda: &Cuda, core_path: &Path, cfg: &KimiConfig, stream: CUstream) -> Result<Self> {
        let st = SafeTensors::open(core_path)?;

        // q4-requantize every GEMV matrix at load (the core is too big for
        // bf16 in VRAM); norms stay f32, the router stays bf16 (tiny GEMV,
        // and its logits feed a host softmax — keep it exact).
        let up_q4 = |name: &str, rows: usize, cols: usize| -> Result<Mat> {
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
        };
        let up_bf16 = |name: &str, rows: usize, cols: usize| -> Result<Mat> {
            let (raw, info) = st.raw(name)?;
            if info.dtype != "BF16" || raw.len() != rows * cols * 2 {
                bail!("{name}: expected BF16 [{rows}, {cols}]");
            }
            let d = cuda.alloc_device(raw.len())?;
            cuda.htod_async(d, &raw, stream)?;
            cuda.sync_stream(stream)?;
            Ok(Mat { dev: d, rows: rows as i32, cols: cols as i32, q4: false })
        };
        let up_norm = |name: &str, n: usize| -> Result<CUdeviceptr> {
            let (vals, _) = st.f32(name)?;
            if vals.len() != n {
                bail!("{name}: {} elems, expected {n}", vals.len());
            }
            let d = cuda.alloc_device(n * 4)?;
            cuda.htod_async(d, f32_bytes(&vals), stream)?;
            cuda.sync_stream(stream)?;
            Ok(d)
        };

        let h = cfg.heads;
        let (kvl, nope, vh) = (cfg.kv_lora, cfg.nope, cfg.v_head);
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}");
            // kv_b [heads*(nope+vh), kv_lora] splits and reorders into the
            // absorbed forms.
            let (kvb, shape) = st.f32(&format!("{p}.self_attn.kv_b_proj.weight"))?;
            if shape != vec![h * (nope + vh), kvl] {
                bail!("kv_b_proj: shape {shape:?}");
            }
            let mut kt = vec![0u16; h * kvl * nope];
            let mut vw = vec![0u16; h * vh * kvl];
            for head in 0..h {
                let base = head * (nope + vh);
                for r in 0..nope {
                    for c in 0..kvl {
                        // kt[head][c][r] = kv_b[base + r][c]
                        kt[head * kvl * nope + c * nope + r] =
                            f32_to_bf16(kvb[(base + r) * kvl + c]);
                    }
                }
                for r in 0..vh {
                    for c in 0..kvl {
                        vw[head * vh * kvl + r * kvl + c] =
                            f32_to_bf16(kvb[(base + nope + r) * kvl + c]);
                    }
                }
            }
            let up_u16 = |vals: &[u16]| -> Result<CUdeviceptr> {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 2)
                };
                let d = cuda.alloc_device(bytes.len())?;
                cuda.htod_async(d, bytes, stream)?;
                cuda.sync_stream(stream)?;
                Ok(d)
            };

            let is_dense = l < cfg.dense_layers;
            let (router, router_bias, shared, dense) = if is_dense {
                let dense = [
                    up_q4(&format!("{p}.mlp.gate_proj.weight"), cfg.dense_inter, cfg.hidden)?,
                    up_q4(&format!("{p}.mlp.up_proj.weight"), cfg.dense_inter, cfg.hidden)?,
                    up_q4(&format!("{p}.mlp.down_proj.weight"), cfg.hidden, cfg.dense_inter)?,
                ];
                (None, None, None, Some(dense))
            } else {
                let router = up_bf16(&format!("{p}.mlp.gate.weight"), cfg.experts, cfg.hidden)?;
                let (bias, _) = st.f32(&format!("{p}.mlp.gate.e_score_correction_bias"))?;
                let shared = [
                    up_q4(
                        &format!("{p}.mlp.shared_experts.gate_proj.weight"),
                        cfg.moe_inter,
                        cfg.hidden,
                    )?,
                    up_q4(
                        &format!("{p}.mlp.shared_experts.up_proj.weight"),
                        cfg.moe_inter,
                        cfg.hidden,
                    )?,
                    up_q4(
                        &format!("{p}.mlp.shared_experts.down_proj.weight"),
                        cfg.hidden,
                        cfg.moe_inter,
                    )?,
                ];
                (Some(router), Some(bias), Some(shared), None)
            };

            layers.push(KimiLayer {
                input_ln: up_norm(&format!("{p}.input_layernorm.weight"), cfg.hidden)?,
                post_ln: up_norm(&format!("{p}.post_attention_layernorm.weight"), cfg.hidden)?,
                q_a: up_q4(&format!("{p}.self_attn.q_a_proj.weight"), cfg.q_lora, cfg.hidden)?,
                q_a_ln: up_norm(&format!("{p}.self_attn.q_a_layernorm.weight"), cfg.q_lora)?,
                q_b: up_q4(
                    &format!("{p}.self_attn.q_b_proj.weight"),
                    h * (nope + cfg.rope),
                    cfg.q_lora,
                )?,
                kv_a: up_q4(
                    &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
                    kvl + cfg.rope,
                    cfg.hidden,
                )?,
                kv_a_ln: up_norm(&format!("{p}.self_attn.kv_a_layernorm.weight"), kvl)?,
                kt: up_u16(&kt)?,
                vw: up_u16(&vw)?,
                o: up_q4(&format!("{p}.self_attn.o_proj.weight"), cfg.hidden, h * vh)?,
                router,
                router_bias,
                shared,
                dense,
            });
            if l % 8 == 0 {
                eprintln!("  core layer {l}/{}", cfg.layers);
            }
        }

        let (embed_host, einfo) = st.raw("model.embed_tokens.weight")?;
        if einfo.dtype != "BF16" || embed_host.len() != cfg.vocab * cfg.hidden * 2 {
            bail!("embed_tokens: expected BF16 [{}, {}]", cfg.vocab, cfg.hidden);
        }
        Ok(Self {
            embed_host,
            final_norm: up_norm("model.norm.weight", cfg.hidden)?,
            lm_head: up_q4("lm_head.weight", cfg.vocab, cfg.hidden)?,
            layers,
        })
    }

    fn device_ptrs(&self) -> Vec<CUdeviceptr> {
        let mut v = vec![self.final_norm, self.lm_head.dev];
        for l in &self.layers {
            v.extend([
                l.input_ln, l.post_ln, l.q_a.dev, l.q_a_ln, l.q_b.dev,
                l.kv_a.dev, l.kv_a_ln, l.kt, l.vw, l.o.dev,
            ]);
            if let Some(r) = &l.router {
                v.push(r.dev);
            }
            if let Some(s) = &l.shared {
                v.extend(s.iter().map(|m| m.dev));
            }
            if let Some(d) = &l.dense {
                v.extend(d.iter().map(|m| m.dev));
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }
}

pub struct KimiDecoder {
    cuda: Arc<Cuda>,
    kernels: Kernels,
    stream: CUstream,
    pub cfg: KimiConfig,
    core: KimiCore,
    pager: Option<Pager>,
    pack_path: PathBuf,
    io_threads: usize,
    direct: bool,
    ram_bytes: u64,
    /// VRAM left free for other processes on the card (bytes).
    reserve_bytes: u64,
    max_seq: usize,
    batch_cap: usize,
    /// Skip routed experts whose normalized weight falls below this
    /// (fetch-traffic saver; 0.0 disables).
    pub min_expert_weight: f32,
    // Device buffers.
    h: CUdeviceptr,
    h_norm: CUdeviceptr,
    qa: CUdeviceptr,
    q: CUdeviceptr,
    kva: CUdeviceptr,
    ckv_norm: CUdeviceptr,
    q_full: CUdeviceptr,
    ctx: CUdeviceptr,
    attn_pre: CUdeviceptr,
    proj_out: CUdeviceptr,
    attn_scratch: CUdeviceptr,
    inv_freq_dev: CUdeviceptr,
    cache: Vec<CUdeviceptr>, // per layer [max_seq, kv_lora + rope] f32
    router_logits: CUdeviceptr,
    gate_out: CUdeviceptr,
    up_out: CUdeviceptr,
    act_out: CUdeviceptr,
    expert_out: CUdeviceptr,
    d_expert_ptrs: CUdeviceptr,
    d_expert_wts: CUdeviceptr,
    sh_gate: CUdeviceptr,
    sh_up: CUdeviceptr,
    sh_out: CUdeviceptr,
    dn_gate: CUdeviceptr,
    dn_up: CUdeviceptr,
    logits: CUdeviceptr,
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    expert_group: i32,
    logits_host: Vec<u8>,
    router_host: Vec<u8>,
    logits_multi: Vec<Vec<f32>>,
    embed_row: Vec<f32>,
    chunk_cap: usize,
    h_buf: CUdeviceptr,      // [chunk_cap, hidden]
    hn_buf: CUdeviceptr,     // [chunk_cap, hidden]
    router_buf: CUdeviceptr, // [chunk_cap, experts]
    router_chunk_host: Vec<u8>,
    release_events: Vec<CUevent>,
    pending_release: VecDeque<(usize, Vec<ExpertHandle>)>,
    next_event: usize,
}

fn mat_gemv(
    ke: &Kernels,
    cu: &Cuda,
    m: &Mat,
    x: CUdeviceptr,
    y: CUdeviceptr,
    st: CUstream,
) -> Result<()> {
    if m.q4 {
        ke.q4g64_gemv(cu, m.dev, x, y, m.rows, m.cols, 64, st)
    } else {
        ke.bf16_gemv(cu, m.dev, x, y, m.rows, m.cols, st)
    }
}

impl KimiDecoder {
    pub fn new(
        pack_path: &Path,
        core_path: &Path,
        slots: u32,
        io_threads: usize,
        max_seq: usize,
        direct: bool,
        ram_bytes: u64,
        reserve_bytes: u64,
        batch_cap: usize,
    ) -> Result<Self> {
        let batch_cap = batch_cap.max(1);
        let meta = PackReader::open(pack_path)?.meta().clone();
        let cfg = KimiConfig::from_json(&meta.config)?;
        let expert_group: i32 = match meta.dtype.as_str() {
            "q4g32-gud" => 32,
            "q4g64-gud" => 64,
            other => bail!("pack dtype {other} unsupported"),
        };
        if meta.num_layers as usize != cfg.layers - cfg.dense_layers {
            bail!(
                "pack has {} layers, config wants {} MoE layers",
                meta.num_layers,
                cfg.layers - cfg.dense_layers
            );
        }

        let cuda = Arc::new(Cuda::init()?);
        let kernels = Kernels::load(&cuda)?;
        let stream = cuda.stream()?;
        eprintln!(
            "loading kimi core ({} layers, {} dense, hidden {}, {} experts top-{}) ...",
            cfg.layers, cfg.dense_layers, cfg.hidden, cfg.experts, cfg.top_k
        );
        let core = KimiCore::load(&cuda, core_path, &cfg, stream)?;
        let pager = Pager::new(
            Arc::clone(&cuda),
            pack_path,
            PagerConfig {
                slots_per_layer: slots,
                io_threads,
                decay_interval: 64.max(slots * 4),
                direct,
                ram_bytes,
                reserve_bytes,
            },
        )?;

        let f = |n: usize| cuda.alloc_device(n * 4);
        let qk = cfg.kv_lora + cfg.rope; // 576
        let cache: Vec<CUdeviceptr> =
            (0..cfg.layers).map(|_| f(batch_cap * max_seq * qk)).collect::<Result<_>>()?;
        let inv_freq_dev = f(cfg.inv_freq.len())?;
        cuda.htod_async(inv_freq_dev, f32_bytes(&cfg.inv_freq), stream)?;
        cuda.sync_stream(stream)?;

        let gate_bytes = llmpager_core::quant::q4_bytes(
            cfg.moe_inter,
            cfg.hidden,
            expert_group as usize,
        ) as u64;
        Ok(Self {
            h: f(cfg.hidden)?,
            h_norm: f(cfg.hidden)?,
            qa: f(cfg.q_lora)?,
            q: f(cfg.heads * (cfg.nope + cfg.rope))?,
            kva: f(qk)?,
            ckv_norm: f(cfg.kv_lora)?,
            q_full: f(cfg.heads * qk)?,
            ctx: f(cfg.heads * cfg.kv_lora)?,
            attn_pre: f(cfg.heads * cfg.v_head)?,
            proj_out: f(cfg.hidden)?,
            attn_scratch: f(cfg.heads * max_seq)?,
            inv_freq_dev,
            cache,
            router_logits: f(cfg.experts)?,
            gate_out: f(cfg.top_k * cfg.moe_inter)?,
            up_out: f(cfg.top_k * cfg.moe_inter)?,
            act_out: f(cfg.top_k * cfg.moe_inter)?,
            expert_out: f(cfg.top_k * cfg.hidden)?,
            d_expert_ptrs: cuda.alloc_device(cfg.top_k * 8)?,
            d_expert_wts: f(cfg.top_k)?,
            sh_gate: f(cfg.moe_inter)?,
            sh_up: f(cfg.moe_inter)?,
            sh_out: f(cfg.hidden)?,
            dn_gate: f(cfg.dense_inter)?,
            dn_up: f(cfg.dense_inter)?,
            logits: f(cfg.vocab)?,
            gate_off: 32,
            up_off: 32 + gate_bytes,
            down_off: 32 + 2 * gate_bytes,
            expert_group,
            logits_host: vec![0u8; cfg.vocab * 4],
            router_host: vec![0u8; cfg.experts * 4],
            logits_multi: Vec::new(),
            embed_row: vec![0f32; cfg.hidden],
            chunk_cap: 64,
            h_buf: f(64 * cfg.hidden)?,
            hn_buf: f(64 * cfg.hidden)?,
            router_buf: f(64 * cfg.experts)?,
            router_chunk_host: vec![0u8; 64 * cfg.experts * 4],
            release_events: (0..8).map(|_| cuda.event()).collect::<Result<_>>()?,
            pending_release: VecDeque::new(),
            next_event: 0,
            cuda,
            kernels,
            stream,
            cfg,
            core,
            pager: Some(pager),
            pack_path: pack_path.to_path_buf(),
            io_threads,
            direct,
            ram_bytes,
            reserve_bytes,
            max_seq,
            batch_cap,
            min_expert_weight: 0.0,
        })
    }

    pub fn resize_cache(&mut self, slots: u32) -> Result<()> {
        self.cuda.sync()?;
        self.pending_release.clear();
        let previous = self.slots_per_layer();
        self.pager = None;
        let build = |slots: u32, reserve_bytes: u64| {
            Pager::new(
                Arc::clone(&self.cuda),
                &self.pack_path,
                PagerConfig {
                    slots_per_layer: slots,
                    io_threads: self.io_threads,
                    decay_interval: 64.max(slots * 4),
                    direct: self.direct,
                    ram_bytes: self.ram_bytes,
                    reserve_bytes,
                },
            )
        };
        // See decode.rs: never leave the decoder without a pager.
        match build(slots, self.reserve_bytes) {
            Ok(p) => {
                self.pager = Some(p);
                Ok(())
            }
            Err(e) => {
                // Restore service at the previous size, ignoring the reserve:
                // an unsatisfiable reserve is exactly what tends to land here,
                // and re-applying it would fail again and leave no pager.
                self.pager = Some(build(previous, 0).context(
                    "cache resize failed and the previous size could not be restored",
                )?);
                Err(e.context(format!("cache resize to {slots} slots failed; kept {previous}")))
            }
        }
    }

    /// LLMPAGER_DEBUG=1: print staged layer-0 probes for the first token,
    /// formatted to diff against tools' ref_layer0.py CPU reference.
    fn dbg_probe(&self, name: &str, dev: CUdeviceptr, n: usize) {
        if std::env::var("LLMPAGER_DEBUG").is_err() {
            return;
        }
        let mut raw = vec![0u8; n * 4];
        let _ = self.cuda.dtoh_async(&mut raw, dev, self.stream);
        let _ = self.cuda.sync_stream(self.stream);
        let v = f32_from_le(&raw);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let head: Vec<f32> = v.iter().take(6).map(|x| (x * 1e5).round() / 1e5).collect();
        eprintln!("{name}: {head:?} norm={norm:.5}");
    }

    pub fn step(&mut self, token: u32, pos: usize, want_logits: bool) -> Result<u32> {
        if pos >= self.max_seq {
            bail!("position {pos} exceeds max_seq {}", self.max_seq);
        }
        let c = self.cfg.clone();
        let cu = Arc::clone(&self.cuda);
        let cu = &*cu;
        let ke = self.kernels;
        let ke = &ke;
        let st = self.stream;
        let hid = c.hidden as i32;
        let heads = c.heads as i32;
        let qk = (c.kv_lora + c.rope) as i32;
        let q_dim = (c.nope + c.rope) as i32;

        // Host-side embedding gather (table too large for VRAM).
        let row = &self.core.embed_host
            [token as usize * c.hidden * 2..(token as usize + 1) * c.hidden * 2];
        for (i, ch) in row.chunks_exact(2).enumerate() {
            self.embed_row[i] = bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]]));
        }
        cu.htod_async(self.h, f32_bytes(&self.embed_row), st)?;
        let dbg = pos == 0 && std::env::var("LLMPAGER_DEBUG").is_ok();
        if dbg {
            self.dbg_probe("embed", self.h, c.hidden);
        }

        for l in 0..c.layers {
            let w = &self.core.layers[l];
            let dbg0 = dbg && l == 0;

            // MLA attention.
            ke.rmsnorm(cu, self.h, w.input_ln, self.h_norm, 1, hid, c.rms_eps, st)?;
            if dbg0 {
                self.dbg_probe("hn", self.h_norm, c.hidden);
            }
            mat_gemv(ke, cu, &w.q_a, self.h_norm, self.qa, st)?;
            if dbg0 {
                self.dbg_probe("qa", self.qa, c.q_lora);
            }
            ke.rmsnorm(cu, self.qa, w.q_a_ln, self.qa, 1, c.q_lora as i32, c.rms_eps, st)?;
            mat_gemv(ke, cu, &w.q_b, self.qa, self.q, st)?;
            if dbg0 {
                self.dbg_probe("q_head0", self.q, c.nope + c.rope);
            }
            mat_gemv(ke, cu, &w.kv_a, self.h_norm, self.kva, st)?;
            if dbg0 {
                self.dbg_probe("kva", self.kva, c.kv_lora + c.rope);
            }
            ke.rmsnorm(cu, self.kva, w.kv_a_ln, self.ckv_norm, 1, c.kv_lora as i32, c.rms_eps, st)?;
            if dbg0 {
                self.dbg_probe("cn", self.ckv_norm, c.kv_lora);
            }
            // RoPE on q rope slices and the shared k_rope.
            ke.mla_rope(
                cu, self.q, heads, q_dim, c.nope as i32, (c.rope / 2) as i32,
                pos as i32, self.inv_freq_dev, 1.0, st,
            )?;
            ke.mla_rope(
                cu, self.kva, 1, qk, c.kv_lora as i32, (c.rope / 2) as i32,
                pos as i32, self.inv_freq_dev, 1.0, st,
            )?;
            // Cache row: [c_kv_norm | k_rope].
            ke.strided_copy(
                cu, self.ckv_norm, c.kv_lora as i32, 0,
                self.cache[l], qk, (pos as i32) * qk, 1, c.kv_lora as i32, st,
            )?;
            ke.strided_copy(
                cu, self.kva, qk, c.kv_lora as i32,
                self.cache[l], qk, (pos as i32) * qk + c.kv_lora as i32, 1, c.rope as i32, st,
            )?;
            // Absorbed query: q_eff = W_kvb_k^T q_nope, then append q_rope.
            ke.bf16_gemv_batch(
                cu, w.kt, (c.kv_lora * c.nope) as u64,
                self.q, q_dim, self.q_full, qk,
                c.kv_lora as i32, c.nope as i32, heads, st,
            )?;
            ke.strided_copy(
                cu, self.q, q_dim, c.nope as i32,
                self.q_full, qk, c.kv_lora as i32, heads, c.rope as i32, st,
            )?;
            ke.mla_attn_decode(
                cu, self.q_full, self.cache[l], self.ctx, self.attn_scratch,
                heads, qk, c.kv_lora as i32,
                (pos + 1) as i32, self.max_seq as i32, c.softmax_scale, st,
            )?;
            // Per-head V from the context, then output projection.
            ke.bf16_gemv_batch(
                cu, w.vw, (c.v_head * c.kv_lora) as u64,
                self.ctx, c.kv_lora as i32, self.attn_pre, c.v_head as i32,
                c.v_head as i32, c.kv_lora as i32, heads, st,
            )?;
            if dbg0 {
                self.dbg_probe("attn_pre", self.attn_pre, c.heads * c.v_head);
            }
            mat_gemv(ke, cu, &w.o, self.attn_pre, self.proj_out, st)?;
            ke.add(cu, self.h, self.proj_out, hid, st)?;
            if dbg0 {
                self.dbg_probe("h_after_attn", self.h, c.hidden);
            }

            // MLP.
            ke.rmsnorm(cu, self.h, w.post_ln, self.h_norm, 1, hid, c.rms_eps, st)?;
            if let Some(dense) = &w.dense {
                mat_gemv(ke, cu, &dense[0], self.h_norm, self.dn_gate, st)?;
                mat_gemv(ke, cu, &dense[1], self.h_norm, self.dn_up, st)?;
                ke.silu_mul(cu, self.dn_gate, self.dn_up, self.dn_gate, c.dense_inter as i32, st)?;
                mat_gemv(ke, cu, &dense[2], self.dn_gate, self.proj_out, st)?;
                ke.add(cu, self.h, self.proj_out, hid, st)?;
                if dbg0 {
                    self.dbg_probe("h_after_l0", self.h, c.hidden);
                }
                continue;
            }

            // Sigmoid router with bias-corrected selection.
            let router = w.router.as_ref().unwrap();
            mat_gemv(ke, cu, router, self.h_norm, self.router_logits, st)?;
            cu.dtoh_async(&mut self.router_host, self.router_logits, st)?;
            cu.sync_stream(st)?;
            let picks = sigmoid_topk(
                &f32_from_le(&self.router_host),
                w.router_bias.as_ref().unwrap(),
                c.top_k,
                c.norm_topk_prob,
                c.routed_scaling,
                self.min_expert_weight,
            );

            // Shared expert (always on) runs while experts fetch.
            let sh = w.shared.as_ref().unwrap();
            mat_gemv(ke, cu, &sh[0], self.h_norm, self.sh_gate, st)?;
            mat_gemv(ke, cu, &sh[1], self.h_norm, self.sh_up, st)?;
            ke.silu_mul(cu, self.sh_gate, self.sh_up, self.sh_gate, c.moe_inter as i32, st)?;
            mat_gemv(ke, cu, &sh[2], self.sh_gate, self.sh_out, st)?;

            // Kimi's top-k can exceed the per-layer slot count (huge experts,
            // small VRAM cache), so decode fetches in waves like prefill:
            // moe_reduce accumulates into h across waves.
            let pack_layer = (l - c.dense_layers) as u16;
            let slots = self.pager.as_ref().unwrap().slots_per_layer() as usize;
            let inter = c.moe_inter as i32;
            for wave in picks.chunks(slots.max(1)) {
                let ids: Vec<u16> = wave.iter().map(|p| p.0).collect();
                let handles = self.pager.as_ref().unwrap().request(pack_layer, &ids)?;
                for handle in &handles {
                    self.pager.as_ref().unwrap().wait_stream(handle, st)?;
                }
                let e = handles.len() as i32;
                let ptrs: Vec<u8> =
                    handles.iter().flat_map(|h| h.dev.to_le_bytes()).collect();
                let wts: Vec<u8> = wave.iter().flat_map(|p| p.1.to_le_bytes()).collect();
                cu.htod_async(self.d_expert_ptrs, &ptrs, st)?;
                cu.htod_async(self.d_expert_wts, &wts, st)?;
                ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.gate_off, self.h_norm, 0, self.gate_out, inter, hid, self.expert_group, e, st)?;
                ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.up_off, self.h_norm, 0, self.up_out, inter, hid, self.expert_group, e, st)?;
                ke.silu_mul(cu, self.gate_out, self.up_out, self.act_out, e * inter, st)?;
                ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.down_off, self.act_out, inter, self.expert_out, hid, inter, self.expert_group, e, st)?;
                ke.moe_reduce(cu, self.expert_out, self.d_expert_wts, self.h, e, hid, st)?;
                // Always eager: a deferred release is only drained by later
                // defer calls, and mixed eager/deferred layers deadlock —
                // pinned slots nobody ever frees (found via a hung
                // expert-dropping run). At Kimi's disk-bound speeds the
                // per-wave sync is noise.
                cu.sync_stream(st)?;
                for h in handles {
                    self.pager.as_ref().unwrap().release(h);
                }
            }
            ke.add(cu, self.h, self.sh_out, hid, st)?;
        }

        if !want_logits {
            return Ok(0);
        }
        ke.rmsnorm(cu, self.h, self.core.final_norm, self.h_norm, 1, hid, c.rms_eps, st)?;
        mat_gemv(ke, cu, &self.core.lm_head, self.h_norm, self.logits, st)?;
        cu.dtoh_async(&mut self.logits_host, self.logits, st)?;
        cu.sync_stream(st)?;
        let logits = f32_from_le(&self.logits_host);
        let mut best = 0u32;
        let mut bestv = f32::MIN;
        for (i, v) in logits.iter().enumerate() {
            if *v > bestv {
                bestv = *v;
                best = i as u32;
            }
        }
        self.pager.as_ref().unwrap().tick();
        Ok(best)
    }

    pub fn chunk_cap(&self) -> usize {
        self.chunk_cap
    }

    /// Union prefill on sequence slot 0 (see decode.rs).
    pub fn step_chunk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Result<u32> {
        let entries: Vec<(u32, usize, usize)> =
            tokens.iter().enumerate().map(|(i, t)| (*t, start_pos + i, 0)).collect();
        let out = self.step_multi(&entries, want_logits)?;
        Ok(out.last().copied().unwrap_or(0))
    }

    pub fn batch_cap(&self) -> usize {
        self.batch_cap
    }

    /// Slots per layer actually allocated (a VRAM reserve may clamp it).
    pub fn slots_per_layer(&self) -> u32 {
        self.pager.as_ref().map(|p| p.slots_per_layer()).unwrap_or(0)
    }

    /// Free and total VRAM on the device, across every process.
    pub fn mem_info(&self) -> Result<(u64, u64)> {
        self.cuda.mem_info()
    }

    /// Change the VRAM reserve; takes effect on the next `resize_cache`,
    /// which frees the current arena before the new one is sized.
    pub fn set_reserve_bytes(&mut self, bytes: u64) {
        self.reserve_bytes = bytes;
    }

    pub fn last_logits_multi(&self) -> &[Vec<f32>] {
        &self.logits_multi
    }

    /// General multi-entry step (see decode.rs): (token, position, sequence
    /// slot); same-slot entries ascending (prefill), distinct slots decode
    /// in lockstep sharing each layer's expert union.
    pub fn step_multi(
        &mut self,
        entries: &[(u32, usize, usize)],
        want_logits: bool,
    ) -> Result<Vec<u32>> {
        let n = entries.len();
        if n == 0 || n > self.chunk_cap {
            bail!("{n} entries (cap {})", self.chunk_cap);
        }
        for &(_, pos, seq) in entries {
            if pos >= self.max_seq {
                bail!("position {pos} exceeds max_seq {}", self.max_seq);
            }
            if seq >= self.batch_cap {
                bail!("sequence slot {seq} exceeds batch cap {}", self.batch_cap);
            }
        }
        let c = self.cfg.clone();
        let cu = Arc::clone(&self.cuda);
        let cu = &*cu;
        let ke = self.kernels;
        let ke = &ke;
        let st = self.stream;
        let hid = c.hidden as i32;
        let heads = c.heads as i32;
        let qk = (c.kv_lora + c.rope) as i32;
        let q_dim = (c.nope + c.rope) as i32;
        let h_at = |b: CUdeviceptr, t: usize| b + (t * c.hidden * 4) as u64;

        for (t, &(tok, _, _)) in entries.iter().enumerate() {
            let row = &self.core.embed_host
                [tok as usize * c.hidden * 2..(tok as usize + 1) * c.hidden * 2];
            for (i, ch) in row.chunks_exact(2).enumerate() {
                self.embed_row[i] = bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]]));
            }
            cu.htod_async(h_at(self.h_buf, t), f32_bytes(&self.embed_row), st)?;
            cu.sync_stream(st)?; // embed_row is reused next iteration
        }

        // Waves release eagerly, so a wave may use every slot in the layer —
        // but never more (request() rejects that outright).
        let wave = (self.pager.as_ref().unwrap().slots_per_layer() as usize).max(1);

        for l in 0..c.layers {
            let w = &self.core.layers[l];

            for (t, &(_, pos, seq)) in entries.iter().enumerate() {
                let ht = h_at(self.h_buf, t);
                let cache_l = self.cache[l] + (seq * self.max_seq * qk as usize * 4) as u64;
                ke.rmsnorm(cu, ht, w.input_ln, self.h_norm, 1, hid, c.rms_eps, st)?;
                mat_gemv(ke, cu, &w.q_a, self.h_norm, self.qa, st)?;
                ke.rmsnorm(cu, self.qa, w.q_a_ln, self.qa, 1, c.q_lora as i32, c.rms_eps, st)?;
                mat_gemv(ke, cu, &w.q_b, self.qa, self.q, st)?;
                mat_gemv(ke, cu, &w.kv_a, self.h_norm, self.kva, st)?;
                ke.rmsnorm(cu, self.kva, w.kv_a_ln, self.ckv_norm, 1, c.kv_lora as i32, c.rms_eps, st)?;
                ke.mla_rope(
                    cu, self.q, heads, q_dim, c.nope as i32, (c.rope / 2) as i32,
                    pos as i32, self.inv_freq_dev, 1.0, st,
                )?;
                ke.mla_rope(
                    cu, self.kva, 1, qk, c.kv_lora as i32, (c.rope / 2) as i32,
                    pos as i32, self.inv_freq_dev, 1.0, st,
                )?;
                ke.strided_copy(
                    cu, self.ckv_norm, c.kv_lora as i32, 0,
                    cache_l, qk, (pos as i32) * qk, 1, c.kv_lora as i32, st,
                )?;
                ke.strided_copy(
                    cu, self.kva, qk, c.kv_lora as i32,
                    cache_l, qk, (pos as i32) * qk + c.kv_lora as i32, 1, c.rope as i32, st,
                )?;
                ke.bf16_gemv_batch(
                    cu, w.kt, (c.kv_lora * c.nope) as u64,
                    self.q, q_dim, self.q_full, qk,
                    c.kv_lora as i32, c.nope as i32, heads, st,
                )?;
                ke.strided_copy(
                    cu, self.q, q_dim, c.nope as i32,
                    self.q_full, qk, c.kv_lora as i32, heads, c.rope as i32, st,
                )?;
                ke.mla_attn_decode(
                    cu, self.q_full, cache_l, self.ctx, self.attn_scratch,
                    heads, qk, c.kv_lora as i32,
                    (pos + 1) as i32, self.max_seq as i32, c.softmax_scale, st,
                )?;
                ke.bf16_gemv_batch(
                    cu, w.vw, (c.v_head * c.kv_lora) as u64,
                    self.ctx, c.kv_lora as i32, self.attn_pre, c.v_head as i32,
                    c.v_head as i32, c.kv_lora as i32, heads, st,
                )?;
                mat_gemv(ke, cu, &w.o, self.attn_pre, self.proj_out, st)?;
                ke.add(cu, ht, self.proj_out, hid, st)?;
                ke.rmsnorm(cu, ht, w.post_ln, h_at(self.hn_buf, t), 1, hid, c.rms_eps, st)?;
            }

            if let Some(dense) = &w.dense {
                for t in 0..n {
                    let x = h_at(self.hn_buf, t);
                    mat_gemv(ke, cu, &dense[0], x, self.dn_gate, st)?;
                    mat_gemv(ke, cu, &dense[1], x, self.dn_up, st)?;
                    ke.silu_mul(cu, self.dn_gate, self.dn_up, self.dn_gate, c.dense_inter as i32, st)?;
                    mat_gemv(ke, cu, &dense[2], self.dn_gate, self.proj_out, st)?;
                    ke.add(cu, h_at(self.h_buf, t), self.proj_out, hid, st)?;
                }
                continue;
            }

            // Shared expert + router logits per token.
            let sh = w.shared.as_ref().unwrap();
            let router = w.router.as_ref().unwrap();
            for t in 0..n {
                let x = h_at(self.hn_buf, t);
                mat_gemv(ke, cu, &sh[0], x, self.sh_gate, st)?;
                mat_gemv(ke, cu, &sh[1], x, self.sh_up, st)?;
                ke.silu_mul(cu, self.sh_gate, self.sh_up, self.sh_gate, c.moe_inter as i32, st)?;
                mat_gemv(ke, cu, &sh[2], self.sh_gate, self.sh_out, st)?;
                ke.add(cu, h_at(self.h_buf, t), self.sh_out, hid, st)?;
                mat_gemv(ke, cu, router, x, self.router_buf + (t * c.experts * 4) as u64, st)?;
            }
            let span = n * c.experts * 4;
            cu.dtoh_async(&mut self.router_chunk_host[..span], self.router_buf, st)?;
            cu.sync_stream(st)?;
            let all = f32_from_le(&self.router_chunk_host[..span]);
            let bias = w.router_bias.as_ref().unwrap();
            let picks: Vec<Vec<(u16, f32)>> = (0..n)
                .map(|t| {
                    sigmoid_topk(
                        &all[t * c.experts..(t + 1) * c.experts],
                        bias,
                        c.top_k,
                        c.norm_topk_prob,
                        c.routed_scaling,
                        self.min_expert_weight,
                    )
                })
                .collect();
            let mut union: Vec<u16> = picks.iter().flatten().map(|p| p.0).collect();
            union.sort_unstable();
            union.dedup();

            let pack_layer = (l - c.dense_layers) as u16;
            for wave_ids in union.chunks(wave) {
                let handles = self.pager.as_ref().unwrap().request(pack_layer, wave_ids)?;
                for h in &handles {
                    self.pager.as_ref().unwrap().wait_stream(h, st)?;
                }
                let dev_of = |id: u16| {
                    handles.iter().find(|h| h.expert == id).map(|h| h.dev).unwrap()
                };
                let inter = c.moe_inter as i32;
                for t in 0..n {
                    let sub: Vec<&(u16, f32)> =
                        picks[t].iter().filter(|p| wave_ids.contains(&p.0)).collect();
                    if sub.is_empty() {
                        continue;
                    }
                    let e = sub.len() as i32;
                    let ptrs: Vec<u8> =
                        sub.iter().flat_map(|p| dev_of(p.0).to_le_bytes()).collect();
                    let wts: Vec<u8> = sub.iter().flat_map(|p| p.1.to_le_bytes()).collect();
                    cu.htod_async(self.d_expert_ptrs, &ptrs, st)?;
                    cu.htod_async(self.d_expert_wts, &wts, st)?;
                    let x = h_at(self.hn_buf, t);
                    ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.gate_off, x, 0, self.gate_out, inter, hid, self.expert_group, e, st)?;
                    ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.up_off, x, 0, self.up_out, inter, hid, self.expert_group, e, st)?;
                    ke.silu_mul(cu, self.gate_out, self.up_out, self.act_out, e * inter, st)?;
                    ke.q4g64_gemv_batch(cu, self.d_expert_ptrs, self.down_off, self.act_out, inter, self.expert_out, hid, inter, self.expert_group, e, st)?;
                    ke.moe_reduce(cu, self.expert_out, self.d_expert_wts, h_at(self.h_buf, t), e, hid, st)?;
                }
                // Eager release (see decode.rs): deferring would deadlock
                // the next wave's request().
                cu.sync_stream(st)?;
                for h in handles {
                    self.pager.as_ref().unwrap().release(h);
                }
            }
        }

        self.pager.as_ref().unwrap().tick();
        if !want_logits {
            return Ok(vec![0; n]);
        }
        self.logits_multi.clear();
        let mut out = Vec::with_capacity(n);
        for (t, &(_, _, seq)) in entries.iter().enumerate() {
            let is_last_for_slot =
                entries.iter().skip(t + 1).all(|&(_, _, s2)| s2 != seq);
            if !is_last_for_slot {
                self.logits_multi.push(Vec::new());
                out.push(0);
                continue;
            }
            ke.rmsnorm(cu, h_at(self.h_buf, t), self.core.final_norm, self.h_norm, 1, hid, c.rms_eps, st)?;
            mat_gemv(ke, cu, &self.core.lm_head, self.h_norm, self.logits, st)?;
            cu.dtoh_async(&mut self.logits_host, self.logits, st)?;
            cu.sync_stream(st)?;
            let logits = f32_from_le(&self.logits_host);
            let (mut best, mut bestv) = (0u32, f32::MIN);
            for (i, v) in logits.iter().enumerate() {
                if *v > bestv {
                    bestv = *v;
                    best = i as u32;
                }
            }
            self.logits_multi.push(logits);
            out.push(best);
        }
        Ok(out)
    }

    pub fn pager_metrics(&self) -> llmpager_cuda::pager::Metrics {
        self.pager.as_ref().unwrap().metrics()
    }

    pub fn expert_stats(&self) -> Vec<u64> {
        self.pager.as_ref().unwrap().expert_stats()
    }

    pub fn prewarm(&self, counts: &[u64]) -> Result<usize> {
        self.pager.as_ref().unwrap().prewarm(counts)
    }

    pub fn last_logits(&self) -> Vec<f32> {
        f32_from_le(&self.logits_host)
    }

    /// Bytes of MLA cache one token occupies across all layers.
    pub fn kv_bytes_per_token(&self) -> usize {
        self.cfg.layers * (self.cfg.kv_lora + self.cfg.rope) * 4
    }

    fn kv_geometry(&self, seq: usize, len: usize) -> Result<(usize, u64)> {
        if seq >= self.batch_cap {
            bail!("sequence slot {seq} exceeds batch cap {}", self.batch_cap);
        }
        if len > self.max_seq {
            bail!("kv length {len} exceeds max_seq {}", self.max_seq);
        }
        let qk = self.cfg.kv_lora + self.cfg.rope;
        Ok((len * qk * 4, (seq * self.max_seq * qk * 4) as u64))
    }

    /// Copy slot `seq`'s MLA cache for positions [0, len) to host memory.
    /// The compressed cache is [max_seq, kv_lora + rope] per layer, so a
    /// prefix is one contiguous run per layer.
    pub fn kv_export(&self, seq: usize, len: usize) -> Result<Vec<u8>> {
        let (run, slot_off) = self.kv_geometry(seq, len)?;
        let mut out = vec![0u8; self.cfg.layers * run];
        for l in 0..self.cfg.layers {
            self.cuda.dtoh_async(
                &mut out[l * run..(l + 1) * run],
                self.cache[l] + slot_off,
                self.stream,
            )?;
        }
        self.cuda.sync_stream(self.stream)?;
        Ok(out)
    }

    /// Inverse of `kv_export`.
    pub fn kv_import(&mut self, seq: usize, len: usize, blob: &[u8]) -> Result<()> {
        let (run, slot_off) = self.kv_geometry(seq, len)?;
        let want = self.cfg.layers * run;
        if blob.len() != want {
            bail!("kv blob is {} bytes, expected {want} for {len} tokens", blob.len());
        }
        for l in 0..self.cfg.layers {
            self.cuda.htod_async(
                self.cache[l] + slot_off,
                &blob[l * run..(l + 1) * run],
                self.stream,
            )?;
        }
        self.cuda.sync_stream(self.stream)?;
        Ok(())
    }

    #[allow(dead_code)] // eager-release everywhere now; kept for symmetry
    fn defer_release(&mut self, handles: Vec<ExpertHandle>) -> Result<()> {
        while let Some((ev_idx, _)) = self.pending_release.front() {
            if self.cuda.event_done(self.release_events[*ev_idx])? {
                let (_, done) = self.pending_release.pop_front().unwrap();
                for h in done {
                    self.pager.as_ref().unwrap().release(h);
                }
            } else {
                break;
            }
        }
        if self.pending_release.len() == self.release_events.len() {
            let (ev_idx, done) = self.pending_release.pop_front().unwrap();
            self.cuda.sync_event(self.release_events[ev_idx])?;
            for h in done {
                self.pager.as_ref().unwrap().release(h);
            }
        }
        let ev = self.next_event;
        self.next_event = (self.next_event + 1) % self.release_events.len();
        self.cuda.record_event(self.release_events[ev], self.stream)?;
        self.pending_release.push_back((ev, handles));
        Ok(())
    }
}

/// Sigmoid scores; top-k selected by score + per-expert bias; returned
/// weights are the *unbiased* scores of the picks, renormalized when
/// `renorm`, scaled by `scaling`. Picks under `min_weight` (after scaling)
/// are dropped — a fetch-traffic knob measured to be near-lossless at
/// small thresholds.
fn sigmoid_topk(
    logits: &[f32],
    bias: &[f32],
    k: usize,
    renorm: bool,
    scaling: f32,
    min_weight: f32,
) -> Vec<(u16, f32)> {
    let scores: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let mut ranked: Vec<usize> = (0..scores.len()).collect();
    ranked.sort_by(|&a, &b| {
        (scores[b] + bias[b]).partial_cmp(&(scores[a] + bias[a])).unwrap()
    });
    ranked.truncate(k);
    let mut picks: Vec<(u16, f32)> =
        ranked.into_iter().map(|i| (i as u16, scores[i])).collect();
    if renorm {
        let s: f32 = picks.iter().map(|p| p.1).sum();
        if s > 0.0 {
            for p in &mut picks {
                p.1 /= s;
            }
        }
    }
    for p in &mut picks {
        p.1 *= scaling;
    }
    if min_weight > 0.0 {
        picks.retain(|p| p.1 >= min_weight);
    }
    picks
}

unsafe impl Send for KimiDecoder {}

impl Drop for KimiDecoder {
    fn drop(&mut self) {
        let _ = self.cuda.sync();
        let mut ptrs = vec![
            self.h, self.h_norm, self.qa, self.q, self.kva, self.ckv_norm,
            self.q_full, self.ctx, self.attn_pre, self.proj_out,
            self.attn_scratch, self.inv_freq_dev, self.router_logits,
            self.gate_out, self.up_out, self.act_out, self.expert_out,
            self.d_expert_ptrs, self.d_expert_wts, self.sh_gate, self.sh_up,
            self.sh_out, self.dn_gate, self.dn_up, self.logits,
            self.h_buf, self.hn_buf, self.router_buf,
        ];
        ptrs.extend(&self.cache);
        ptrs.extend(self.core.device_ptrs());
        ptrs.sort_unstable();
        ptrs.dedup();
        for p in ptrs {
            self.cuda.free_device(p);
        }
        for e in &self.release_events {
            self.cuda.destroy_event(*e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sigmoid_topk;

    #[test]
    fn bias_affects_selection_not_weights() {
        // Expert 0 has the best raw score; expert 2's bias promotes it.
        let logits = [2.0f32, -2.0, 1.0, -3.0];
        let bias = [0.0f32, 0.0, 5.0, 0.0];
        let picks = sigmoid_topk(&logits, &bias, 2, true, 2.0, 0.0);
        assert_eq!(picks.len(), 2);
        assert_eq!(picks[0].0, 2); // promoted by bias
        assert_eq!(picks[1].0, 0);
        // Weights come from unbiased scores, renormalized then scaled by 2.
        let s0 = 1.0 / (1.0 + (-2.0f32).exp());
        let s2 = 1.0 / (1.0 + (-1.0f32).exp());
        let w2 = s2 / (s0 + s2) * 2.0;
        assert!((picks[0].1 - w2).abs() < 1e-5);
    }

    #[test]
    fn min_weight_drops_tail() {
        let logits = [4.0f32, 4.0, -6.0];
        let bias = [0.0f32; 3];
        let picks = sigmoid_topk(&logits, &bias, 3, true, 1.0, 0.05);
        assert_eq!(picks.len(), 2, "tiny-weight expert should be dropped");
    }
}
