//! Kimi K2.6 (kimi_k25) conversion: DeepseekV3 text stack inside a
//! multimodal wrapper.
//!
//! - Routed experts ship as QAT int4 group-32 in compressed-tensors form:
//!   `weight_packed` I32 `[rows, cols/8]` (8 signed nibbles per word, value
//!   k of a word at bits 4k), `weight_scale` BF16 `[rows, cols/32]`,
//!   `weight_shape` I32 `[2]`. We repack bit-exactly into our q4 g32 blob
//!   layout (nibble = signed value + 8; scales converted bf16 -> f16).
//! - Everything else in `language_model.` (MLA attention, shared expert,
//!   dense layer-0 MLP, norms, embeddings, lm_head, router gates) is BF16
//!   and goes verbatim into the core file with the `language_model.` prefix
//!   stripped, so runtime names look like any other model's.
//! - `vision_tower.` / `mm_projector.` are dropped (text-only serving).
//! - Layer 0 is dense (`first_k_dense_replace`), so the pack holds only the
//!   MoE layers; `moe_layer_offset` in the pack config maps pack layer i to
//!   model layer i + offset.

use std::path::Path;

use anyhow::{bail, Context, Result};
use llmpager_core::pack::{PackMeta, PackWriter};
use llmpager_core::quant::{f16_bits_to_f32, f32_to_f16_bits, q4_bytes};

use crate::{Checkpoint, ConvertReport, write_safetensors_renamed, BLOB_HEADER_BYTES};

const PREFIX: &str = "language_model.";
const KIMI_GROUP: usize = 32;

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// `language_model.model.layers.L.mlp.experts.E.{proj}.weight_packed` ->
/// (layer, expert, proj index).
fn parse_expert_packed(name: &str) -> Option<(u16, u16, usize)> {
    let rest = name.strip_prefix("language_model.model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (expert, rest) = rest.split_once('.')?;
    let proj = match rest {
        "gate_proj.weight_packed" => 0,
        "up_proj.weight_packed" => 1,
        "down_proj.weight_packed" => 2,
        _ => return None,
    };
    Some((layer.parse().ok()?, expert.parse().ok()?, proj))
}

pub fn convert_kimi(model_dir: &Path, out_pack: &Path, out_core: &Path) -> Result<ConvertReport> {
    let ckpt = Checkpoint::open(model_dir)?;
    let text = ckpt.config["text_config"].clone();
    if text.is_null() {
        bail!("kimi convert: config.json has no text_config");
    }
    let layers = text["num_hidden_layers"].as_u64().context("num_hidden_layers")? as u16;
    let experts = text["n_routed_experts"].as_u64().context("n_routed_experts")? as u16;
    let dense_layers = text["first_k_dense_replace"].as_u64().unwrap_or(0) as u16;
    let moe_layers = layers - dense_layers;

    // Index the packed expert tensors; everything else under language_model.
    // that is not expert metadata goes to the core.
    let mut expert_base: Vec<Vec<[Option<String>; 3]>> =
        vec![vec![[None, None, None]; experts as usize]; moe_layers as usize];
    let mut core: Vec<(String, String)> = Vec::new(); // (src, renamed)
    for name in ckpt.tensors.keys() {
        if let Some((l, e, p)) = parse_expert_packed(name) {
            if l < dense_layers || l >= layers || e >= experts {
                bail!("expert tensor out of range: {name}");
            }
            let base = name.strip_suffix("_packed").unwrap().to_string();
            expert_base[(l - dense_layers) as usize][e as usize][p] = Some(base);
            continue;
        }
        if !name.starts_with(PREFIX) {
            continue; // vision_tower / mm_projector
        }
        // Skip the experts' companion tensors (scale/shape) and anything
        // else under mlp.experts (routed experts live in the pack).
        if name.contains(".mlp.experts.") {
            continue;
        }
        core.push((name.clone(), name[PREFIX.len()..].to_string()));
    }

    let core_bytes = write_safetensors_renamed(&ckpt, &core, out_core)
        .context("writing resident core")?;

    // Pack config: text_config plus the layer mapping the runtime needs.
    let mut cfg = text.clone();
    cfg["moe_layer_offset"] = serde_json::json!(dense_layers);
    let meta = PackMeta {
        model: "kimi_k2.6".into(),
        num_layers: moe_layers,
        experts_per_layer: experts,
        dtype: "q4g32-gud".into(),
        config: cfg,
    };
    let mut writer = PackWriter::create(out_pack, meta)?;
    let mut max_scale_err = 0.0f32;

    for l in 0..moe_layers {
        let blobs: Vec<Result<(Vec<u8>, f32)>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..experts)
                .map(|e| {
                    let names = &expert_base[l as usize][e as usize];
                    let ckpt = &ckpt;
                    s.spawn(move || build_kimi_blob(ckpt, names, l, e))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for b in blobs {
            let (blob, serr) = b?;
            max_scale_err = max_scale_err.max(serr);
            writer.add_blob(&blob)?;
        }
    }
    writer.finish()?;
    let pack_bytes = std::fs::metadata(out_pack)?.len();

    Ok(ConvertReport {
        layers: moe_layers,
        experts,
        pack_bytes,
        core_bytes,
        core_tensors: core.len(),
        max_quant_err: max_scale_err,
    })
}

/// Repack one expert's three projections. Returns (blob, worst relative
/// scale error from the bf16 -> f16 conversion; nibbles are bit-exact).
fn build_kimi_blob(
    ckpt: &Checkpoint,
    names: &[Option<String>; 3],
    layer: u16,
    expert: u16,
) -> Result<(Vec<u8>, f32)> {
    let mut dims = [[0u32; 2]; 3];
    let mut regions: Vec<Vec<u8>> = Vec::with_capacity(3);
    let mut max_scale_err = 0.0f32;

    for (p, base) in names.iter().enumerate() {
        let base = base.as_ref().with_context(|| {
            format!("missing packed expert (moe layer {layer}, expert {expert}, proj {p})")
        })?;
        let packed_loc = ckpt
            .tensors
            .get(&format!("{base}_packed"))
            .with_context(|| format!("{base}_packed not found"))?;
        let scale_loc = ckpt
            .tensors
            .get(&format!("{base}_scale"))
            .with_context(|| format!("{base}_scale not found"))?;
        if packed_loc.dtype != "I32" || scale_loc.dtype != "BF16" {
            bail!("{base}: unexpected dtypes {} / {}", packed_loc.dtype, scale_loc.dtype);
        }
        // Authoritative dims come from weight_shape when present, else from
        // the packed tensor ([rows, cols/8]).
        let (rows, cols) = match ckpt.tensors.get(&format!("{base}_shape")) {
            Some(shape_loc) => {
                let raw = ckpt.read_raw(shape_loc)?;
                (
                    i32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize,
                    i32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize,
                )
            }
            None => (packed_loc.shape[0], packed_loc.shape[1] * 8),
        };
        if packed_loc.shape != vec![rows, cols / 8] {
            bail!("{base}: packed shape {:?} != [{rows}, {}]", packed_loc.shape, cols / 8);
        }
        if scale_loc.shape != vec![rows, cols / KIMI_GROUP] {
            bail!("{base}: scale shape {:?} != [{rows}, {}]", scale_loc.shape, cols / KIMI_GROUP);
        }
        dims[p] = [rows as u32, cols as u32];

        let packed = ckpt.read_raw(packed_loc)?;
        let scales_raw = ckpt.read_raw(scale_loc)?;
        let mut buf = vec![0u8; q4_bytes(rows, cols, KIMI_GROUP)];
        let groups_per_row = cols / KIMI_GROUP;
        let scales_len = rows * groups_per_row * 2;
        {
            let (sdst, ddst) = buf.split_at_mut(scales_len);
            // Scales: bf16 -> f16, same row-major order.
            for (i, c) in scales_raw.chunks_exact(2).enumerate() {
                let s = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                let h = f32_to_f16_bits(s);
                sdst[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
                let back = f16_bits_to_f32(h);
                if s != 0.0 {
                    max_scale_err = max_scale_err.max(((s - back) / s).abs());
                }
            }
            // Nibbles: word k holds signed values 8k..8k+7, value j of the
            // word at bits 4j. Ours: byte i holds values 2i (low) 2i+1
            // (high), stored as value+8.
            for (w, c) in packed.chunks_exact(4).enumerate() {
                let word = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let out = &mut ddst[w * 4..w * 4 + 4];
                for i in 0..4 {
                    let r0 = ((word >> (8 * i)) & 0xF) as u8;
                    let r1 = ((word >> (8 * i + 4)) & 0xF) as u8;
                    out[i] = ((r0 + 8) & 0xF) | (((r1 + 8) & 0xF) << 4);
                }
            }
        }
        regions.push(buf);
    }

    let mut blob =
        Vec::with_capacity(BLOB_HEADER_BYTES + regions.iter().map(Vec::len).sum::<usize>());
    for d in dims {
        blob.extend_from_slice(&d[0].to_le_bytes());
        blob.extend_from_slice(&d[1].to_le_bytes());
    }
    blob.extend_from_slice(&(KIMI_GROUP as u32).to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    for r in &regions {
        blob.extend_from_slice(r);
    }
    Ok((blob, max_scale_err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmpager_core::pack::PackReader;
    use llmpager_core::quant::q4_dequantize;
    use std::path::PathBuf;

    fn f32_to_bf16(x: f32) -> u16 {
        ((x.to_bits() + 0x8000) >> 16) as u16
    }

    /// Tiny kimi-shaped checkpoint: 3 layers (1 dense), 2 experts,
    /// hidden 64, moe inter 32, plus vision tensors that must be skipped.
    fn write_kimi_checkpoint(dir: &Path) -> Result<()> {
        let (layers, dense, experts, hidden, inter) = (3usize, 1usize, 2usize, 64usize, 32usize);
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "kimi_k25",
                "text_config": {
                    "model_type": "kimi_k2",
                    "num_hidden_layers": layers,
                    "n_routed_experts": experts,
                    "first_k_dense_replace": dense,
                    "hidden_size": hidden,
                    "moe_intermediate_size": inter,
                    "num_experts_per_tok": 2,
                },
            }))?,
        )?;

        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        let mut rng = 7u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut add = |name: String,
                       dtype: &str,
                       shape: Vec<usize>,
                       bytes: Vec<u8>,
                       header: &mut serde_json::Map<String, serde_json::Value>,
                       data: &mut Vec<u8>| {
            let start = data.len();
            data.extend_from_slice(&bytes);
            header.insert(
                name,
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, start + bytes.len()]}),
            );
        };

        // Core-ish tensors (bf16) + a vision tensor to be dropped.
        for (name, n) in [
            ("language_model.model.embed_tokens.weight".to_string(), 8 * hidden),
            ("language_model.model.layers.0.mlp.gate_proj.weight".to_string(), inter * hidden),
            ("language_model.model.layers.1.mlp.shared_experts.gate_proj.weight".to_string(), inter * hidden),
            ("vision_tower.encoder.blocks.0.attn.weight".to_string(), 16),
        ] {
            let bytes: Vec<u8> = (0..n)
                .flat_map(|_| f32_to_bf16((next() % 1000) as f32 / 1000.0 - 0.5).to_le_bytes())
                .collect();
            add(name, "BF16", vec![n], bytes, &mut header, &mut data);
        }

        // Packed experts for MoE layers 1..3: gate/up [inter, hidden],
        // down [hidden, inter].
        for l in dense..layers {
            for e in 0..experts {
                for (proj, rows, cols) in [
                    ("gate_proj", inter, hidden),
                    ("up_proj", inter, hidden),
                    ("down_proj", hidden, inter),
                ] {
                    let base = format!(
                        "language_model.model.layers.{l}.mlp.experts.{e}.{proj}.weight"
                    );
                    let words: Vec<u8> = (0..rows * cols / 8)
                        .flat_map(|_| (next() as u32).to_le_bytes())
                        .collect();
                    add(format!("{base}_packed"), "I32", vec![rows, cols / 8], words, &mut header, &mut data);
                    let scales: Vec<u8> = (0..rows * cols / 32)
                        .flat_map(|_| {
                            f32_to_bf16(((next() % 900) + 100) as f32 / 10000.0).to_le_bytes()
                        })
                        .collect();
                    add(format!("{base}_scale"), "BF16", vec![rows, cols / 32], scales, &mut header, &mut data);
                    let shape: Vec<u8> = [(rows as i32).to_le_bytes(), (cols as i32).to_le_bytes()]
                        .concat();
                    add(format!("{base}_shape"), "I32", vec![2], shape, &mut header, &mut data);
                }
            }
        }

        let mut hdr = serde_json::to_vec(&serde_json::Value::Object(header))?;
        while hdr.len() % 8 != 0 {
            hdr.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&data);
        std::fs::write(dir.join("model.safetensors"), out)?;
        Ok(())
    }

    #[test]
    fn kimi_repack_round_trip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_kimi_checkpoint(dir.path())?;
        let pack_path: PathBuf = dir.path().join("k.llmpk");
        let core_path: PathBuf = dir.path().join("k.core.safetensors");
        let report = convert_kimi(dir.path(), &pack_path, &core_path)?;
        assert_eq!((report.layers, report.experts), (2, 2)); // 3 layers - 1 dense
        assert_eq!(report.core_tensors, 3); // embed + dense gate + shared gate
        assert!(report.max_quant_err < 0.005, "scale err {}", report.max_quant_err);

        let r = PackReader::open(&pack_path)?;
        assert_eq!(r.meta().dtype, "q4g32-gud");
        assert_eq!(r.meta().config["moe_layer_offset"], 1);

        // Dequant our repacked gate_proj of (pack layer 1 = model layer 2,
        // expert 1) and compare against the compressed-tensors reference.
        let mut buf = vec![0u8; (r.entry(1, 1).nbytes as usize).div_ceil(4096) * 4096];
        r.read_blob_into(1, 1, &mut buf)?;
        let (rows, cols) = (
            u32::from_le_bytes(buf[0..4].try_into()?) as usize,
            u32::from_le_bytes(buf[4..8].try_into()?) as usize,
        );
        assert_eq!((rows, cols), (32, 64));
        let group = u32::from_le_bytes(buf[24..28].try_into()?) as usize;
        assert_eq!(group, 32);
        let region = &buf[BLOB_HEADER_BYTES..BLOB_HEADER_BYTES + q4_bytes(rows, cols, 32)];
        let mut ours = vec![0f32; rows * cols];
        q4_dequantize(region, rows, cols, 32, &mut ours)?;

        let ckpt = Checkpoint::open(dir.path())?;
        let base = "language_model.model.layers.2.mlp.experts.1.gate_proj.weight";
        let packed = ckpt.read_raw(&ckpt.tensors[&format!("{base}_packed")])?;
        let scales = ckpt.read_raw(&ckpt.tensors[&format!("{base}_scale")])?;
        for r_ in 0..rows {
            for c in 0..cols {
                let vi = r_ * cols + c;
                let word =
                    u32::from_le_bytes(packed[vi / 8 * 4..vi / 8 * 4 + 4].try_into().unwrap());
                let raw = ((word >> (4 * (vi % 8))) & 0xF) as i32;
                let signed = if raw >= 8 { raw - 16 } else { raw };
                let sidx = (r_ * (cols / 32) + c / 32) * 2;
                let scale = bf16_to_f32(u16::from_le_bytes([scales[sidx], scales[sidx + 1]]));
                let want = signed as f32 * scale;
                let got = ours[vi];
                // f16 scale rounding is the only allowed difference.
                assert!(
                    (want - got).abs() <= scale.abs() * 8.0 * 0.005 + 1e-7,
                    "({r_},{c}): want {want} got {got}"
                );
            }
        }

        // Core: renamed, vision dropped.
        let raw = std::fs::read(&core_path)?;
        let hlen = u64::from_le_bytes(raw[..8].try_into()?) as usize;
        let chdr: serde_json::Value = serde_json::from_slice(&raw[8..8 + hlen])?;
        assert!(chdr.get("model.embed_tokens.weight").is_some());
        assert!(chdr.get("model.layers.1.mlp.shared_experts.gate_proj.weight").is_some());
        assert!(
            chdr.as_object().unwrap().keys().all(|k| !k.contains("vision")),
            "vision tensor leaked into core"
        );
        Ok(())
    }
}
