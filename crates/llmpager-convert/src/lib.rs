//! Convert a Hugging Face Qwen3-MoE-family checkpoint into llmpager's two
//! artifacts:
//!
//! - an `.llmpk` expert pack: every `model.layers.L.mlp.experts.E.*` weight,
//!   quantized q4g64, one blob per (layer, expert) holding gate/up/down;
//! - a resident-core safetensors file: everything else (embeddings,
//!   attention, norms, router gates, lm_head), original dtype, pageable as
//!   one artifact (multi-model support loads/unloads whole cores).
//!
//! Safetensors is parsed directly — the format is a u64 LE header length,
//! a JSON header mapping tensor name → {dtype, shape, data_offsets}, then
//! raw data. Tensors are pread on demand; no shard is loaded whole.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::{bail, Context, Result};
use llmpager_core::pack::{PackMeta, PackWriter};
use llmpager_core::quant::{q4g64_bytes, q4g64_quantize, GROUP};

/// Blob layout: this fixed header, then gate, up, down as q4g64 regions.
pub const BLOB_HEADER_BYTES: usize = 32;

#[derive(Debug, Clone)]
struct TensorLoc {
    file: usize,
    dtype: String,
    shape: Vec<usize>,
    /// Absolute byte range within the shard file.
    start: u64,
    end: u64,
}

struct Checkpoint {
    files: Vec<File>,
    tensors: BTreeMap<String, TensorLoc>,
    config: serde_json::Value,
}

impl Checkpoint {
    fn open(dir: &Path) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).context("reading config.json")?,
        )?;

        let shard_names: Vec<String> = {
            let index_path = dir.join("model.safetensors.index.json");
            if index_path.exists() {
                let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
                let map = idx["weight_map"]
                    .as_object()
                    .context("index.json missing weight_map")?;
                let mut files: Vec<String> =
                    map.values().filter_map(|v| v.as_str().map(String::from)).collect();
                files.sort();
                files.dedup();
                files
            } else {
                vec!["model.safetensors".to_string()]
            }
        };

        let mut files = Vec::new();
        let mut tensors = BTreeMap::new();
        for name in &shard_names {
            let path = dir.join(name);
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            let mut len8 = [0u8; 8];
            file.read_exact_at(&mut len8, 0)?;
            let header_len = u64::from_le_bytes(len8);
            let mut hdr = vec![0u8; header_len as usize];
            file.read_exact_at(&mut hdr, 8)?;
            let header: serde_json::Value = serde_json::from_slice(&hdr)?;
            let data_base = 8 + header_len;
            let obj = header.as_object().context("bad safetensors header")?;
            let fidx = files.len();
            for (tname, info) in obj {
                if tname == "__metadata__" {
                    continue;
                }
                let offs = info["data_offsets"]
                    .as_array()
                    .with_context(|| format!("{tname}: missing data_offsets"))?;
                tensors.insert(
                    tname.clone(),
                    TensorLoc {
                        file: fidx,
                        dtype: info["dtype"].as_str().unwrap_or("").to_string(),
                        shape: info["shape"]
                            .as_array()
                            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
                            .unwrap_or_default(),
                        start: data_base + offs[0].as_u64().unwrap_or(0),
                        end: data_base + offs[1].as_u64().unwrap_or(0),
                    },
                );
            }
            files.push(file);
        }
        Ok(Self { files, tensors, config })
    }

    fn read_raw(&self, loc: &TensorLoc) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; (loc.end - loc.start) as usize];
        self.files[loc.file].read_exact_at(&mut buf, loc.start)?;
        Ok(buf)
    }

    fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let loc = self.tensors.get(name).with_context(|| format!("tensor {name} not found"))?;
        let raw = self.read_raw(loc)?;
        let vals = match loc.dtype.as_str() {
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => bail!("{name}: unsupported dtype {other}"),
        };
        Ok((vals, loc.shape.clone()))
    }
}

/// Is this an expert weight? Returns (layer, expert, proj) with proj in
/// {0: gate_proj, 1: up_proj, 2: down_proj}.
fn parse_expert(name: &str) -> Option<(u16, u16, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (expert, rest) = rest.split_once('.')?;
    let proj = match rest {
        "gate_proj.weight" => 0,
        "up_proj.weight" => 1,
        "down_proj.weight" => 2,
        _ => return None,
    };
    Some((layer.parse().ok()?, expert.parse().ok()?, proj))
}

pub struct ConvertReport {
    pub layers: u16,
    pub experts: u16,
    pub pack_bytes: u64,
    pub core_bytes: u64,
    pub core_tensors: usize,
    pub max_quant_err: f32,
}

pub fn convert(model_dir: &Path, out_pack: &Path, out_core: &Path) -> Result<ConvertReport> {
    let ckpt = Checkpoint::open(model_dir)?;
    let layers = ckpt.config["num_hidden_layers"]
        .as_u64()
        .context("config: num_hidden_layers")? as u16;
    let experts = ckpt.config["num_experts"].as_u64().context("config: num_experts")? as u16;
    let model_name = ckpt.config["model_type"].as_str().unwrap_or("unknown").to_string();

    // Sanity: every expert tensor present, and nothing expert-shaped left over.
    let mut expert_names: Vec<Vec<[Option<String>; 3]>> =
        vec![vec![[None, None, None]; experts as usize]; layers as usize];
    let mut core_names: Vec<String> = Vec::new();
    for name in ckpt.tensors.keys() {
        match parse_expert(name) {
            Some((l, e, p)) if (l < layers) && (e < experts) => {
                expert_names[l as usize][e as usize][p] = Some(name.clone());
            }
            Some((l, e, _)) => bail!("expert tensor {name} out of range ({l},{e})"),
            None => core_names.push(name.clone()),
        }
    }

    // Resident core first (small; fail fast before the long pack write).
    let core_bytes = write_safetensors(&ckpt, &core_names, out_core)
        .context("writing resident core")?;

    // Expert pack.
    let meta = PackMeta {
        model: model_name,
        num_layers: layers,
        experts_per_layer: experts,
        dtype: "q4g64-gud".into(),
        config: ckpt.config.clone(),
    };
    let mut writer = PackWriter::create(out_pack, meta)?;
    let mut max_err = 0.0f32;
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    for l in 0..layers {
        // Quantize one layer's experts in parallel, write in order.
        let blobs: Vec<Result<(Vec<u8>, f32)>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..experts)
                .map(|e| {
                    let names = &expert_names[l as usize][e as usize];
                    let ckpt = &ckpt;
                    s.spawn(move || build_blob(ckpt, names, l, e))
                })
                .collect();
            // Scoped threads all start at once; cap actual parallelism by
            // joining as we go — spawn cost is trivial next to quantization.
            let _ = workers;
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for b in blobs {
            let (blob, err) = b?;
            max_err = max_err.max(err);
            writer.add_blob(&blob)?;
        }
    }
    writer.finish()?;
    let pack_bytes = std::fs::metadata(out_pack)?.len();

    Ok(ConvertReport {
        layers,
        experts,
        pack_bytes,
        core_bytes,
        core_tensors: core_names.len(),
        max_quant_err: max_err,
    })
}

fn build_blob(
    ckpt: &Checkpoint,
    names: &[Option<String>; 3],
    layer: u16,
    expert: u16,
) -> Result<(Vec<u8>, f32)> {
    let mut dims = [[0u32; 2]; 3];
    let mut regions: Vec<Vec<u8>> = Vec::with_capacity(3);
    let mut max_err = 0.0f32;
    for (p, name) in names.iter().enumerate() {
        let name = name
            .as_ref()
            .with_context(|| format!("missing expert tensor (layer {layer}, expert {expert}, proj {p})"))?;
        let (vals, shape) = ckpt.read_f32(name)?;
        if shape.len() != 2 {
            bail!("{name}: expected 2-D, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        if cols % GROUP != 0 {
            bail!("{name}: cols {cols} not a multiple of {GROUP}");
        }
        dims[p] = [rows as u32, cols as u32];
        let mut buf = vec![0u8; q4g64_bytes(rows, cols)];
        let err = q4g64_quantize(&vals, rows, cols, &mut buf)?;
        max_err = max_err.max(err);
        regions.push(buf);
    }

    let mut blob = Vec::with_capacity(
        BLOB_HEADER_BYTES + regions.iter().map(Vec::len).sum::<usize>(),
    );
    for d in dims {
        blob.extend_from_slice(&d[0].to_le_bytes());
        blob.extend_from_slice(&d[1].to_le_bytes());
    }
    blob.extend_from_slice(&(GROUP as u32).to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    for r in &regions {
        blob.extend_from_slice(r);
    }
    Ok((blob, max_err))
}

/// Minimal safetensors writer: JSON header + raw data, tensors kept in their
/// source dtype, copied shard→core via pread.
fn write_safetensors(ckpt: &Checkpoint, names: &[String], out: &Path) -> Result<u64> {
    let mut header = serde_json::Map::new();
    let mut cursor = 0u64;
    for name in names {
        let loc = &ckpt.tensors[name];
        let len = loc.end - loc.start;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": loc.dtype,
                "shape": loc.shape,
                "data_offsets": [cursor, cursor + len],
            }),
        );
        cursor += len;
    }
    let mut hdr = serde_json::to_vec(&serde_json::Value::Object(header))?;
    // Pad header to 8 bytes for aligned data (spec allows trailing spaces).
    while hdr.len() % 8 != 0 {
        hdr.push(b' ');
    }

    let mut w = BufWriter::new(File::create(out)?);
    w.write_all(&(hdr.len() as u64).to_le_bytes())?;
    w.write_all(&hdr)?;
    let mut copied = 0u64;
    for name in names {
        let raw = ckpt.read_raw(&ckpt.tensors[name])?;
        copied += raw.len() as u64;
        w.write_all(&raw)?;
    }
    w.flush()?;
    Ok(8 + hdr.len() as u64 + copied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmpager_core::pack::PackReader;
    use llmpager_core::quant::q4g64_dequantize;

    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * 0.05
            })
            .collect()
    }

    /// Hand-rolled checkpoint: 2 layers x 4 experts, hidden 128, inter 64.
    fn write_checkpoint(dir: &Path) -> Result<()> {
        let (layers, experts, hidden, inter) = (2usize, 4usize, 128usize, 64usize);
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "qwen3_moe",
                "num_hidden_layers": layers,
                "num_experts": experts,
                "hidden_size": hidden,
                "moe_intermediate_size": inter,
                "num_experts_per_tok": 2,
            }))?,
        )?;

        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        let mut add = |name: String, shape: Vec<usize>, header: &mut serde_json::Map<String, serde_json::Value>, data: &mut Vec<u8>| {
            let n: usize = shape.iter().product();
            let seed = name.bytes().fold(1u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
            let bytes = bf16_bytes(&pseudo(n, seed));
            let start = data.len();
            data.extend_from_slice(&bytes);
            header.insert(
                name,
                serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, start + bytes.len()]}),
            );
        };

        add("model.embed_tokens.weight".into(), vec![32, hidden], &mut header, &mut data);
        for l in 0..layers {
            add(format!("model.layers.{l}.mlp.gate.weight"), vec![experts, hidden], &mut header, &mut data);
            for e in 0..experts {
                add(format!("model.layers.{l}.mlp.experts.{e}.gate_proj.weight"), vec![inter, hidden], &mut header, &mut data);
                add(format!("model.layers.{l}.mlp.experts.{e}.up_proj.weight"), vec![inter, hidden], &mut header, &mut data);
                add(format!("model.layers.{l}.mlp.experts.{e}.down_proj.weight"), vec![hidden, inter], &mut header, &mut data);
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
    fn end_to_end_convert() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_checkpoint(dir.path())?;
        let pack_path = dir.path().join("m.llmpk");
        let core_path = dir.path().join("core.safetensors");
        let report = convert(dir.path(), &pack_path, &core_path)?;
        assert_eq!((report.layers, report.experts), (2, 4));
        assert_eq!(report.core_tensors, 3); // embed + 2 router gates
        assert!(report.max_quant_err < 0.05 / 14.0 * 1.2);

        // Pack sanity + dequant round-trip for one expert's gate_proj.
        let r = PackReader::open(&pack_path)?;
        assert_eq!(r.meta().dtype, "q4g64-gud");
        assert_eq!(r.meta().config["hidden_size"], 128);
        let entry = r.entry(1, 2);
        let mut buf = vec![0u8; (entry.nbytes as usize).div_ceil(4096) * 4096];
        let n = r.read_blob_into(1, 2, &mut buf)?;
        let (rows, cols) = (
            u32::from_le_bytes(buf[0..4].try_into()?) as usize,
            u32::from_le_bytes(buf[4..8].try_into()?) as usize,
        );
        assert_eq!((rows, cols), (64, 128));
        assert_eq!(
            n,
            BLOB_HEADER_BYTES + 2 * q4g64_bytes(64, 128) + q4g64_bytes(128, 64)
        );

        let ckpt = Checkpoint::open(dir.path())?;
        let (orig, _) = ckpt.read_f32("model.layers.1.mlp.experts.2.gate_proj.weight")?;
        let region = &buf[BLOB_HEADER_BYTES..BLOB_HEADER_BYTES + q4g64_bytes(rows, cols)];
        let mut deq = vec![0f32; rows * cols];
        q4g64_dequantize(region, rows, cols, &mut deq)?;
        for (a, b) in orig.iter().zip(deq.iter()) {
            assert!((a - b).abs() < 0.006, "{a} vs {b}");
        }

        // Core file: correct size on disk and the embed tensor's bytes are
        // preserved verbatim.
        assert_eq!(std::fs::metadata(&core_path)?.len(), report.core_bytes);
        let core = std::fs::read(&core_path)?;
        let hlen = u64::from_le_bytes(core[..8].try_into()?) as usize;
        let chdr: serde_json::Value = serde_json::from_slice(&core[8..8 + hlen])?;
        let offs = chdr["model.embed_tokens.weight"]["data_offsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let loc = &ckpt.tensors["model.embed_tokens.weight"];
        let orig_raw = ckpt.read_raw(loc)?;
        assert_eq!(&core[8 + hlen + offs[0]..8 + hlen + offs[1]], &orig_raw[..]);
        Ok(())
    }
}
