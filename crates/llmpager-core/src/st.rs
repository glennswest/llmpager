//! Minimal safetensors reader: u64 LE header length, JSON header mapping
//! tensor name -> {dtype, shape, data_offsets}, then raw data. Tensors are
//! pread on demand — the file is never loaded whole.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte range within the file.
    pub start: u64,
    pub end: u64,
}

pub struct SafeTensors {
    file: File,
    tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut len8 = [0u8; 8];
        file.read_exact_at(&mut len8, 0)?;
        let header_len = u64::from_le_bytes(len8);
        let mut hdr = vec![0u8; header_len as usize];
        file.read_exact_at(&mut hdr, 8)?;
        let header: serde_json::Value = serde_json::from_slice(&hdr)?;
        let data_base = 8 + header_len;
        let obj = header.as_object().context("bad safetensors header")?;
        let mut tensors = BTreeMap::new();
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let offs = info["data_offsets"]
                .as_array()
                .with_context(|| format!("{name}: missing data_offsets"))?;
            tensors.insert(
                name.clone(),
                TensorInfo {
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
        Ok(Self { file, tensors })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not found"))
    }

    /// Raw bytes, source dtype.
    pub fn raw(&self, name: &str) -> Result<(Vec<u8>, TensorInfo)> {
        let info = self.info(name)?.clone();
        let mut buf = vec![0u8; (info.end - info.start) as usize];
        self.file.read_exact_at(&mut buf, info.start)?;
        Ok((buf, info))
    }

    /// Tensor converted to f32 (from BF16 or F32).
    pub fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (raw, info) = self.raw(name)?;
        let vals = match info.dtype.as_str() {
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
        Ok((vals, info.shape))
    }
}
