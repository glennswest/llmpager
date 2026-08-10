//! Minimal GGUF v3 reader — enough to inventory and repack Unsloth
//! Dynamic-quant models into `.llmpk` packs (M9).
//!
//! Format: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count,
//! then KV pairs, then tensor infos (name, dims, ggml type, offset),
//! then aligned tensor data. Multi-file models repeat the header per
//! shard with `split.*` KVs.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

/// GGML tensor types we care about (llama.cpp ggml.h numbering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    /// (values per block, bytes per block); None for unknown types.
    pub fn block_layout(self) -> Option<(usize, usize)> {
        Some(match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Other(_) => return None,
        })
    }

    pub fn row_bytes(self, cols: u64) -> Option<u64> {
        let (vals, bytes) = self.block_layout()?;
        if cols as usize % vals != 0 {
            return None;
        }
        Some(cols / vals as u64 * bytes as u64)
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub file: usize,
    /// Dims in ggml order (fastest-varying first): [cols, rows, experts?].
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset within its file.
    pub abs_offset: u64,
}

pub struct Gguf {
    pub files: Vec<File>,
    pub paths: Vec<PathBuf>,
    pub tensors: BTreeMap<String, GgufTensor>,
    /// Selected metadata (string/int keys only, stringified).
    pub meta: BTreeMap<String, String>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            bail!("gguf header truncated");
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn skip_value(&mut self, ty: u32) -> Result<Option<String>> {
        Ok(match ty {
            0 | 1 => {
                self.take(1)?;
                None
            }
            2 | 3 => {
                self.take(2)?;
                None
            }
            4 | 5 => Some(self.u32()?.to_string()),
            6 => {
                self.take(4)?;
                None
            }
            7 => Some((self.take(1)?[0] != 0).to_string()),
            8 => Some(self.string()?),
            9 => {
                let ety = self.u32()?;
                let count = self.u64()?;
                for _ in 0..count {
                    self.skip_value(ety)?;
                }
                None
            }
            10 | 11 => Some(self.u64()?.to_string()),
            12 => {
                self.take(8)?;
                None
            }
            other => bail!("gguf: unknown kv type {other}"),
        })
    }
}

impl Gguf {
    /// Open a GGUF model from any one shard path; sibling shards
    /// (`-00002-of-000NN.gguf`) are discovered automatically.
    pub fn open(first: &Path) -> Result<Self> {
        let mut paths = vec![first.to_path_buf()];
        let name = first.file_name().unwrap().to_string_lossy().to_string();
        if let Some(idx) = name.find("-00001-of-") {
            let total: usize = name[idx + 10..idx + 15].parse().unwrap_or(1);
            for i in 2..=total {
                let sib = name.replace("-00001-of-", &format!("-{i:05}-of-"));
                paths.push(first.with_file_name(sib));
            }
        }

        let mut files = Vec::new();
        let mut tensors = BTreeMap::new();
        let mut meta = BTreeMap::new();
        for (fidx, path) in paths.iter().enumerate() {
            let mut file =
                File::open(path).with_context(|| format!("opening {}", path.display()))?;
            // Headers are small relative to shards; 32MB covers the largest.
            let mut head = vec![0u8; 32 << 20];
            let got = file.read(&mut head)?;
            head.truncate(got);
            let mut c = Cursor { buf: &head, pos: 0 };
            if c.u32()? != GGUF_MAGIC {
                bail!("{}: not a GGUF file", path.display());
            }
            let version = c.u32()?;
            if !(2..=3).contains(&version) {
                bail!("{}: unsupported GGUF version {version}", path.display());
            }
            let tensor_count = c.u64()?;
            let kv_count = c.u64()?;
            let mut alignment = 32u64;
            for _ in 0..kv_count {
                let key = c.string()?;
                let ty = c.u32()?;
                let val = c.skip_value(ty)?;
                if let Some(v) = val {
                    if key == "general.alignment" {
                        alignment = v.parse().unwrap_or(32);
                    }
                    if fidx == 0 && (key.starts_with("general.") || key.contains("expert")) {
                        meta.insert(key, v);
                    }
                }
            }
            let mut infos = Vec::with_capacity(tensor_count as usize);
            for _ in 0..tensor_count {
                let name = c.string()?;
                let n_dims = c.u32()? as usize;
                let mut dims = Vec::with_capacity(n_dims);
                for _ in 0..n_dims {
                    dims.push(c.u64()?);
                }
                let ty = GgmlType::from_u32(c.u32()?);
                let off = c.u64()?;
                infos.push((name, dims, ty, off));
            }
            let data_start = (c.pos as u64).div_ceil(alignment) * alignment;
            for (name, dims, ty, off) in infos {
                tensors.insert(
                    name,
                    GgufTensor { file: fidx, dims, ty, abs_offset: data_start + off },
                );
            }
            files.push(file);
        }
        Ok(Self { files, paths, tensors, meta })
    }

    /// Read `len` bytes of a tensor starting `rel` bytes into its data.
    pub fn read_at(&self, t: &GgufTensor, rel: u64, buf: &mut [u8]) -> Result<()> {
        self.files[t.file].read_exact_at(buf, t.abs_offset + rel)?;
        Ok(())
    }
}

/// Print an inventory (name, type, dims) — spike/debug tool.
pub fn info(path: &Path) -> Result<()> {
    let g = Gguf::open(path)?;
    println!("shards: {}", g.paths.len());
    for (k, v) in &g.meta {
        println!("meta {k} = {v}");
    }
    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    for (name, t) in &g.tensors {
        *by_type.entry(format!("{:?}", t.ty)).or_default() += 1;
        if name.contains("exps") && name.contains("blk.4.") || name.starts_with("blk.4.attn") {
            println!("{name}: {:?} dims={:?}", t.ty, t.dims);
        }
    }
    println!("tensor counts by type: {by_type:?}");
    println!("total tensors: {}", g.tensors.len());
    Ok(())
}
