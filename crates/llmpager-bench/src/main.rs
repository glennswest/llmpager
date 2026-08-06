//! M0 microbenchmarks for llmpager.
//!
//! Subcommands:
//!   gen    — write a synthetic expert pack
//!   disk   — random-read throughput from the pack (optionally O_DIRECT)
//!   paged  — end-to-end paged fetch: cache lookup → pread → pinned → VRAM
//!            (requires --features cuda and an NVIDIA GPU)

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llmpager_core::pack::{AlignedBuf, PackMeta, PackReader, PackWriter, ALIGN};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod paged;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        bail!("usage: llmpager-bench <gen|disk|paged> [--key=value ...]");
    };
    let flags = Flags::parse(&args[1..]);
    match cmd.as_str() {
        "gen" => gen(&flags),
        "disk" => disk(&flags),
        #[cfg(feature = "cuda")]
        "paged" => paged::run(&flags),
        #[cfg(not(feature = "cuda"))]
        "paged" => bail!("rebuild with --features cuda for the paged benchmark"),
        other => bail!("unknown subcommand {other}"),
    }
}

pub struct Flags(Vec<(String, String)>);

impl Flags {
    fn parse(args: &[String]) -> Self {
        Self(
            args.iter()
                .filter_map(|a| {
                    let a = a.strip_prefix("--")?;
                    let (k, v) = a.split_once('=')?;
                    Some((k.to_string(), v.to_string()))
                })
                .collect(),
        )
    }

    pub fn str(&self, key: &str, default: &str) -> String {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| default.to_string())
    }

    pub fn num(&self, key: &str, default: u64) -> u64 {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.parse().ok())
            .unwrap_or(default)
    }

    pub fn frac(&self, key: &str, default: f64) -> f64 {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.parse().ok())
            .unwrap_or(default)
    }

    pub fn path(&self, key: &str) -> Result<PathBuf> {
        let v = self.str(key, "");
        if v.is_empty() {
            bail!("--{key}=<path> is required");
        }
        Ok(PathBuf::from(v))
    }
}

/// Deterministic xorshift so runs are repeatable without a rand dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn gen(f: &Flags) -> Result<()> {
    let path = f.path("path")?;
    let layers = f.num("layers", 24) as u16;
    let experts = f.num("experts", 64) as u16;
    let bytes = f.num("bytes", 3_000_000) as usize;
    let total_gb = layers as f64 * experts as f64 * bytes as f64 / 1e9;
    println!("gen: {layers} layers x {experts} experts x {bytes} B  (~{total_gb:.1} GB)");

    let meta = PackMeta {
        model: "synthetic".into(),
        num_layers: layers,
        experts_per_layer: experts,
        dtype: "raw".into(),
    };
    let mut w = PackWriter::create(&path, meta).context("creating pack")?;
    // Patterned, compressible-hostile filler; content is irrelevant to I/O.
    let mut rng = Rng::new(42);
    let mut blob = vec![0u8; bytes];
    let start = Instant::now();
    for l in 0..layers {
        for e in 0..experts {
            let tag = (l as u64) << 32 | e as u64;
            for chunk in blob.chunks_mut(8) {
                let v = (rng.next() ^ tag).to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
            w.add_blob(&blob)?;
        }
    }
    w.finish()?;
    let secs = start.elapsed().as_secs_f64();
    println!("gen: wrote {total_gb:.1} GB in {secs:.1}s ({:.2} GB/s)", total_gb / secs);
    Ok(())
}

fn disk(f: &Flags) -> Result<()> {
    let path = f.path("path")?;
    let threads = f.num("threads", 8) as usize;
    let reads = f.num("reads", 4000);
    let direct = f.str("direct", "true") == "true";

    let open = |direct: bool| -> Result<PackReader> {
        #[cfg(target_os = "linux")]
        if direct {
            return PackReader::open_direct(&path);
        }
        let _ = direct;
        PackReader::open(&path)
    };
    let reader = open(direct)?;
    let meta = reader.meta().clone();
    let span = reader.max_blob_bytes().div_ceil(ALIGN) * ALIGN;
    println!(
        "disk: {} layers x {} experts, blob span {span} B, {threads} threads, {reads} reads, direct={direct}",
        meta.num_layers, meta.experts_per_layer
    );

    let done = AtomicU64::new(0);
    let bytes = AtomicU64::new(0);
    let start = Instant::now();
    std::thread::scope(|s| -> Result<()> {
        let mut handles = Vec::new();
        for t in 0..threads {
            let reader = open(direct)?;
            let done = &done;
            let bytes = &bytes;
            let meta = &meta;
            handles.push(s.spawn(move || -> Result<()> {
                let mut rng = Rng::new(0x9e3779b97f4a7c15 ^ t as u64);
                let mut buf = AlignedBuf::new(span as usize);
                loop {
                    if done.fetch_add(1, Ordering::Relaxed) >= reads {
                        return Ok(());
                    }
                    let l = rng.below(meta.num_layers as u64) as u16;
                    let e = rng.below(meta.experts_per_layer as u64) as u16;
                    let n = reader.read_blob_into(l, e, buf.as_mut())?;
                    bytes.fetch_add(n as u64, Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join().unwrap()?;
        }
        Ok(())
    })?;
    let secs = start.elapsed().as_secs_f64();
    let gb = bytes.load(Ordering::Relaxed) as f64 / 1e9;
    println!(
        "disk: {gb:.2} GB in {secs:.2}s = {:.2} GB/s ({:.0} blobs/s, {:.2} ms/blob avg across {threads} threads)",
        gb / secs,
        reads as f64 / secs,
        secs * 1000.0 * threads as f64 / reads as f64,
    );
    Ok(())
}
