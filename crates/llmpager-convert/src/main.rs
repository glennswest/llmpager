use std::path::PathBuf;

use anyhow::{bail, Result};

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .find_map(|a| a.strip_prefix(&format!("--{key}=")).map(String::from))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(model_dir), Some(out_pack), Some(out_core)) = (
        arg(&args, "model-dir"),
        arg(&args, "out-pack"),
        arg(&args, "out-core"),
    ) else {
        bail!("usage: llmpager-convert --model-dir=DIR --out-pack=FILE.llmpk --out-core=FILE.safetensors");
    };

    let t0 = std::time::Instant::now();
    let report = llmpager_convert::convert(
        &PathBuf::from(model_dir),
        &PathBuf::from(&out_pack),
        &PathBuf::from(&out_core),
    )?;
    println!(
        "converted {} layers x {} experts in {:.0}s",
        report.layers,
        report.experts,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "pack: {out_pack} ({:.2} GB), core: {out_core} ({:.2} GB, {} tensors), max quant err {:.5}",
        report.pack_bytes as f64 / 1e9,
        report.core_bytes as f64 / 1e9,
        report.core_tensors,
        report.max_quant_err
    );
    Ok(())
}
