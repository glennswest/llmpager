use std::path::PathBuf;

use anyhow::{bail, Result};

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .find_map(|a| a.strip_prefix(&format!("--{key}=")).map(String::from))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(dir) = arg(&args, "gen-test") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        llmpager_convert::write_test_checkpoint(&dir)?;
        println!("synthetic checkpoint written to {}", dir.display());
        return Ok(());
    }
    if let Some(f) = arg(&args, "gguf-info") {
        return llmpager_convert::gguf::info(&PathBuf::from(f));
    }
    if let Some(dir) = arg(&args, "gen-test-kimi") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        llmpager_convert::kimi::write_test_checkpoint_kimi(&dir)?;
        println!("synthetic kimi checkpoint written to {}", dir.display());
        return Ok(());
    }
    let (Some(model_dir), Some(out_pack), Some(out_core)) = (
        arg(&args, "model-dir"),
        arg(&args, "out-pack"),
        arg(&args, "out-core"),
    ) else {
        bail!("usage: llmpager-convert --model-dir=DIR --out-pack=FILE.llmpk --out-core=FILE.safetensors  (or --gen-test=DIR)");
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
