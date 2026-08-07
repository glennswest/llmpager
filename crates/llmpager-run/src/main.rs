mod decode;
mod model;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().find_map(|a| a.strip_prefix(&format!("--{key}=")).map(String::from))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(pack), Some(core)) = (arg(&args, "pack"), arg(&args, "core")) else {
        bail!(
            "usage: llmpager-run --pack=F.llmpk --core=F.core.safetensors \
             [--tokenizer=DIR|tokenizer.json] [--prompt=TEXT | --prompt-ids=1,2,3] \
             [--max-tokens=64] [--slots=32] [--io-threads=4] [--max-seq=4096]"
        );
    };
    let max_tokens: usize = arg(&args, "max-tokens").and_then(|v| v.parse().ok()).unwrap_or(64);
    let slots: u32 = arg(&args, "slots").and_then(|v| v.parse().ok()).unwrap_or(32);
    let io_threads: usize = arg(&args, "io-threads").and_then(|v| v.parse().ok()).unwrap_or(4);
    let max_seq: usize = arg(&args, "max-seq").and_then(|v| v.parse().ok()).unwrap_or(4096);
    // Core GEMV encoding. Default bf16: measured faster than q4 (the q4
    // kernel's unpack cost eats the bandwidth win) and greedy output drifts
    // under double quantization. --core-dtype=q4 kept for experiments.
    let core_q4 = arg(&args, "core-dtype").as_deref() == Some("q4");

    // Tokenizer is optional: --prompt-ids allows raw-id smoke tests.
    let tokenizer = match arg(&args, "tokenizer") {
        Some(t) => {
            let p = PathBuf::from(&t);
            let file = if p.is_dir() { p.join("tokenizer.json") } else { p };
            Some(
                tokenizers::Tokenizer::from_file(&file)
                    .map_err(|e| anyhow::anyhow!("loading {}: {e}", file.display()))?,
            )
        }
        None => None,
    };

    let prompt_ids: Vec<u32> = if let Some(ids) = arg(&args, "prompt-ids") {
        ids.split(',').map(|s| s.trim().parse().context("bad --prompt-ids")).collect::<Result<_>>()?
    } else if let Some(text) = arg(&args, "prompt") {
        let tok = tokenizer.as_ref().context("--prompt requires --tokenizer")?;
        tok.encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec()
    } else {
        bail!("need --prompt or --prompt-ids");
    };
    if prompt_ids.is_empty() {
        bail!("empty prompt");
    }

    let mut dec = decode::Decoder::new(
        &PathBuf::from(&pack),
        &PathBuf::from(&core),
        slots,
        io_threads,
        max_seq,
        core_q4,
    )?;

    // Prefill: feed prompt tokens; logits of the last one seed generation.
    eprintln!("prefill: {} tokens", prompt_ids.len());
    let t0 = Instant::now();
    let mut next = 0u32;
    for (pos, id) in prompt_ids.iter().enumerate() {
        let last = pos + 1 == prompt_ids.len();
        next = dec.step(*id, pos, last)?;
    }
    let prefill_s = t0.elapsed().as_secs_f64();

    let mut generated: Vec<u32> = Vec::new();
    let mut printed = String::new();
    let t1 = Instant::now();
    for i in 0..max_tokens {
        generated.push(next);
        if let Some(tok) = tokenizer.as_ref() {
            let full = tok
                .decode(&generated, true)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            print!("{}", &full[printed.len()..]);
            use std::io::Write;
            std::io::stdout().flush().ok();
            printed = full;
        }
        let pos = prompt_ids.len() + i;
        if pos >= max_seq {
            break;
        }
        next = dec.step(next, pos, true)?;
    }
    let gen_s = t1.elapsed().as_secs_f64();
    if tokenizer.is_none() {
        println!("ids: {generated:?}");
    } else {
        println!();
    }

    let m = dec.pager_metrics();
    eprintln!(
        "prefill {:.2}s ({:.2} tok/s) | decode {:.2}s ({:.2} tok/s) | \
         expert cache {:.1}% hit, {:.2} GB streamed",
        prefill_s,
        prompt_ids.len() as f64 / prefill_s,
        gen_s,
        generated.len() as f64 / gen_s,
        100.0 * m.hit_rate(),
        m.bytes_fetched as f64 / 1e9,
    );
    Ok(())
}
