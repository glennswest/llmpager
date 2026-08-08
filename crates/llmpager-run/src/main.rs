use llmpager_run::{decode, kimi};

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};

/// Architecture dispatch: same decode surface over both engines.
enum AnyDecoder {
    Qwen(decode::Decoder),
    Kimi(kimi::KimiDecoder),
}

impl AnyDecoder {
    fn step(&mut self, token: u32, pos: usize, want_logits: bool) -> Result<u32> {
        match self {
            AnyDecoder::Qwen(d) => d.step(token, pos, want_logits),
            AnyDecoder::Kimi(d) => d.step(token, pos, want_logits),
        }
    }
    fn last_logits(&self) -> Vec<f32> {
        match self {
            AnyDecoder::Qwen(d) => d.last_logits(),
            AnyDecoder::Kimi(d) => d.last_logits(),
        }
    }
    fn pager_metrics(&self) -> llmpager_cuda::pager::Metrics {
        match self {
            AnyDecoder::Qwen(d) => d.pager_metrics(),
            AnyDecoder::Kimi(d) => d.pager_metrics(),
        }
    }
    fn eos(&self) -> &[u32] {
        match self {
            AnyDecoder::Qwen(d) => &d.cfg.eos,
            AnyDecoder::Kimi(d) => &d.cfg.eos,
        }
    }
}

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
    // --direct=0: let the OS page cache act as a RAM tier for the pack —
    // right when the pack fits in host RAM; keep O_DIRECT for huge packs.
    let direct = arg(&args, "direct").as_deref() != Some("0");

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
    } else if arg(&args, "ppl").is_some() {
        vec![1] // unused in --ppl mode
    } else {
        bail!("need --prompt, --prompt-ids, or --ppl");
    };
    if prompt_ids.is_empty() {
        bail!("empty prompt");
    }

    let pack_meta =
        llmpager_core::pack::PackReader::open(&PathBuf::from(&pack))?.meta().clone();
    let mut dec = if kimi::KimiConfig::is_kimi(&pack_meta.config) {
        let mut d = kimi::KimiDecoder::new(
            &PathBuf::from(&pack),
            &PathBuf::from(&core),
            slots,
            io_threads,
            max_seq,
            direct,
        )?;
        // Fetch-traffic knob: drop routed experts below this scaled weight.
        d.min_expert_weight = arg(&args, "min-expert-weight")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        AnyDecoder::Kimi(d)
    } else {
        let mut d = decode::Decoder::new(
            &PathBuf::from(&pack),
            &PathBuf::from(&core),
            slots,
            io_threads,
            max_seq,
            core_q4,
            direct,
        )?;
        // Default off: measured 18.5 -> 10.8 tok/s. Qwen3 expert routing has
        // ~zero cross-layer correlation; wrong prefetches evict good entries
        // and triple disk traffic. Kept for experiments.
        d.prefetch_next = arg(&args, "prefetch-next").as_deref() == Some("1");
        AnyDecoder::Qwen(d)
    };

    // Perplexity mode: teacher-forced NLL over a text file. Validates the
    // whole pipeline numerically — and PPL must be identical across cache
    // sizes if paging is lossless.
    if let Some(ppl_file) = arg(&args, "ppl") {
        let tok = tokenizer.as_ref().context("--ppl requires --tokenizer")?;
        let text = std::fs::read_to_string(&ppl_file).context("reading --ppl file")?;
        let ids = tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        let n = ids.len().min(max_seq);
        if n < 2 {
            bail!("--ppl text too short");
        }
        eprintln!("ppl: {n} tokens");
        let t0 = Instant::now();
        let mut nll = 0f64;
        for pos in 0..n - 1 {
            dec.step(ids[pos], pos, true)?;
            let logits = dec.last_logits();
            let target = ids[pos + 1] as usize;
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let z: f64 = logits.iter().map(|v| ((v - m) as f64).exp()).sum();
            nll -= (logits[target] - m) as f64 - z.ln();
        }
        let count = (n - 1) as f64;
        println!(
            "ppl: {:.4} over {} tokens ({:.3} bits/token, {:.2} tok/s)",
            (nll / count).exp(),
            n - 1,
            nll / count / std::f64::consts::LN_2,
            count / t0.elapsed().as_secs_f64(),
        );
        return Ok(());
    }

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
        if dec.eos().contains(&next) {
            break;
        }
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
