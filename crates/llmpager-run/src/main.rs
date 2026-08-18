use llmpager_run::any::AnyDecoder;

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
    // VRAM to leave free for other processes on the card. The expert cache
    // is sized down to fit around it.
    let reserve_mb: u64 = arg(&args, "reserve-mb").and_then(|v| v.parse().ok()).unwrap_or(0);
    // Core GEMV encoding. Default bf16: measured faster than q4 (the q4
    // kernel's unpack cost eats the bandwidth win) and greedy output drifts
    // under double quantization. --core-dtype=q4 kept for experiments.
    let core_q4 = arg(&args, "core-dtype").as_deref() == Some("q4");
    // --direct=0: let the OS page cache act as a RAM tier for the pack —
    // right when the pack fits in host RAM; keep O_DIRECT for huge packs.
    let direct = arg(&args, "direct").as_deref() != Some("0");
    // Managed host-RAM expert tier (GB); the lever for packs >> RAM.
    let ram_gb: f64 = arg(&args, "ram-gb").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let batch: usize = arg(&args, "batch").and_then(|v| v.parse().ok()).unwrap_or(1);

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

    let mut dec = AnyDecoder::new(
        &PathBuf::from(&pack),
        &PathBuf::from(&core),
        slots,
        io_threads,
        max_seq,
        core_q4,
        direct,
        (ram_gb * 1e9) as u64,
        reserve_mb * 1_000_000,
        batch,
    )?;
    // Profiled pre-warm: load a fetch-count profile into the RAM tier
    // before the first token; save one at exit with --profile-out.
    if let Some(f) = arg(&args, "prewarm") {
        let counts: Vec<u64> =
            serde_json::from_slice(&std::fs::read(&f).context("reading --prewarm")?)?;
        let t = Instant::now();
        let n = dec.prewarm(&counts)?;
        eprintln!("prewarm: {n} experts into the RAM tier in {:.1}s", t.elapsed().as_secs_f64());
    }
    let profile_out = arg(&args, "profile-out");

    // Qwen cross-layer prefetch: default off (measured 18.5 -> 10.8 tok/s).
    dec.set_prefetch_next(arg(&args, "prefetch-next").as_deref() == Some("1"));
    // Kimi fetch-traffic knob: drop routed experts below this scaled weight.
    dec.set_min_expert_weight(
        arg(&args, "min-expert-weight").and_then(|v| v.parse().ok()).unwrap_or(0.0),
    );

    // Lockstep batch self-test: decode the same prompt as N independent
    // sequence slots; outputs must be bit-identical across slots (proves
    // KV isolation) and reports aggregate throughput.
    if let Some(nstr) = arg(&args, "batch-selftest").and_then(|v| v.parse::<usize>().ok()) {
        let t0 = Instant::now();
        let mut next = vec![0u32; nstr];
        for s in 0..nstr {
            let mut pos = 0usize;
            for chunk in prompt_ids.chunks(dec.chunk_cap()) {
                let entries: Vec<(u32, usize, usize)> =
                    chunk.iter().enumerate().map(|(i, t)| (*t, pos + i, s)).collect();
                let last = pos + chunk.len() == prompt_ids.len();
                let out = dec.step_multi(&entries, last)?;
                if last {
                    next[s] = *out.last().unwrap();
                }
                pos += chunk.len();
            }
        }
        eprintln!("prefill x{nstr} done in {:.1}s", t0.elapsed().as_secs_f64());
        let t1 = Instant::now();
        let mut total = 0usize;
        let mut pos = prompt_ids.len();
        for _ in 0..max_tokens {
            if next.iter().any(|t| dec.eos().contains(t)) || pos >= max_seq {
                break;
            }
            let entries: Vec<(u32, usize, usize)> =
                next.iter().map(|t| (*t, pos, 0)).enumerate()
                    .map(|(s, (t, p, _))| (t, p, s)).collect();
            let out = dec.step_multi(&entries, true)?;
            if out.windows(2).any(|w| w[0] != w[1]) {
                bail!("STREAM DIVERGENCE at pos {pos}: {out:?}");
            }
            next = out;
            total += nstr;
            pos += 1;
        }
        let secs = t1.elapsed().as_secs_f64();
        let m = dec.pager_metrics();
        println!(
            "batch-selftest PASS: {nstr} identical streams, {total} tokens in {secs:.2}s \
             = {:.2} tok/s aggregate ({:.2} per stream); cache {:.1}% hit",
            total as f64 / secs,
            total as f64 / secs / nstr as f64,
            100.0 * m.hit_rate(),
        );
        return Ok(());
    }

    // Session self-test. Chunked prefill is not bit-reproducible across
    // chunk groupings -- each layer fetches its chunk's expert *union* in
    // waves, so regrouping the tokens regroups the partial sums -- and the
    // engine has always behaved that way (--chunk=N changes outputs today).
    // So the test measures that spread first and judges prefix reuse
    // against it, while holding the KV copy itself to exact equality: the
    // export/import path runs identical call boundaries, so any difference
    // there is a copy bug, not arithmetic.
    if args.iter().any(|a| a == "--session-selftest") {
        let n = prompt_ids.len();
        if n < 8 {
            bail!("--session-selftest needs a prompt of at least 8 tokens");
        }
        if dec.batch_cap() < 2 {
            bail!("--session-selftest needs --batch=2 or more (slot 1 holds the session)");
        }
        let split = n / 2;

        /// Prefill `ids[from..]` on `slot` in chunks of `cap`; returns the
        /// next-token logits.
        fn prefill_logits(
            dec: &mut AnyDecoder,
            slot: usize,
            ids: &[u32],
            from: usize,
            cap: usize,
        ) -> Result<Vec<f32>> {
            let cap = cap.min(dec.chunk_cap()).max(1);
            let mut pos = from;
            for chunk in ids[from..].chunks(cap) {
                let entries: Vec<(u32, usize, usize)> =
                    chunk.iter().enumerate().map(|(i, t)| (*t, pos + i, slot)).collect();
                let last = pos + chunk.len() == ids.len();
                dec.step_multi(&entries, last)?;
                pos += chunk.len();
            }
            Ok(dec
                .last_logits_multi_or_single()
                .last()
                .cloned()
                .filter(|l| !l.is_empty())
                .unwrap_or_else(|| dec.last_logits()))
        }

        /// Greedy continuation on `slot`, starting from `first`.
        fn decode_from(
            dec: &mut AnyDecoder,
            slot: usize,
            first: u32,
            prompt_len: usize,
            max_tokens: usize,
            max_seq: usize,
        ) -> Result<Vec<u32>> {
            let mut next = first;
            let mut got = Vec::new();
            for i in 0..max_tokens {
                if dec.eos().contains(&next) {
                    break;
                }
                got.push(next);
                let p = prompt_len + i;
                if p + 1 >= max_seq {
                    break;
                }
                next = dec.step_multi(&[(next, p, slot)], true)?[0];
            }
            Ok(got)
        }

        let delta = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
        };
        let am = |v: &[f32]| -> u32 {
            v.iter()
                .enumerate()
                .fold((0usize, f32::MIN), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
                .0 as u32
        };

        // 1. The engine's own spread: one context, three chunk groupings.
        let base = prefill_logits(&mut dec, 0, &prompt_ids, 0, usize::MAX)?;
        let g32 = prefill_logits(&mut dec, 1, &prompt_ids, 0, 32)?;
        let g17 = prefill_logits(&mut dec, 1, &prompt_ids, 0, 17)?;
        let spread = delta(&base, &g32).max(delta(&base, &g17));
        let scale = base.iter().fold(0f32, |m, v| m.max(v.abs()));

        // 2. Prefix reuse: resume at `split` instead of prefilling it again.
        prefill_logits(&mut dec, 1, &prompt_ids[..split], 0, usize::MAX)?;
        let reuse = prefill_logits(&mut dec, 1, &prompt_ids, split, usize::MAX)?;
        let toks_reuse =
            decode_from(&mut dec, 1, am(&reuse), prompt_ids.len(), max_tokens, max_seq)?;

        // 3. The same resume, with the prefix parked in host RAM and the
        //    slot deliberately clobbered in between.
        prefill_logits(&mut dec, 1, &prompt_ids[..split], 0, usize::MAX)?;
        let blob = dec.kv_export(1, split)?;
        prefill_logits(&mut dec, 1, &prompt_ids[split..], 0, usize::MAX)?;
        dec.kv_import(1, split, &blob)?;
        let restored = prefill_logits(&mut dec, 1, &prompt_ids, split, usize::MAX)?;
        let toks_restored =
            decode_from(&mut dec, 1, am(&restored), prompt_ids.len(), max_tokens, max_seq)?;

        let d_reuse = delta(&base, &reuse);
        let d_kv = delta(&reuse, &restored);
        eprintln!(
            "prompt {n} tokens, split {split}, chunk_cap {}, logit scale {scale:.2}",
            dec.chunk_cap()
        );
        eprintln!(
            "  chunk-grouping spread: {spread:.5}   prefix reuse: {d_reuse:.5}   kv copy: {d_kv:.5}"
        );

        if am(&base) != am(&reuse) || am(&base) != am(&restored) {
            bail!("session-selftest FAIL: prefix reuse changed the next token");
        }
        // Reuse must stay inside the engine's own nondeterminism, with room
        // for it to land on the unlucky side of it.
        if d_reuse > 2.0 * spread + 1e-3 {
            bail!(
                "session-selftest FAIL (prefix reuse): logits moved {d_reuse:.5}, more than \
                 twice the {spread:.5} the engine already spans across chunk groupings"
            );
        }
        // The KV copy changes no boundaries, so it must change no bits.
        if d_kv != 0.0 {
            bail!("session-selftest FAIL (kv export/import): not lossless, logits moved {d_kv:.5}");
        }
        if toks_reuse != toks_restored {
            bail!(
                "session-selftest FAIL (kv export/import): tokens differ\n  direct:   \
                 {toks_reuse:?}\n  restored: {toks_restored:?}"
            );
        }
        eprintln!(
            "session-selftest PASS: kv export/import bit-exact ({:.1} MB for {split} tokens, \
             {} bytes/token); prefix reuse within the engine's chunk-grouping spread",
            blob.len() as f64 / 1e6,
            dec.kv_bytes_per_token(),
        );
        return Ok(());
    }


    // Serial self-test: N complete generations (prefill + decode) back to
    // back on one decoder — the shape a server sees, which a single CLI run
    // never exercises. Regression guard for the deferred-release ring
    // leaking pins across generations: the second prefill then stalled
    // forever on slots the same thread held. Greedy, so all runs must match.
    if let Some(runs) = arg(&args, "serial-selftest").and_then(|v| v.parse::<usize>().ok()) {
        let mut first: Option<Vec<u32>> = None;
        for r in 0..runs {
            let t0 = Instant::now();
            let mut next = 0u32;
            let mut pos = 0usize;
            for chunk in prompt_ids.chunks(dec.chunk_cap()) {
                let last = pos + chunk.len() == prompt_ids.len();
                next = dec.step_chunk(chunk, pos, last)?;
                pos += chunk.len();
            }
            let mut out: Vec<u32> = Vec::new();
            for i in 0..max_tokens {
                if dec.eos().contains(&next) {
                    break;
                }
                out.push(next);
                let p = prompt_ids.len() + i;
                if p >= max_seq {
                    break;
                }
                next = dec.step(next, p, true)?;
            }
            eprintln!(
                "run {} of {runs}: {} tokens in {:.2}s",
                r + 1,
                out.len(),
                t0.elapsed().as_secs_f64()
            );
            match &first {
                None => first = Some(out),
                Some(f) if *f == out => {}
                Some(f) => bail!("run {} diverged from run 1:\n  {f:?}\n  {out:?}", r + 1),
            }
        }
        eprintln!("serial-selftest PASS: {runs} identical generations on one decoder");
        return Ok(());
    }

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

    // Prefill in chunks: each layer fetches the union of the chunk's
    // experts once instead of per token.
    eprintln!("prefill: {} tokens", prompt_ids.len());
    let t0 = Instant::now();
    let mut next = 0u32;
    let cap = dec.chunk_cap();
    // --chunk=1 restores per-token prefill (A/B); default = full chunks.
    let chunk_size = arg(&args, "chunk")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(cap)
        .clamp(1, cap);
    let mut pos = 0usize;
    for chunk in prompt_ids.chunks(chunk_size) {
        let last = pos + chunk.len() == prompt_ids.len();
        next = dec.step_chunk(chunk, pos, last)?;
        pos += chunk.len();
    }
    let prefill_s = t0.elapsed().as_secs_f64();

    // Sampling: greedy unless --temp given.
    let sampling = llmpager_run::sample::Sampling {
        temperature: arg(&args, "temp").and_then(|v| v.parse().ok()).unwrap_or(0.0),
        top_p: arg(&args, "top-p").and_then(|v| v.parse().ok()).unwrap_or(1.0),
        top_k: arg(&args, "top-k").and_then(|v| v.parse().ok()).unwrap_or(0),
        repeat_penalty: arg(&args, "repeat-penalty").and_then(|v| v.parse().ok()).unwrap_or(1.0),
        ..Default::default()
    };
    let mut rng = llmpager_run::sample::SampleRng::new(
        arg(&args, "seed").and_then(|v| v.parse().ok()).unwrap_or(0x5eed),
    );
    if !sampling.is_greedy() || sampling.repeat_penalty != 1.0 {
        next = llmpager_run::sample::sample(&dec.last_logits(), &[], &sampling, &mut rng);
    }

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
        let greedy = dec.step(next, pos, true)?;
        next = if sampling.is_greedy() && sampling.repeat_penalty == 1.0 {
            greedy
        } else {
            llmpager_run::sample::sample(&dec.last_logits(), &generated, &sampling, &mut rng)
        };
    }
    let gen_s = t1.elapsed().as_secs_f64();
    if tokenizer.is_none() {
        println!("ids: {generated:?}");
    } else {
        println!();
    }

    if let Some(f) = &profile_out {
        std::fs::write(f, serde_json::to_vec(&dec.expert_stats())?)?;
        eprintln!("profile written to {f}");
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
    if m.ram_hits > 0 {
        eprintln!(
            "ram tier: {} hits of {} fetches ({:.1}%)",
            m.ram_hits,
            m.fetches,
            100.0 * m.ram_hits as f64 / m.fetches.max(1) as f64
        );
    }
    Ok(())
}
