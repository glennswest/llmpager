//! OpenAI-compatible HTTP server over the llmpager decode runtime, with a
//! multi-model registry (M5).
//!
//! Endpoints:
//!   GET  /v1/models
//!   POST /v1/completions        {"model": "...", "prompt": "...", "max_tokens": N}
//!   POST /v1/chat/completions   {"model": "...", "messages": [...], "max_tokens": N}
//!
//! Models come from a JSON config. Up to `max_warm` models stay loaded;
//! requesting a cold model evicts the least-recently-used warm one (its
//! Decoder drop returns the VRAM) and loads the new one (~seconds).
//! Greedy decode, requests served serially.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llmpager_run::any::AnyDecoder;
use llmpager_run::sample::{sample, SampleRng, Sampling};

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().find_map(|a| a.strip_prefix(&format!("--{key}=")).map(String::from))
}

#[derive(Clone)]
struct ModelSpec {
    name: String,
    pack: PathBuf,
    core: PathBuf,
    tokenizer: PathBuf,
    /// Cache size when sharing VRAM with other warm models.
    slots: u32,
    /// Cache size when this is the only warm model (budgeter grows it).
    slots_solo: u32,
    io_threads: usize,
    direct: bool,
    /// Managed host-RAM expert tier (GB); for packs larger than RAM.
    ram_gb: f64,
    max_seq: usize,
    batch: usize,
}

struct Engine {
    dec: AnyDecoder,
    tok: tokenizers::Tokenizer,
    cur_slots: u32,
    max_seq: usize,
}

struct Registry {
    specs: Vec<ModelSpec>,
    max_warm: usize,
    /// Most-recently-used first.
    warm: Vec<(String, Engine)>,
}

impl Registry {
    fn spec(&self, name: &str) -> Result<&ModelSpec> {
        self.specs
            .iter()
            .find(|s| s.name == name)
            .with_context(|| format!("unknown model {name}"))
    }

    /// Resize every warm model's cache to its target for `count` warm
    /// models: solo models get the big cache, sharers the small one.
    fn rebalance(&mut self, count: usize) -> Result<()> {
        for (name, engine) in &mut self.warm {
            let spec = self
                .specs
                .iter()
                .find(|s| &s.name == name)
                .expect("warm model has a spec");
            let want = if count <= 1 { spec.slots_solo } else { spec.slots };
            if engine.cur_slots != want {
                eprintln!("budgeter: {name} cache {} -> {want} slots", engine.cur_slots);
                engine.dec.resize_cache(want)?;
                engine.cur_slots = want;
            }
        }
        Ok(())
    }

    /// Warm the named model, evicting LRU entries as needed, and move it to
    /// the front. Returns its index in `warm` (always 0).
    fn warm_up(&mut self, name: &str) -> Result<usize> {
        if let Some(i) = self.warm.iter().position(|(n, _)| n == name) {
            let e = self.warm.remove(i);
            self.warm.insert(0, e);
            return Ok(0);
        }
        let spec = self.spec(name)?.clone();
        while self.warm.len() >= self.max_warm.max(1) {
            let (evicted, engine) = self.warm.pop().unwrap();
            drop(engine);
            eprintln!("evicted model {evicted}");
        }
        // Shrink current residents before loading another mouth to feed.
        let count_after = self.warm.len() + 1;
        self.rebalance(count_after)?;
        let slots = if count_after <= 1 { spec.slots_solo } else { spec.slots };

        let t0 = Instant::now();
        let load = |spec: &ModelSpec, slots: u32| -> Result<Engine> {
            let tok_file = if spec.tokenizer.is_dir() {
                spec.tokenizer.join("tokenizer.json")
            } else {
                spec.tokenizer.clone()
            };
            let tok = tokenizers::Tokenizer::from_file(&tok_file)
                .map_err(|e| anyhow::anyhow!("loading {}: {e}", tok_file.display()))?;
            let dec = AnyDecoder::new(
                &spec.pack, &spec.core, slots, spec.io_threads, spec.max_seq,
                false, spec.direct, (spec.ram_gb * 1e9) as u64, spec.batch,
            )?;
            Ok(Engine { dec, tok, cur_slots: slots, max_seq: spec.max_seq })
        };
        let engine = match load(&spec, slots) {
            Ok(e) => e,
            Err(first_err) => {
                // Likely VRAM pressure: drop everything warm and retry once.
                if self.warm.is_empty() {
                    return Err(first_err);
                }
                eprintln!("load of {name} failed ({first_err:#}); evicting all warm models and retrying");
                self.warm.clear();
                load(&spec, self.spec(name)?.slots_solo)?
            }
        };
        eprintln!("loaded model {name} ({} slots) in {:.1}s", engine.cur_slots, t0.elapsed().as_secs_f64());
        self.warm.insert(0, (name.to_string(), engine));
        // If eviction/retry left us solo, grow back.
        let n = self.warm.len();
        self.rebalance(n)?;
        Ok(0)
    }
}

/// Generate with the given sampling settings (greedy when temperature 0);
/// when `on_delta` is given, call it with each new text fragment (streaming).
fn generate(
    engine: &mut Engine,
    prompt: &str,
    max_tokens: usize,
    sampling: &Sampling,
    mut on_delta: Option<&mut dyn FnMut(&str)>,
) -> Result<(String, usize, usize, f64)> {
    let ids = engine
        .tok
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    if ids.is_empty() {
        bail!("empty prompt after tokenization");
    }
    let t0 = Instant::now();
    // Union prefill: chunked so each layer fetches each expert once per chunk.
    let mut next = 0u32;
    let cap = engine.dec.chunk_cap();
    let mut pos = 0usize;
    for chunk in ids.chunks(cap) {
        let last = pos + chunk.len() == ids.len();
        next = engine.dec.step_chunk(chunk, pos, last)?;
        pos += chunk.len();
    }
    let plain_greedy = sampling.is_greedy() && sampling.repeat_penalty == 1.0;
    let mut rng = SampleRng::new(sampling.seed);
    if !plain_greedy {
        next = sample(&engine.dec.last_logits(), &[], sampling, &mut rng);
    }
    let mut out_ids: Vec<u32> = Vec::new();
    let mut printed = String::new();
    for i in 0..max_tokens {
        if engine.dec.eos().contains(&next) {
            break;
        }
        out_ids.push(next);
        if let Some(cb) = on_delta.as_deref_mut() {
            let full = engine
                .tok
                .decode(&out_ids, true)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            if full.len() > printed.len() {
                cb(&full[printed.len()..]);
                printed = full;
            }
        }
        let pos = ids.len() + i;
        if pos + 1 >= engine.max_seq {
            break;
        }
        let greedy = engine.dec.step(next, pos, true)?;
        next = if plain_greedy {
            greedy
        } else {
            sample(&engine.dec.last_logits(), &out_ids, sampling, &mut rng)
        };
    }
    let text = engine
        .tok
        .decode(&out_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    Ok((text, ids.len(), out_ids.len(), t0.elapsed().as_secs_f64()))
}

/// Lockstep batch generation: N prompts decode together on independent
/// sequence slots, sharing every layer's expert fetches. Greedy or sampled
/// per stream (seed offset by stream index). Returns per-stream
/// (text, prompt_tokens, completion_tokens).
fn generate_batch(
    engine: &mut Engine,
    prompts: &[String],
    max_tokens: usize,
    sampling: &Sampling,
) -> Result<Vec<(String, usize, usize)>> {
    let nstr = prompts.len();
    if nstr > engine.dec.batch_cap() {
        bail!(
            "{nstr} prompts exceed this model's batch capacity {} (raise `batch` in serve.json)",
            engine.dec.batch_cap()
        );
    }
    let cap = engine.dec.chunk_cap();
    let mut ids: Vec<Vec<u32>> = Vec::with_capacity(nstr);
    for p in prompts {
        let v = engine
            .tok
            .encode(p.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        if v.is_empty() {
            bail!("empty prompt after tokenization");
        }
        ids.push(v);
    }

    // Prefill each stream on its own slot (chunked union fetching). The
    // logits buffer only survives until the next step_multi call, so the
    // sampled first token is drawn inside each stream's prefill.
    let plain_greedy = sampling.is_greedy() && sampling.repeat_penalty == 1.0;
    let mut rngs: Vec<SampleRng> =
        (0..nstr).map(|s| SampleRng::new(sampling.seed.wrapping_add(s as u64))).collect();
    let mut next = vec![0u32; nstr];
    for s in 0..nstr {
        let mut pos = 0usize;
        for chunk in ids[s].chunks(cap) {
            let entries: Vec<(u32, usize, usize)> =
                chunk.iter().enumerate().map(|(i, t)| (*t, pos + i, s)).collect();
            let last = pos + chunk.len() == ids[s].len();
            let out = engine.dec.step_multi(&entries, last)?;
            if last {
                next[s] = *out.last().unwrap();
                if !plain_greedy {
                    if let Some(l) = engine
                        .dec
                        .last_logits_multi_or_single()
                        .last()
                        .filter(|l| !l.is_empty())
                    {
                        next[s] = sample(l, &[], sampling, &mut rngs[s]);
                    }
                }
            }
            pos += chunk.len();
        }
    }

    // Lockstep decode; streams retire on EOS or max_seq independently.
    let mut out_ids: Vec<Vec<u32>> = vec![Vec::new(); nstr];
    let mut done = vec![false; nstr];
    let mut pos: Vec<usize> = ids.iter().map(|v| v.len()).collect();
    for _ in 0..max_tokens {
        for s in 0..nstr {
            if done[s] {
                continue;
            }
            if engine.dec.eos().contains(&next[s]) || pos[s] >= engine.max_seq {
                done[s] = true;
                continue;
            }
            out_ids[s].push(next[s]);
        }
        let active: Vec<usize> = (0..nstr).filter(|&s| !done[s]).collect();
        if active.is_empty() {
            break;
        }
        let entries: Vec<(u32, usize, usize)> =
            active.iter().map(|&s| (next[s], pos[s], s)).collect();
        let out = engine.dec.step_multi(&entries, true)?;
        let lm = engine.dec.last_logits_multi_or_single();
        for (i, &s) in active.iter().enumerate() {
            next[s] = if plain_greedy {
                out[i]
            } else {
                sample(&lm[i], &out_ids[s], sampling, &mut rngs[s])
            };
            pos[s] += 1;
        }
    }

    let mut res = Vec::with_capacity(nstr);
    for s in 0..nstr {
        let text = engine
            .tok
            .decode(&out_ids[s], true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        res.push((text, ids[s].len(), out_ids[s].len()));
    }
    Ok(res)
}

/// Blocking reader over an mpsc of byte chunks — bridges a generation
/// thread into tiny_http's streaming response body.
struct ChannelReader {
    rx: std::sync::mpsc::Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}

impl std::io::Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(chunk) => {
                    self.buf = chunk;
                    self.pos = 0;
                }
                Err(_) => return Ok(0), // sender gone: EOF
            }
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Qwen3 ChatML-style template.
fn chat_prompt(messages: &[serde_json::Value]) -> String {
    let mut s = String::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("user");
        let content = m["content"].as_str().unwrap_or("");
        s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

/// Kimi K2.x template (from the checkpoint's chat_template.jinja):
/// role marker + name + <|im_middle|> + content + <|im_end|>.
fn kimi_chat_prompt(messages: &[serde_json::Value]) -> String {
    let mut s = String::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("user");
        let content = m["content"].as_str().unwrap_or("");
        let marker = match role {
            "user" => "<|im_user|>",
            "assistant" => "<|im_assistant|>",
            _ => "<|im_system|>",
        };
        s.push_str(&format!("{marker}{role}<|im_middle|>{content}<|im_end|>"));
    }
    s.push_str("<|im_assistant|>assistant<|im_middle|>");
    s
}

/// OpenAI-style sampling fields; absent fields keep greedy defaults.
fn sampling_from_request(req: &serde_json::Value) -> Sampling {
    Sampling {
        temperature: req["temperature"].as_f64().unwrap_or(0.0) as f32,
        top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
        top_k: req["top_k"].as_u64().unwrap_or(0) as usize,
        repeat_penalty: req["repetition_penalty"].as_f64().unwrap_or(1.0) as f32,
        seed: req["seed"].as_u64().unwrap_or(0x5eed),
        ..Default::default()
    }
}

fn json_response(status: u32, v: &serde_json::Value) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::to_vec(v).unwrap_or_default();
    tiny_http::Response::from_data(body)
        .with_status_code(status as u16)
        .with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
        )
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(config_path) = arg(&args, "config") else {
        bail!("usage: llmpager-serve --config=serve.json  (see deploy/serve.json)");
    };
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).context("reading config")?)?;
    let port = cfg["port"].as_u64().unwrap_or(8090) as u16;
    let max_warm = cfg["max_warm"].as_u64().unwrap_or(2) as usize;
    let specs: Vec<ModelSpec> = cfg["models"]
        .as_array()
        .context("config: models[]")?
        .iter()
        .map(|m| {
            Ok(ModelSpec {
                name: m["name"].as_str().context("model name")?.to_string(),
                pack: PathBuf::from(m["pack"].as_str().context("model pack")?),
                core: PathBuf::from(m["core"].as_str().context("model core")?),
                tokenizer: PathBuf::from(m["tokenizer"].as_str().context("model tokenizer")?),
                slots: m["slots"].as_u64().unwrap_or(32) as u32,
                slots_solo: m["slots_solo"]
                    .as_u64()
                    .map(|v| v as u32)
                    .unwrap_or(2 * m["slots"].as_u64().unwrap_or(32) as u32),
                io_threads: m["io_threads"].as_u64().unwrap_or(4) as usize,
                direct: m["direct"].as_bool().unwrap_or(false),
                ram_gb: m["ram_gb"].as_f64().unwrap_or(0.0),
                max_seq: m["max_seq"].as_u64().unwrap_or(4096) as usize,
                batch: m["batch"].as_u64().unwrap_or(1) as usize,
            })
        })
        .collect::<Result<_>>()?;
    if specs.is_empty() {
        bail!("config: at least one model required");
    }
    let default_model = specs[0].name.clone();
    let registry = Arc::new(Mutex::new(Registry { specs, max_warm, warm: Vec::new() }));

    // Warm the default model up front so the first request is fast.
    registry.lock().unwrap().warm_up(&default_model)?;

    let server = tiny_http::Server::http(("0.0.0.0", port))
        .map_err(|e| anyhow::anyhow!("bind :{port}: {e}"))?;
    eprintln!("llmpager-serve: {} model(s), default {default_model}, :{port}", registry.lock().unwrap().specs.len());

    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let method = req.method().as_str().to_string();
        let mut body = String::new();
        use std::io::Read;
        let _ = req.as_reader().read_to_string(&mut body);
        let resp = handle(&registry, &default_model, &method, &url, &body);
        let _ = match resp {
            Ok(Resp::Json(v)) => req.respond(json_response(200, &v)),
            Ok(Resp::Stream(reader)) => {
                let headers = vec![
                    tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap(),
                    tiny_http::Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).unwrap(),
                ];
                req.respond(tiny_http::Response::new(200.into(), headers, reader, None, None))
            }
            Err(e) => req.respond(json_response(
                400,
                &serde_json::json!({"error": {"message": format!("{e:#}")}}),
            )),
        };
    }
    Ok(())
}

enum Resp {
    Json(serde_json::Value),
    Stream(ChannelReader),
}

fn handle(
    registry: &Arc<Mutex<Registry>>,
    default_model: &str,
    method: &str,
    url: &str,
    body: &str,
) -> Result<Resp> {
    match (method, url) {
        ("GET", "/v1/models") => {
            let r = registry.lock().unwrap();
            let data: Vec<_> = r
                .specs
                .iter()
                .map(|s| {
                    let slots = r
                        .warm
                        .iter()
                        .find(|(n, _)| n == &s.name)
                        .map(|(_, e)| e.cur_slots);
                    serde_json::json!({"id": s.name, "object": "model",
                                        "owned_by": "llmpager",
                                        "warm": slots.is_some(), "slots": slots})
                })
                .collect();
            Ok(Resp::Json(serde_json::json!({"object": "list", "data": data})))
        }
        ("POST", "/v1/completions") | ("POST", "/v1/chat/completions") => {
            let chat = url == "/v1/chat/completions";
            let req: serde_json::Value = serde_json::from_str(body).context("bad JSON")?;
            let model = req["model"].as_str().unwrap_or(default_model).to_string();
            let stream = req["stream"].as_bool().unwrap_or(false);
            let max_tokens =
                req["max_tokens"].as_u64().unwrap_or(if chat { 256 } else { 128 }) as usize;
            let sampling = sampling_from_request(&req);
            // The chat template depends on the engine, so warm it first.
            let kimi = {
                let mut r = registry.lock().unwrap();
                r.warm_up(&model)?;
                r.warm[0].1.dec.is_kimi()
            };
            // Batch forms (completions only): "prompt" as an array, or n>1
            // duplicating one prompt. Streams decode in lockstep, sharing
            // every expert fetch.
            if !chat && !stream {
                let n_req = req["n"].as_u64().unwrap_or(1) as usize;
                let arr: Option<Vec<String>> = req["prompt"].as_array().map(|a| {
                    a.iter().filter_map(|v| v.as_str().map(String::from)).collect()
                });
                let prompts: Option<Vec<String>> = match (arr, n_req) {
                    (Some(v), _) if v.len() > 1 => Some(v),
                    (None, n) if n > 1 => req["prompt"]
                        .as_str()
                        .map(|p| vec![p.to_string(); n]),
                    _ => None,
                };
                if let Some(prompts) = prompts {
                    let t0 = Instant::now();
                    let mut r = registry.lock().unwrap();
                    r.warm_up(&model)?;
                    let engine = &mut r.warm[0].1;
                    let results = generate_batch(engine, &prompts, max_tokens, &sampling)?;
                    let secs = t0.elapsed().as_secs_f64();
                    let (p_sum, c_sum): (usize, usize) = results
                        .iter()
                        .fold((0, 0), |(p, c), (_, pp, cc)| (p + pp, c + cc));
                    let choices: Vec<_> = results
                        .iter()
                        .enumerate()
                        .map(|(i, (text, _, _))| {
                            serde_json::json!({"index": i, "text": text,
                                                "finish_reason": "stop"})
                        })
                        .collect();
                    return Ok(Resp::Json(serde_json::json!({
                        "id": "cmpl-llmpager", "object": "text_completion", "model": model,
                        "choices": choices,
                        "usage": {"prompt_tokens": p_sum, "completion_tokens": c_sum,
                                   "total_tokens": p_sum + c_sum},
                        "llmpager": {"seconds": secs, "streams": prompts.len(),
                            "tok_per_sec_aggregate": c_sum as f64 / secs.max(1e-9)}})));
                }
            }

            let prompt = if chat {
                let messages = req["messages"].as_array().context("missing messages")?;
                if kimi {
                    kimi_chat_prompt(messages)
                } else {
                    chat_prompt(messages)
                }
            } else {
                req["prompt"].as_str().context("missing prompt")?.to_string()
            };

            if stream {
                // SSE: a generation thread owns the registry lock and feeds
                // chunks through a channel that backs the response body.
                let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
                let reg = Arc::clone(registry);
                std::thread::spawn(move || {
                    let send = |v: &serde_json::Value| {
                        let _ = tx.send(format!("data: {v}\n\n").into_bytes());
                    };
                    let mut r = reg.lock().unwrap();
                    if let Err(e) = r.warm_up(&model) {
                        send(&serde_json::json!({"error": {"message": format!("{e:#}")}}));
                        return;
                    }
                    let engine = &mut r.warm[0].1;
                    let obj = if chat { "chat.completion.chunk" } else { "text_completion" };
                    if chat {
                        send(&serde_json::json!({"object": obj, "model": model,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}}]}));
                    }
                    let mut cb = |piece: &str| {
                        let choice = if chat {
                            serde_json::json!({"index": 0, "delta": {"content": piece}})
                        } else {
                            serde_json::json!({"index": 0, "text": piece})
                        };
                        send(&serde_json::json!({"object": obj, "model": model,
                                                  "choices": [choice]}));
                    };
                    match generate(engine, &prompt, max_tokens, &sampling, Some(&mut cb)) {
                        Ok((_, p, c, secs)) => {
                            let done_choice = if chat {
                                serde_json::json!({"index": 0, "delta": {}, "finish_reason": "stop"})
                            } else {
                                serde_json::json!({"index": 0, "text": "", "finish_reason": "stop"})
                            };
                            send(&serde_json::json!({"object": obj, "model": model,
                                "choices": [done_choice],
                                "usage": {"prompt_tokens": p, "completion_tokens": c,
                                           "total_tokens": p + c},
                                "llmpager": {"seconds": secs,
                                              "tok_per_sec": c as f64 / secs.max(1e-9)}}));
                            let _ = tx.send(b"data: [DONE]\n\n".to_vec());
                        }
                        Err(e) => {
                            send(&serde_json::json!({"error": {"message": format!("{e:#}")}}));
                        }
                    }
                });
                return Ok(Resp::Stream(ChannelReader { rx, buf: Vec::new(), pos: 0 }));
            }

            let mut r = registry.lock().unwrap();
            r.warm_up(&model)?;
            let engine = &mut r.warm[0].1;
            let (text, p_toks, c_toks, secs) = generate(engine, &prompt, max_tokens, &sampling, None)?;
            let usage = serde_json::json!({"prompt_tokens": p_toks,
                "completion_tokens": c_toks, "total_tokens": p_toks + c_toks});
            let perf = serde_json::json!({"seconds": secs,
                "tok_per_sec": c_toks as f64 / secs.max(1e-9)});
            Ok(Resp::Json(if chat {
                serde_json::json!({
                    "id": "chatcmpl-llmpager", "object": "chat.completion", "model": model,
                    "choices": [{"index": 0,
                                  "message": {"role": "assistant", "content": text},
                                  "finish_reason": "stop"}],
                    "usage": usage, "llmpager": perf})
            } else {
                serde_json::json!({
                    "id": "cmpl-llmpager", "object": "text_completion", "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                    "usage": usage, "llmpager": perf})
            }))
        }
        _ => bail!("no such endpoint: {method} {url}"),
    }
}
