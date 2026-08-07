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
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llmpager_run::decode::Decoder;

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
}

struct Engine {
    dec: Decoder,
    tok: tokenizers::Tokenizer,
    cur_slots: u32,
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
            let dec = Decoder::new(
                &spec.pack, &spec.core, slots, spec.io_threads, 4096, false, spec.direct,
            )?;
            Ok(Engine { dec, tok, cur_slots: slots })
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

fn generate(engine: &mut Engine, prompt: &str, max_tokens: usize) -> Result<(String, usize, usize, f64)> {
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
    let mut next = 0u32;
    for (pos, id) in ids.iter().enumerate() {
        next = engine.dec.step(*id, pos, pos + 1 == ids.len())?;
    }
    let mut out_ids: Vec<u32> = Vec::new();
    for i in 0..max_tokens {
        if engine.dec.cfg.eos.contains(&next) {
            break;
        }
        out_ids.push(next);
        let pos = ids.len() + i;
        if pos + 1 >= 4096 {
            break;
        }
        next = engine.dec.step(next, pos, true)?;
    }
    let text = engine
        .tok
        .decode(&out_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    Ok((text, ids.len(), out_ids.len(), t0.elapsed().as_secs_f64()))
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
            })
        })
        .collect::<Result<_>>()?;
    if specs.is_empty() {
        bail!("config: at least one model required");
    }
    let default_model = specs[0].name.clone();
    let registry = Mutex::new(Registry { specs, max_warm, warm: Vec::new() });

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
            Ok(v) => req.respond(json_response(200, &v)),
            Err(e) => req.respond(json_response(
                400,
                &serde_json::json!({"error": {"message": format!("{e:#}")}}),
            )),
        };
    }
    Ok(())
}

fn handle(
    registry: &Mutex<Registry>,
    default_model: &str,
    method: &str,
    url: &str,
    body: &str,
) -> Result<serde_json::Value> {
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
            Ok(serde_json::json!({"object": "list", "data": data}))
        }
        ("POST", "/v1/completions") | ("POST", "/v1/chat/completions") => {
            let chat = url == "/v1/chat/completions";
            let req: serde_json::Value = serde_json::from_str(body).context("bad JSON")?;
            let model = req["model"].as_str().unwrap_or(default_model).to_string();
            let max_tokens =
                req["max_tokens"].as_u64().unwrap_or(if chat { 256 } else { 128 }) as usize;
            let prompt = if chat {
                chat_prompt(req["messages"].as_array().context("missing messages")?)
            } else {
                req["prompt"].as_str().context("missing prompt")?.to_string()
            };

            let mut r = registry.lock().unwrap();
            r.warm_up(&model)?;
            let engine = &mut r.warm[0].1;
            let (text, p_toks, c_toks, secs) = generate(engine, &prompt, max_tokens)?;
            let usage = serde_json::json!({"prompt_tokens": p_toks,
                "completion_tokens": c_toks, "total_tokens": p_toks + c_toks});
            let perf = serde_json::json!({"seconds": secs,
                "tok_per_sec": c_toks as f64 / secs.max(1e-9)});
            Ok(if chat {
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
            })
        }
        _ => bail!("no such endpoint: {method} {url}"),
    }
}
