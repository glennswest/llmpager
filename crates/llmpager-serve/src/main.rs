//! Minimal OpenAI-compatible HTTP server over the llmpager decode runtime.
//!
//! Endpoints:
//!   GET  /v1/models
//!   POST /v1/completions        {"prompt": "...", "max_tokens": N}
//!   POST /v1/chat/completions   {"messages": [{role, content}...], "max_tokens": N}
//!
//! One model, greedy decode, requests served serially (the decoder owns the
//! GPU). Multi-model routing is the M5 milestone.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llmpager_run::decode::Decoder;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().find_map(|a| a.strip_prefix(&format!("--{key}=")).map(String::from))
}

struct Engine {
    dec: Decoder,
    tok: tokenizers::Tokenizer,
    model_name: String,
}

impl Engine {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<(String, usize, usize, f64)> {
        let ids = self
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
            next = self.dec.step(*id, pos, pos + 1 == ids.len())?;
        }
        let mut out_ids: Vec<u32> = Vec::new();
        for i in 0..max_tokens {
            if self.dec.cfg.eos.contains(&next) {
                break;
            }
            out_ids.push(next);
            let pos = ids.len() + i;
            if pos + 1 >= 4096 {
                break;
            }
            next = self.dec.step(next, pos, true)?;
        }
        let text = self
            .tok
            .decode(&out_ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok((text, ids.len(), out_ids.len(), t0.elapsed().as_secs_f64()))
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
    let (Some(pack), Some(core), Some(tokenizer)) =
        (arg(&args, "pack"), arg(&args, "core"), arg(&args, "tokenizer"))
    else {
        bail!(
            "usage: llmpager-serve --pack=F.llmpk --core=F.core.safetensors --tokenizer=DIR \
             [--port=8090] [--slots=48] [--io-threads=4] [--direct=0|1] [--model-name=NAME]"
        );
    };
    let port: u16 = arg(&args, "port").and_then(|v| v.parse().ok()).unwrap_or(8090);
    let slots: u32 = arg(&args, "slots").and_then(|v| v.parse().ok()).unwrap_or(48);
    let io_threads: usize = arg(&args, "io-threads").and_then(|v| v.parse().ok()).unwrap_or(4);
    let direct = arg(&args, "direct").as_deref() != Some("0");
    let model_name = arg(&args, "model-name").unwrap_or_else(|| {
        PathBuf::from(&pack)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "llmpager".into())
    });

    let tok_path = {
        let p = PathBuf::from(&tokenizer);
        if p.is_dir() { p.join("tokenizer.json") } else { p }
    };
    let tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("loading {}: {e}", tok_path.display()))?;
    let dec = Decoder::new(
        &PathBuf::from(&pack),
        &PathBuf::from(&core),
        slots,
        io_threads,
        4096,
        false,
        direct,
    )?;
    let engine = Mutex::new(Engine { dec, tok, model_name: model_name.clone() });

    let server = tiny_http::Server::http(("0.0.0.0", port))
        .map_err(|e| anyhow::anyhow!("bind :{port}: {e}"))?;
    eprintln!("llmpager-serve: {model_name} on :{port}");

    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let method = req.method().as_str().to_string();
        let mut body = String::new();
        use std::io::Read;
        let _ = req.as_reader().read_to_string(&mut body);

        let resp = handle(&engine, &method, &url, &body);
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
    engine: &Mutex<Engine>,
    method: &str,
    url: &str,
    body: &str,
) -> Result<serde_json::Value> {
    match (method, url) {
        ("GET", "/v1/models") => {
            let name = engine.lock().unwrap().model_name.clone();
            Ok(serde_json::json!({
                "object": "list",
                "data": [{"id": name, "object": "model", "owned_by": "llmpager"}]
            }))
        }
        ("POST", "/v1/completions") => {
            let req: serde_json::Value = serde_json::from_str(body).context("bad JSON")?;
            let prompt = req["prompt"].as_str().context("missing prompt")?.to_string();
            let max_tokens = req["max_tokens"].as_u64().unwrap_or(128) as usize;
            let mut e = engine.lock().unwrap();
            let (text, p_toks, c_toks, secs) = e.generate(&prompt, max_tokens)?;
            Ok(serde_json::json!({
                "id": "cmpl-llmpager",
                "object": "text_completion",
                "model": e.model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": p_toks, "completion_tokens": c_toks,
                           "total_tokens": p_toks + c_toks},
                "llmpager": {"seconds": secs, "tok_per_sec": c_toks as f64 / secs.max(1e-9)}
            }))
        }
        ("POST", "/v1/chat/completions") => {
            let req: serde_json::Value = serde_json::from_str(body).context("bad JSON")?;
            let messages = req["messages"].as_array().context("missing messages")?;
            let max_tokens = req["max_tokens"].as_u64().unwrap_or(256) as usize;
            let prompt = chat_prompt(messages);
            let mut e = engine.lock().unwrap();
            let (text, p_toks, c_toks, secs) = e.generate(&prompt, max_tokens)?;
            Ok(serde_json::json!({
                "id": "chatcmpl-llmpager",
                "object": "chat.completion",
                "model": e.model_name,
                "choices": [{"index": 0,
                              "message": {"role": "assistant", "content": text},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": p_toks, "completion_tokens": c_toks,
                           "total_tokens": p_toks + c_toks},
                "llmpager": {"seconds": secs, "tok_per_sec": c_toks as f64 / secs.max(1e-9)}
            }))
        }
        _ => bail!("no such endpoint: {method} {url}"),
    }
}
