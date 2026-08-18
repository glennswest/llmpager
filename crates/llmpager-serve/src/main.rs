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
    /// Requantize the resident core to q4g64 at load. Required for models
    /// whose bf16 core alone exceeds VRAM (the 235B core is 16GB on a 16GB
    /// card); harmless but lossy elsewhere, so it defaults off.
    core_q4: bool,
    max_seq: usize,
    batch: usize,
    /// Skip routed experts under this scaled weight (kimi fetch saver).
    min_expert_weight: f64,
    /// Fetch-count profile file: loaded into the RAM tier at warm-up,
    /// rewritten with fresh stats at eviction (self-improving).
    prewarm: Option<PathBuf>,
    /// Host RAM for parked session contexts (GB). Sessions evicted from a
    /// VRAM sequence slot keep their KV here instead of re-prefilling.
    session_ram_gb: f64,
    /// VRAM to leave free for other processes sharing the card (MB). The
    /// expert cache is clamped to fit around it at warm-up and on every
    /// rebalance, so a co-tenant is not starved by cache we can give up.
    reserve_mb: u64,
}

struct Engine {
    dec: AnyDecoder,
    tok: tokenizers::Tokenizer,
    cur_slots: u32,
    max_seq: usize,
    sessions: SessionStore,
    /// VRAM reserve currently in force (bytes); admin requests change it.
    reserve_bytes: u64,
}

impl Engine {
    /// Give `id` a sequence slot and report how much of `prompt` its KV
    /// already covers. Borrows two disjoint fields, so it has to be a
    /// method rather than a free function.
    fn acquire_session(&mut self, id: &str, prompt: &[u32]) -> Result<(usize, usize)> {
        self.sessions.acquire(&mut self.dec, id, prompt)
    }
}

/// One named context: the tokens whose KV we hold, and where that KV lives.
/// Position i of the cache holds the state for `tokens[..=i]`, so any prompt
/// sharing a prefix with `tokens` can skip prefilling that prefix.
struct Session {
    tokens: Vec<u32>,
    /// Resident VRAM sequence slot, if any.
    slot: Option<usize>,
    /// Host-parked KV covering `tokens`; set exactly when `slot` is None
    /// and the context survived eviction.
    parked: Option<Vec<u8>>,
    last_used: u64,
    turns: u64,
}

/// Per-model session store. The decoder has `batch_cap` sequence slots;
/// slot 0 stays the anonymous lane (sessionless requests decode there, as
/// they always have), so sessions occupy slots 1.. and never collide with
/// legacy traffic. Sessions evicted from a slot are parked in host RAM,
/// which costs a copy of a few hundred MB/s instead of seconds of expert
/// paging to re-prefill.
struct SessionStore {
    map: std::collections::HashMap<String, Session>,
    /// owner[i] holds the session using sequence slot i + 1.
    owner: Vec<Option<String>>,
    ram_budget: usize,
    clock: u64,
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl SessionStore {
    fn new(batch_cap: usize, ram_budget: usize) -> Self {
        Self {
            map: std::collections::HashMap::new(),
            owner: (0..batch_cap.saturating_sub(1)).map(|_| None).collect(),
            ram_budget,
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn parked_bytes(&self) -> usize {
        self.map.values().filter_map(|s| s.parked.as_ref().map(|b| b.len())).sum()
    }

    /// Give `id` a slot and report (seq_slot, reusable prefix length).
    fn acquire(
        &mut self,
        dec: &mut AnyDecoder,
        id: &str,
        prompt: &[u32],
    ) -> Result<(usize, usize)> {
        if self.owner.is_empty() {
            bail!(
                "sessions need at least 2 sequence slots; raise `batch` for this \
                 model in serve.json (slot 0 is the sessionless lane)"
            );
        }
        let now = self.tick();
        let e = self.map.entry(id.to_string()).or_insert_with(|| Session {
            tokens: Vec::new(),
            slot: None,
            parked: None,
            last_used: now,
            turns: 0,
        });
        e.last_used = now;
        // Always leave one token to run: the forward pass over it is what
        // produces the logits the first new token is sampled from.
        let mut reuse =
            common_prefix(&e.tokens, prompt).min(prompt.len().saturating_sub(1));
        let stored_len = e.tokens.len();
        let resident = e.slot;
        let parked = if resident.is_none() { e.parked.take() } else { None };

        let slot = match resident {
            Some(s) => s,
            None => {
                let s = self.claim_slot(dec, id)?;
                match parked {
                    // Restoring costs one H2D copy; re-prefilling costs
                    // seconds of expert paging.
                    Some(blob) if reuse > 0 => dec.kv_import(s, stored_len, &blob)?,
                    _ => reuse = 0,
                }
                s
            }
        };
        let e = self.map.get_mut(id).expect("just inserted");
        e.slot = Some(slot);
        // From here the caller overwrites everything from `reuse` onward, so
        // only that prefix is still guaranteed to match the slot. Record it
        // now: a generation that fails part-way never reaches `commit`, and
        // a stale longer token list would let the next turn claim reuse the
        // KV can no longer honour — silently wrong output rather than a
        // slower one. On success `commit` replaces this with the full
        // context.
        e.tokens.truncate(reuse);
        Ok((slot, reuse))
    }

    /// A free sequence slot, parking the least recently used session if
    /// every slot is taken.
    fn claim_slot(&mut self, dec: &mut AnyDecoder, for_id: &str) -> Result<usize> {
        if let Some(i) = self.owner.iter().position(|o| o.is_none()) {
            self.owner[i] = Some(for_id.to_string());
            return Ok(i + 1);
        }
        let victim = self
            .owner
            .iter()
            .flatten()
            .min_by_key(|id| self.map.get(*id).map(|s| s.last_used).unwrap_or(0))
            .cloned()
            .context("no session slots")?;
        self.park(dec, &victim)?;
        let i = self.owner.iter().position(|o| o.is_none()).context("park freed no slot")?;
        self.owner[i] = Some(for_id.to_string());
        Ok(i + 1)
    }

    /// Copy a session's KV out of VRAM into host RAM and free its slot.
    fn park(&mut self, dec: &mut AnyDecoder, id: &str) -> Result<()> {
        let Some(s) = self.map.get(id) else { return Ok(()) };
        let (Some(slot), len) = (s.slot, s.tokens.len()) else { return Ok(()) };
        let blob = if len > 0 { Some(dec.kv_export(slot, len)?) } else { None };
        let s = self.map.get_mut(id).expect("checked above");
        s.parked = blob;
        s.slot = None;
        self.owner[slot - 1] = None;
        self.enforce_budget();
        Ok(())
    }

    /// Drop parked contexts, least recently used first, until the host
    /// budget is met. A dropped context loses its KV, so its token list
    /// goes too — otherwise we would claim reuse we cannot honour.
    fn enforce_budget(&mut self) {
        while self.parked_bytes() > self.ram_budget {
            let victim = self
                .map
                .iter()
                .filter(|(_, s)| s.parked.is_some())
                .min_by_key(|(_, s)| s.last_used)
                .map(|(id, _)| id.clone());
            let Some(id) = victim else { break };
            if let Some(s) = self.map.get_mut(&id) {
                s.parked = None;
                s.tokens.clear();
            }
        }
    }

    /// Record the context a finished generation leaves in the slot.
    fn commit(&mut self, id: &str, tokens: Vec<u32>) {
        let now = self.tick();
        if let Some(s) = self.map.get_mut(id) {
            s.tokens = tokens;
            s.turns += 1;
            s.last_used = now;
        }
    }

    fn forget(&mut self, id: &str) -> bool {
        match self.map.remove(id) {
            Some(s) => {
                if let Some(slot) = s.slot {
                    self.owner[slot - 1] = None;
                }
                true
            }
            None => false,
        }
    }
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
                // A VRAM reserve may have clamped it below `want`.
                engine.cur_slots = engine.dec.slots_per_layer();
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
            // Persist the workload profile so the next warm-up pre-warms.
            if let Ok(spec) = self.spec(&evicted) {
                if let Some(f) = &spec.prewarm {
                    if let Ok(j) = serde_json::to_vec(&engine.dec.expert_stats()) {
                        let _ = std::fs::write(f, j);
                    }
                }
            }
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
                spec.core_q4, spec.direct, (spec.ram_gb * 1e9) as u64,
                spec.reserve_mb * 1_000_000, spec.batch,
            )?;
            let mut dec = dec;
            dec.set_min_expert_weight(spec.min_expert_weight as f32);
            if let Some(f) = &spec.prewarm {
                if let Ok(raw) = std::fs::read(f) {
                    if let Ok(counts) = serde_json::from_slice::<Vec<u64>>(&raw) {
                        let t = Instant::now();
                        match dec.prewarm(&counts) {
                            Ok(n) => eprintln!(
                                "prewarm: {n} experts in {:.1}s",
                                t.elapsed().as_secs_f64()
                            ),
                            Err(e) => eprintln!("prewarm failed: {e:#}"),
                        }
                    }
                }
            }
            let sessions =
                SessionStore::new(dec.batch_cap(), (spec.session_ram_gb * 1e9) as usize);
            Ok(Engine {
                dec,
                tok,
                cur_slots: slots,
                max_seq: spec.max_seq,
                sessions,
                reserve_bytes: spec.reserve_mb * 1_000_000,
            })
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
        let mut engine = engine;
        engine.cur_slots = engine.dec.slots_per_layer();
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

fn tokenize(engine: &Engine, prompt: &str) -> Result<Vec<u32>> {
    let ids = engine
        .tok
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    if ids.is_empty() {
        bail!("empty prompt after tokenization");
    }
    Ok(ids)
}

/// Generate inside a session: decode on the session's own sequence slot,
/// prefilling only the part of the prompt its KV does not already cover.
/// Returns (text, full context after this turn, prefilled, generated, secs).
#[allow(clippy::too_many_arguments)]
fn generate_session(
    engine: &mut Engine,
    slot: usize,
    reuse: usize,
    ids: &[u32],
    max_tokens: usize,
    sampling: &Sampling,
    mut on_delta: Option<&mut dyn FnMut(&str)>,
) -> Result<(String, Vec<u32>, usize, usize, f64)> {
    let t0 = Instant::now();
    let cap = engine.dec.chunk_cap();
    let plain_greedy = sampling.is_greedy() && sampling.repeat_penalty == 1.0;
    let mut rng = SampleRng::new(sampling.seed);
    let mut next = 0u32;

    // Prefill the divergent suffix only. Positions before `reuse` keep the
    // KV the session already holds; stale entries after the new context are
    // harmless, since every position is rewritten before it is attended to.
    let mut pos = reuse;
    for chunk in ids[reuse..].chunks(cap) {
        let entries: Vec<(u32, usize, usize)> =
            chunk.iter().enumerate().map(|(i, t)| (*t, pos + i, slot)).collect();
        let last = pos + chunk.len() == ids.len();
        let out = engine.dec.step_multi(&entries, last)?;
        if last {
            next = *out.last().context("prefill produced no token")?;
            if !plain_greedy {
                if let Some(l) =
                    engine.dec.last_logits_multi_or_single().last().filter(|l| !l.is_empty())
                {
                    next = sample(l, &[], sampling, &mut rng);
                }
            }
        }
        pos += chunk.len();
    }

    let mut out_ids: Vec<u32> = Vec::new();
    // Generated tokens whose KV actually landed in the slot. Every token is
    // fed back before the next is produced, so this is normally all of them
    // -- except the last one at the max_seq boundary, which is emitted and
    // then never stepped. The stored context must not claim it.
    let mut kv_tokens = 0usize;
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
        let p = ids.len() + i;
        if p + 1 >= engine.max_seq {
            // Emitted but never fed back, so it has no KV — see `kv_tokens`.
            break;
        }
        let out = engine.dec.step_multi(&[(next, p, slot)], true)?;
        kv_tokens = out_ids.len();
        next = if plain_greedy {
            out[0]
        } else {
            let lm = engine.dec.last_logits_multi_or_single();
            sample(&lm[0], &out_ids, sampling, &mut rng)
        };
    }
    let text = engine
        .tok
        .decode(&out_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    // The context handed back must describe exactly what the slot holds.
    let mut context = ids.to_vec();
    context.extend_from_slice(&out_ids[..kv_tokens]);
    Ok((text, context, ids.len() - reuse, out_ids.len(), t0.elapsed().as_secs_f64()))
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
                core_q4: m["core_q4"].as_bool().unwrap_or(false),
                max_seq: m["max_seq"].as_u64().unwrap_or(4096) as usize,
                batch: m["batch"].as_u64().unwrap_or(1) as usize,
                min_expert_weight: m["min_expert_weight"].as_f64().unwrap_or(0.0),
                prewarm: m["prewarm"].as_str().map(PathBuf::from),
                session_ram_gb: m["session_ram_gb"].as_f64().unwrap_or(8.0),
                reserve_mb: m["reserve_mb"]
                    .as_u64()
                    .or_else(|| cfg["reserve_mb"].as_u64())
                    .unwrap_or(0),
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
            // A named context. Two requests naming the same session share
            // KV: continuations skip re-prefilling the conversation, and
            // fan-outs behind one system prompt skip re-prefilling it.
            let session = req["session"]
                .as_str()
                .or_else(|| req["session_id"].as_str())
                .map(String::from);
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
                    if let Some(sid) = &session {
                        bail!(
                            "session {sid:?} cannot be combined with a prompt array or n>1: \
                             batch streams occupy the sequence slots sessions live in"
                        );
                    }
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
                    let sess = match &session {
                        Some(sid) => match tokenize(engine, &prompt)
                            .and_then(|ids| Ok((engine.acquire_session(sid, &ids)?, ids)))
                        {
                            Ok(((slot, reuse), ids)) => Some((sid.clone(), slot, reuse, ids)),
                            Err(e) => {
                                send(&serde_json::json!({"error": {"message": format!("{e:#}")}}));
                                return;
                            }
                        },
                        None => None,
                    };
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
                    // Streaming reports the same session accounting the
                    // non-streaming path does, in the final event.
                    let mut sess_info = None;
                    let result = match &sess {
                        Some((sid, slot, reuse, ids)) => generate_session(
                            engine, *slot, *reuse, ids, max_tokens, &sampling, Some(&mut cb),
                        )
                        .map(|(_, ctx, prefilled, c, secs)| {
                            let p = ids.len();
                            engine.sessions.commit(sid, ctx);
                            sess_info = Some(serde_json::json!({"session": sid, "slot": slot,
                                "prompt_tokens_reused": reuse,
                                "prompt_tokens_prefilled": prefilled}));
                            (String::new(), p, c, secs)
                        }),
                        None => generate(engine, &prompt, max_tokens, &sampling, Some(&mut cb)),
                    };
                    match result {
                        Ok((_, p, c, secs)) => {
                            let done_choice = if chat {
                                serde_json::json!({"index": 0, "delta": {}, "finish_reason": "stop"})
                            } else {
                                serde_json::json!({"index": 0, "text": "", "finish_reason": "stop"})
                            };
                            let mut perf = serde_json::json!({"seconds": secs,
                                "tok_per_sec": c as f64 / secs.max(1e-9)});
                            if let (Some(info), Some(o)) = (sess_info, perf.as_object_mut()) {
                                if let Some(fields) = info.as_object() {
                                    o.extend(fields.clone().into_iter());
                                }
                            }
                            send(&serde_json::json!({"object": obj, "model": model,
                                "choices": [done_choice],
                                "usage": {"prompt_tokens": p, "completion_tokens": c,
                                           "total_tokens": p + c},
                                "llmpager": perf}));
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
            let (text, p_toks, c_toks, secs, sess_info) = match &session {
                Some(sid) => {
                    let ids = tokenize(engine, &prompt)?;
                    let (slot, reuse) = engine.acquire_session(sid, &ids)?;
                    let (text, ctx, prefilled, c, secs) = generate_session(
                        engine, slot, reuse, &ids, max_tokens, &sampling, None,
                    )?;
                    engine.sessions.commit(sid, ctx);
                    let info = serde_json::json!({"session": sid, "slot": slot,
                        "prompt_tokens_reused": reuse,
                        "prompt_tokens_prefilled": prefilled});
                    (text, ids.len(), c, secs, Some(info))
                }
                None => {
                    let (t, p, c, s) = generate(engine, &prompt, max_tokens, &sampling, None)?;
                    (t, p, c, s, None)
                }
            };
            let usage = serde_json::json!({"prompt_tokens": p_toks,
                "completion_tokens": c_toks, "total_tokens": p_toks + c_toks});
            let mut perf = serde_json::json!({"seconds": secs,
                "tok_per_sec": c_toks as f64 / secs.max(1e-9)});
            if let (Some(info), Some(obj)) = (sess_info, perf.as_object_mut()) {
                if let Some(fields) = info.as_object() {
                    obj.extend(fields.clone().into_iter());
                }
            }
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
        // Co-tenancy control. CUDA has no cross-process pressure signal, so
        // a neighbour that needs the card cannot ask for it implicitly —
        // these let it ask explicitly, reusing the budgeter's resize path.
        ("GET", "/v1/admin/vram") => {
            let r = registry.lock().unwrap();
            let mem = r.warm.first().map(|(_, e)| e.dec.mem_info());
            let (free, total) = match mem {
                Some(Ok(v)) => v,
                Some(Err(e)) => bail!("mem_info: {e:#}"),
                None => (0, 0),
            };
            let models: Vec<_> = r
                .warm
                .iter()
                .map(|(n, e)| serde_json::json!({"model": n, "slots": e.cur_slots}))
                .collect();
            Ok(Resp::Json(serde_json::json!({
                "free_mb": free / 1_000_000, "total_mb": total / 1_000_000,
                "warm": models})))
        }
        ("POST", "/v1/admin/slots") => {
            let req: serde_json::Value = serde_json::from_str(body).context("bad JSON")?;
            let target = req["target"].as_u64().map(|v| v as u32);
            let reserve_mb = req["reserve_mb"].as_u64();
            if target.is_none() && reserve_mb.is_none() {
                bail!("expected a \"target\" slot count or a \"reserve_mb\" free-VRAM target");
            }
            let mut r = registry.lock().unwrap();
            let names: Vec<String> = r.warm.iter().map(|(n, _)| n.clone()).collect();
            let mut out = Vec::new();
            for name in names {
                // Resizing frees the old arena first, so the reserve is
                // measured against genuinely free memory.
                let want = match target {
                    Some(t) => t,
                    None => r.spec(&name)?.slots_solo,
                };
                let Some((_, engine)) = r.warm.iter_mut().find(|(n, _)| n == &name) else {
                    continue;
                };
                let prev_reserve = engine.reserve_bytes;
                if let Some(mb) = reserve_mb {
                    engine.reserve_bytes = mb * 1_000_000;
                    engine.dec.set_reserve_bytes(engine.reserve_bytes);
                }
                if let Err(e) = engine.dec.resize_cache(want) {
                    // Do not leave an unsatisfiable reserve armed: the next
                    // rebalance would trip over it too.
                    engine.reserve_bytes = prev_reserve;
                    engine.dec.set_reserve_bytes(prev_reserve);
                    engine.cur_slots = engine.dec.slots_per_layer();
                    return Err(e);
                }
                engine.cur_slots = engine.dec.slots_per_layer();
                out.push(serde_json::json!({"model": name, "slots": engine.cur_slots}));
            }
            let (free, total) = match r.warm.first().map(|(_, e)| e.dec.mem_info()) {
                Some(Ok(v)) => v,
                _ => (0, 0),
            };
            eprintln!("admin: resized to {out:?}, {} MB free", free / 1_000_000);
            Ok(Resp::Json(serde_json::json!({
                "warm": out, "free_mb": free / 1_000_000, "total_mb": total / 1_000_000})))
        }
        ("GET", "/v1/sessions") => {
            let r = registry.lock().unwrap();
            let mut data = Vec::new();
            for (name, e) in &r.warm {
                for (id, s) in &e.sessions.map {
                    data.push(serde_json::json!({"id": id, "model": name,
                        "tokens": s.tokens.len(), "resident": s.slot.is_some(),
                        "slot": s.slot, "turns": s.turns,
                        "parked_bytes": s.parked.as_ref().map(|b| b.len()).unwrap_or(0)}));
                }
            }
            Ok(Resp::Json(serde_json::json!({"object": "list", "data": data})))
        }
        ("DELETE", u) if u.starts_with("/v1/sessions/") => {
            let id = u.trim_start_matches("/v1/sessions/");
            let mut r = registry.lock().unwrap();
            let mut dropped = 0usize;
            for (_, e) in r.warm.iter_mut() {
                if e.sessions.forget(id) {
                    dropped += 1;
                }
            }
            Ok(Resp::Json(serde_json::json!({"id": id, "deleted": dropped > 0})))
        }
        _ => bail!("no such endpoint: {method} {url}"),
    }
}
