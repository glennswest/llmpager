//! M1 benchmark: same workload as `paged`, but through the async
//! `llmpager_cuda::pager::Pager` — io worker pool, event-based readiness,
//! and layer-ahead prefetch. Routing for a whole token is computed up front
//! (a perfect predictor), so `--prefetch=N` fetches layer L+N's experts
//! while layer L "computes"; real models get this from router-output
//! heuristics instead (M3).

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use llmpager_core::pack::PackReader;
use llmpager_cuda::driver::Cuda;
use llmpager_cuda::pager::{Pager, PagerConfig};

use crate::{route, Flags, Rng};

pub fn run(f: &Flags) -> Result<()> {
    let path = f.path("path")?;
    let slots = f.num("slots", 32) as u32;
    let tokens = f.num("tokens", 500);
    let topk = f.num("topk", 8) as usize;
    let hot_frac = f.frac("hot-frac", 0.8);
    let hot_size = f.num("hot-size", 24);
    let io_threads = f.num("io-threads", 4) as usize;
    let prefetch = f.num("prefetch", 1) as u16;

    let (layers, experts) = {
        let r = PackReader::open(&path)?;
        (r.meta().num_layers, r.meta().experts_per_layer as u64)
    };
    println!(
        "pager: {layers} layers x {experts} experts, {slots} slots/layer, top-{topk}, \
         hot {hot_size}@{hot_frac}, {io_threads} io threads, prefetch={prefetch}, {tokens} tokens"
    );

    let cuda = Arc::new(Cuda::init()?);
    let pager = Pager::new(
        Arc::clone(&cuda),
        &path,
        PagerConfig { slots_per_layer: slots, io_threads, decay_interval: 64.max(slots * 4) },
    )?;

    let mut rng = Rng::new(7);
    let mut wait_ms_total = 0.0f64;
    let start = Instant::now();

    for _tok in 0..tokens {
        // Whole-token routing up front — stands in for a perfect predictor.
        let routing: Vec<Vec<u16>> = (0..layers)
            .map(|l| route(&mut rng, l, experts, topk, hot_frac, hot_size))
            .collect();

        for layer in 0..layers {
            let ahead = layer + prefetch;
            if prefetch > 0 && ahead < layers {
                pager.prefetch(ahead, &routing[ahead as usize]);
            }
            let handles = pager.request(layer, &routing[layer as usize])?;
            let t0 = Instant::now();
            for h in &handles {
                pager.wait(h)?;
            }
            wait_ms_total += t0.elapsed().as_secs_f64() * 1000.0;
            // Compute would run here, reading h.dev on its stream.
            for h in handles {
                pager.release(h);
            }
        }
    }
    cuda.sync()?;
    let secs = start.elapsed().as_secs_f64();
    let m = pager.metrics();
    println!(
        "pager: {tokens} tokens in {secs:.2}s = {:.2} tok/s  (decode-loop ceiling, no compute)",
        tokens as f64 / secs
    );
    println!(
        "cache: {} hits / {} misses = {:.1}% hit rate",
        m.hits,
        m.misses,
        100.0 * m.hit_rate()
    );
    println!(
        "fetch: {:.2} GB in {} fetches, wait {:.2} ms/token, latency {}",
        m.bytes_fetched as f64 / 1e9,
        m.fetches,
        wait_ms_total / tokens as f64,
        m.histogram()
    );
    Ok(())
}
