//! End-to-end paged-fetch benchmark: simulated MoE decode where each token
//! routes to top-k experts per layer; cache misses are read from the pack
//! (O_DIRECT) into pinned buffers and copied async into per-slot VRAM
//! buffers. This is the M1 pager's data path, measured before the model
//! exists — it tells us the ceiling for tokens/sec at a given hit rate.

use std::sync::Mutex;
use std::time::Instant;

use anyhow::Result;
use llmpager_core::cache::{ExpertCache, Lookup};
use llmpager_core::pack::{PackReader, ALIGN};

use crate::cuda::Cuda;
use crate::{Flags, Rng};

pub fn run(f: &Flags) -> Result<()> {
    let path = f.path("path")?;
    let slots = f.num("slots", 16) as u32;
    let tokens = f.num("tokens", 500);
    let topk = f.num("topk", 8) as usize;
    let hot_frac = f.frac("hot-frac", 0.8);
    let hot_size = f.num("hot-size", 24);
    let io_threads = f.num("io-threads", 8) as usize;

    let reader = PackReader::open_direct(&path)?;
    let meta = reader.meta().clone();
    let span = (reader.max_blob_bytes().div_ceil(ALIGN) * ALIGN) as usize;
    let layers = meta.num_layers;
    let experts = meta.experts_per_layer as u64;
    println!(
        "paged: {layers} layers x {experts} experts, span {span} B, {slots} slots/layer, \
         top-{topk}, hot {hot_size}@{hot_frac}, {io_threads} io threads, {tokens} tokens"
    );

    let cuda = Cuda::init()?;

    // Warm-up sanity: one pinned H2D round trip and raw H2D bandwidth.
    {
        let mb = 256 * 1024 * 1024;
        let mut pin = cuda.alloc_pinned(mb)?;
        pin.as_mut()[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let dev = cuda.alloc_device(mb)?;
        let s = cuda.stream()?;
        let start = Instant::now();
        let iters = 20;
        for _ in 0..iters {
            cuda.htod_async(dev, pin.as_ref(), s)?;
        }
        cuda.sync_stream(s)?;
        let secs = start.elapsed().as_secs_f64();
        println!(
            "h2d: pinned {:.2} GB/s",
            iters as f64 * mb as f64 / 1e9 / secs
        );
        cuda.free_device(dev);
        cuda.free_pinned(&pin);
    }

    // Device cache: one buffer per (layer, slot).
    let dev_slots: Vec<u64> = (0..layers as usize * slots as usize)
        .map(|_| cuda.alloc_device(span))
        .collect::<Result<_>>()?;
    let dev_slot = |layer: u16, slot: u32| dev_slots[layer as usize * slots as usize + slot as usize];

    // Per-IO-thread resources: its own O_DIRECT fd, pinned buffer, stream.
    let mut workers: Vec<Mutex<Worker>> = Vec::new();
    for _ in 0..io_threads {
        workers.push(Mutex::new(Worker {
            reader: PackReader::open_direct(&path)?,
            pin: cuda.alloc_pinned(span)?,
            stream: cuda.stream()?,
        }));
    }

    let mut cache = ExpertCache::new(layers, slots, 64.max(slots * 4));
    let mut rng = Rng::new(7);
    let mut fetch_ms_total = 0.0f64;
    let mut fetched_bytes = 0u64;
    let mut stalled = 0u64;
    let start = Instant::now();

    for _tok in 0..tokens {
        for layer in 0..layers {
            // Route: top-k distinct experts, skewed toward a per-layer hot set.
            let mut chosen: Vec<u16> = Vec::with_capacity(topk);
            while chosen.len() < topk.min(experts as usize) {
                let e = if rng.unit() < hot_frac {
                    // Hot set: spread deterministically across the id space.
                    let h = rng.below(hot_size.min(experts));
                    ((h * 2654435761 + layer as u64 * 97) % experts) as u16
                } else {
                    rng.below(experts) as u16
                };
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }

            // Cache pass: split hits from misses.
            let mut misses: Vec<(u16, u32)> = Vec::new(); // (expert, slot)
            let mut held: Vec<u32> = Vec::new();
            for e in &chosen {
                match cache.acquire(layer, *e) {
                    Lookup::Hit(s) => held.push(s),
                    Lookup::Miss { slot, .. } => misses.push((*e, slot)),
                    Lookup::Stalled => {
                        stalled += 1;
                        // Benchmark: skip rather than block; real pager waits.
                    }
                }
            }

            // Fetch misses in parallel across the worker pool.
            if !misses.is_empty() {
                let t0 = Instant::now();
                let bytes = std::thread::scope(|sc| -> Result<u64> {
                    let mut handles = Vec::new();
                    for (wi, chunk) in misses.chunks(misses.len().div_ceil(io_threads)).enumerate()
                    {
                        let worker = &workers[wi % io_threads];
                        let cuda = &cuda;
                        handles.push(sc.spawn(move || -> Result<u64> {
                            cuda.bind_thread(true)?;
                            let mut w = worker.lock().unwrap();
                            let mut n = 0u64;
                            for (e, slot) in chunk {
                                let len = {
                                    let Worker { reader, pin, .. } = &mut *w;
                                    reader.read_blob_into(layer, *e, pin.as_mut())?
                                };
                                let aligned = len.div_ceil(ALIGN as usize) * ALIGN as usize;
                                cuda.htod_async(
                                    dev_slot(layer, *slot),
                                    &w.pin.as_ref()[..aligned],
                                    w.stream,
                                )?;
                                cuda.sync_stream(w.stream)?;
                                n += len as u64;
                            }
                            Ok(n)
                        }));
                    }
                    let mut n = 0;
                    for h in handles {
                        n += h.join().unwrap()?;
                    }
                    Ok(n)
                })?;
                fetch_ms_total += t0.elapsed().as_secs_f64() * 1000.0;
                fetched_bytes += bytes;
                for (_, slot) in &misses {
                    cache.publish(layer, *slot);
                    held.push(*slot);
                }
            }

            for s in held {
                cache.release(layer, s);
            }
        }
    }
    cuda.sync()?;
    let secs = start.elapsed().as_secs_f64();
    let (hits, miss) = cache.stats();
    let total = hits + miss;
    println!(
        "paged: {tokens} tokens in {secs:.2}s = {:.2} tok/s  (decode-loop ceiling, no compute)",
        tokens as f64 / secs
    );
    println!(
        "cache: {hits} hits / {miss} misses = {:.1}% hit rate ({stalled} stalls)",
        100.0 * hits as f64 / total.max(1) as f64
    );
    println!(
        "fetch: {:.2} GB streamed, {:.2} ms/token avg fetch, {:.2} GB/s effective",
        fetched_bytes as f64 / 1e9,
        fetch_ms_total / tokens as f64,
        fetched_bytes as f64 / 1e9 / (fetch_ms_total / 1000.0).max(1e-9),
    );
    Ok(())
}

struct Worker {
    reader: PackReader,
    pin: crate::cuda::Pinned,
    stream: crate::cuda::CUstream,
}

// Safety: CUDA stream handles may be used from any thread that has the
// context current (bind_thread), and each Worker is owned by one thread at a
// time behind its Mutex.
unsafe impl Send for Worker {}
