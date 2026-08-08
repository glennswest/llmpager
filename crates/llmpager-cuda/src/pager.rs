//! Async expert pager: cache lookups on the caller's thread, misses fetched
//! by an I/O worker pool (O_DIRECT pread → pinned staging → async H2D on the
//! worker's stream), readiness published under one lock + condvar.
//!
//! Concurrency model:
//! - `State` (cache bookkeeping + per-slot fill state) lives under a single
//!   mutex; a single condvar broadcasts both "a pin was released" (stall
//!   relief) and "a slot became ready".
//! - A slot being fetched holds a cache pin, so it can't be evicted while in
//!   flight; the pin transfers to the requester (`request`) or is dropped by
//!   the worker when the fetch was a prefetch.
//! - Each (layer, slot) has a CUDA event recorded after its H2D copy;
//!   `wait_stream` lets a compute stream consume an expert without any host
//!   blocking. `wait` blocks the host instead (used by the bench).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use anyhow::{bail, Result};
use llmpager_core::cache::{ExpertCache, Lookup};
use llmpager_core::pack::{PackReader, ALIGN};

use crate::driver::{CUdeviceptr, CUevent, CUstream, Cuda};

pub struct PagerConfig {
    pub slots_per_layer: u32,
    pub io_threads: usize,
    /// Cache frequency counters halve every this many insertions per layer.
    pub decay_interval: u32,
    /// O_DIRECT reads (bypass the OS page cache). True is right when the
    /// pack exceeds host RAM; false lets the page cache act as a RAM tier —
    /// misses cost a memory copy instead of a disk read once warm.
    pub direct: bool,
}

impl Default for PagerConfig {
    fn default() -> Self {
        Self { slots_per_layer: 32, io_threads: 4, decay_interval: 256, direct: true }
    }
}

/// A pinned, resident (or in-flight) expert. Call [`Pager::wait`] or
/// [`Pager::wait_stream`] before reading `dev`, and [`Pager::release`] when
/// the forward pass is done with it.
#[derive(Debug, Clone, Copy)]
pub struct ExpertHandle {
    pub layer: u16,
    pub expert: u16,
    pub slot: u32,
    /// Device address of the expert's weights.
    pub dev: CUdeviceptr,
    /// Valid byte length (blob size, unpadded).
    pub len: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Fill {
    Empty,
    InFlight,
    Ready,
}

struct Job {
    layer: u16,
    expert: u16,
    slot: u32,
    release_after_fill: bool,
}

struct State {
    cache: ExpertCache,
    fill: Vec<Fill>,
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
    dev_slots: Vec<CUdeviceptr>,
    events: Vec<CUevent>,
    slots_per_layer: u32,
    span: usize,
    bytes_fetched: AtomicU64,
    fetches: AtomicU64,
    // Latency histogram buckets: <1, <2, <5, <10, <20, <50, >=50 ms.
    lat_buckets: [AtomicU64; 7],
}

impl Shared {
    fn idx(&self, layer: u16, slot: u32) -> usize {
        layer as usize * self.slots_per_layer as usize + slot as usize
    }
}

// Safety: the raw CUevent handles in `events` are valid from any thread with
// the context current; fill/cache coordination goes through `state`'s mutex.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

pub struct Pager {
    cuda: Arc<Cuda>,
    shared: Arc<Shared>,
    index: PackReader,
    tx: Option<Sender<Job>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    arenas: Vec<CUdeviceptr>,
}

fn open_reader(path: &Path, direct: bool) -> Result<PackReader> {
    #[cfg(target_os = "linux")]
    if direct {
        return PackReader::open_direct(path);
    }
    let _ = direct;
    PackReader::open(path)
}

impl Pager {
    pub fn new(cuda: Arc<Cuda>, pack: &Path, cfg: PagerConfig) -> Result<Self> {
        let index = open_reader(pack, cfg.direct)?;
        let meta = index.meta().clone();
        let span = (index.max_blob_bytes().div_ceil(ALIGN) * ALIGN) as usize;
        let layers = meta.num_layers;
        let total_slots = layers as usize * cfg.slots_per_layer as usize;

        // One arena per layer, sliced into slots: thousands of individual
        // ~2.5MB cuMemAllocs each round up to the allocation granularity
        // (~2MB steps), wasting up to ~60% of the cache budget.
        let mut arenas: Vec<CUdeviceptr> = Vec::with_capacity(layers as usize);
        let mut dev_slots: Vec<CUdeviceptr> = Vec::with_capacity(total_slots);
        for _ in 0..layers {
            let base = cuda.alloc_device(cfg.slots_per_layer as usize * span)?;
            arenas.push(base);
            for s in 0..cfg.slots_per_layer {
                dev_slots.push(base + s as u64 * span as u64);
            }
        }
        let events: Vec<CUevent> = (0..total_slots).map(|_| cuda.event()).collect::<Result<_>>()?;

        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                cache: ExpertCache::new(layers, cfg.slots_per_layer, cfg.decay_interval),
                fill: vec![Fill::Empty; total_slots],
            }),
            cv: Condvar::new(),
            dev_slots,
            events,
            slots_per_layer: cfg.slots_per_layer,
            span,
            bytes_fetched: AtomicU64::new(0),
            fetches: AtomicU64::new(0),
            lat_buckets: Default::default(),
        });

        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let path: PathBuf = pack.to_path_buf();
        let direct = cfg.direct;
        let mut workers = Vec::new();
        for _ in 0..cfg.io_threads.max(1) {
            let cuda = Arc::clone(&cuda);
            let shared = Arc::clone(&shared);
            let rx = Arc::clone(&rx);
            let path = path.clone();
            workers.push(std::thread::spawn(move || {
                if let Err(e) = worker(cuda, shared, rx, &path, direct) {
                    eprintln!("llmpager io worker died: {e:#}");
                }
            }));
        }

        Ok(Self { cuda, shared, index, tx: Some(tx), workers, arenas })
    }

    /// Acquire pinned handles for `experts` of `layer`, dispatching fetches
    /// for misses. Blocks only if every slot in the layer is pinned (i.e.
    /// concurrent requests exceed capacity), not for I/O.
    pub fn request(&self, layer: u16, experts: &[u16]) -> Result<Vec<ExpertHandle>> {
        if experts.len() > self.shared.slots_per_layer as usize {
            bail!(
                "requested {} experts but layer has only {} slots",
                experts.len(),
                self.shared.slots_per_layer
            );
        }
        let mut out = Vec::with_capacity(experts.len());
        for &expert in experts {
            let slot = loop {
                let mut st = self.shared.state.lock().unwrap();
                match st.cache.acquire(layer, expert) {
                    Lookup::Hit(slot) => break slot,
                    Lookup::Miss { slot, .. } => {
                        let idx = self.shared.idx(layer, slot);
                        st.fill[idx] = Fill::InFlight;
                        drop(st);
                        self.tx
                            .as_ref()
                            .unwrap()
                            .send(Job { layer, expert, slot, release_after_fill: false })
                            .expect("io workers gone");
                        break slot;
                    }
                    Lookup::Stalled => {
                        let _unused = self.shared.cv.wait(st).unwrap();
                    }
                }
            };
            out.push(self.handle(layer, expert, slot));
        }
        Ok(out)
    }

    /// Best-effort: warm the cache for a future layer. Never blocks; skips
    /// experts whose layer is fully pinned. The fetch's pin is dropped by the
    /// worker after the fill, so prefetched experts are evictable again.
    pub fn prefetch(&self, layer: u16, experts: &[u16]) {
        for &expert in experts {
            let mut st = self.shared.state.lock().unwrap();
            match st.cache.acquire(layer, expert) {
                Lookup::Hit(slot) => st.cache.release(layer, slot),
                Lookup::Miss { slot, .. } => {
                    let idx = self.shared.idx(layer, slot);
                    st.fill[idx] = Fill::InFlight;
                    drop(st);
                    let _ = self.tx.as_ref().unwrap().send(Job {
                        layer,
                        expert,
                        slot,
                        release_after_fill: true,
                    });
                }
                Lookup::Stalled => {}
            }
        }
    }

    /// Block the host until the expert's weights are valid in VRAM.
    pub fn wait(&self, h: &ExpertHandle) -> Result<()> {
        let idx = self.shared.idx(h.layer, h.slot);
        let mut st = self.shared.state.lock().unwrap();
        while st.fill[idx] != Fill::Ready {
            st = self.shared.cv.wait(st).unwrap();
        }
        Ok(())
    }

    /// Device-side ordering: make `stream` wait for the expert's H2D copy.
    /// The host still waits for the copy to be *enqueued* (fill state leaves
    /// `InFlight` once the worker recorded the event).
    pub fn wait_stream(&self, h: &ExpertHandle, stream: CUstream) -> Result<()> {
        self.wait(h)?;
        let idx = self.shared.idx(h.layer, h.slot);
        self.cuda.stream_wait_event(stream, self.shared.events[idx])
    }

    pub fn release(&self, h: ExpertHandle) {
        let mut st = self.shared.state.lock().unwrap();
        st.cache.release(h.layer, h.slot);
        drop(st);
        self.shared.cv.notify_all();
    }

    pub fn slots_per_layer(&self) -> u32 {
        self.shared.slots_per_layer
    }

    pub fn metrics(&self) -> Metrics {
        let st = self.shared.state.lock().unwrap();
        let (hits, misses) = st.cache.stats();
        drop(st);
        Metrics {
            hits,
            misses,
            bytes_fetched: self.shared.bytes_fetched.load(Ordering::Relaxed),
            fetches: self.shared.fetches.load(Ordering::Relaxed),
            latency_ms_buckets: self
                .shared
                .lat_buckets
                .iter()
                .map(|b| b.load(Ordering::Relaxed))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    fn handle(&self, layer: u16, expert: u16, slot: u32) -> ExpertHandle {
        ExpertHandle {
            layer,
            expert,
            slot,
            dev: self.shared.dev_slots[self.shared.idx(layer, slot)],
            len: self.index.entry(layer, expert).nbytes as usize,
        }
    }
}

impl Drop for Pager {
    fn drop(&mut self) {
        self.tx.take(); // close the channel; workers drain and exit
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        // Model unloading (M5) reclaims the VRAM this pager held.
        for a in self.arenas.drain(..) {
            self.cuda.free_device(a);
        }
        for e in &self.shared.events {
            self.cuda.destroy_event(*e);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Metrics {
    pub hits: u64,
    pub misses: u64,
    pub bytes_fetched: u64,
    pub fetches: u64,
    /// Fetch wall-time histogram: <1, <2, <5, <10, <20, <50, >=50 ms.
    pub latency_ms_buckets: [u64; 7],
}

impl Metrics {
    pub fn hit_rate(&self) -> f64 {
        self.hits as f64 / (self.hits + self.misses).max(1) as f64
    }

    pub fn histogram(&self) -> String {
        let labels = ["<1ms", "<2ms", "<5ms", "<10ms", "<20ms", "<50ms", ">=50ms"];
        labels
            .iter()
            .zip(self.latency_ms_buckets.iter())
            .map(|(l, n)| format!("{l}:{n}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn worker(
    cuda: Arc<Cuda>,
    shared: Arc<Shared>,
    rx: Arc<Mutex<Receiver<Job>>>,
    pack: &Path,
    direct: bool,
) -> Result<()> {
    cuda.bind_thread()?;
    let reader = open_reader(pack, direct)?;
    let mut pin = cuda.alloc_pinned(shared.span)?;
    let stream = cuda.stream()?;

    loop {
        let job = match rx.lock().unwrap().recv() {
            Ok(j) => j,
            Err(_) => break, // channel closed: pager dropped
        };
        let t0 = Instant::now();
        let idx = shared.idx(job.layer, job.slot);
        let len = reader.read_blob_into(job.layer, job.expert, pin.as_mut())?;
        let aligned = len.div_ceil(ALIGN as usize) * ALIGN as usize;
        cuda.htod_async(shared.dev_slots[idx], &pin.as_ref()[..aligned], stream)?;
        cuda.record_event(shared.events[idx], stream)?;
        // Completing the copy before publishing keeps Ready == data-valid for
        // host-side waiters; the recorded event still serves stream waiters.
        cuda.sync_stream(stream)?;

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let bucket = match ms {
            x if x < 1.0 => 0,
            x if x < 2.0 => 1,
            x if x < 5.0 => 2,
            x if x < 10.0 => 3,
            x if x < 20.0 => 4,
            x if x < 50.0 => 5,
            _ => 6,
        };
        shared.lat_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        shared.bytes_fetched.fetch_add(len as u64, Ordering::Relaxed);
        shared.fetches.fetch_add(1, Ordering::Relaxed);

        let mut st = shared.state.lock().unwrap();
        st.fill[idx] = Fill::Ready;
        st.cache.publish(job.layer, job.slot);
        if job.release_after_fill {
            st.cache.release(job.layer, job.slot);
        }
        drop(st);
        shared.cv.notify_all();
    }
    cuda.free_pinned(&pin);
    Ok(())
}
