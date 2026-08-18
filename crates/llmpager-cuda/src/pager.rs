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
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use llmpager_core::cache::{ExpertCache, Lookup};
use llmpager_core::pack::{PackReader, ALIGN};

use crate::driver::{CUdeviceptr, CUevent, CUstream, Cuda};

/// How long `request` waits on a fully-pinned layer before checking whether
/// any fetch is still in flight to release it. Long enough that a healthy
/// pipeline never reaches it, short enough to turn a hang into an error.
const STALL_GRACE: Duration = Duration::from_secs(10);

pub struct PagerConfig {
    pub slots_per_layer: u32,
    pub io_threads: usize,
    /// Cache frequency counters halve every this many insertions per layer.
    pub decay_interval: u32,
    /// O_DIRECT reads (bypass the OS page cache). True is right when the
    /// pack exceeds host RAM; false lets the page cache act as a RAM tier —
    /// misses cost a memory copy instead of a disk read once warm.
    pub direct: bool,
    /// Managed host-RAM expert tier in bytes (0 disables). For packs far
    /// larger than RAM: frequency-aware admission/eviction at expert
    /// granularity; a hit costs a host memcpy instead of a disk read.
    pub ram_bytes: u64,
    /// VRAM to leave free for other processes on the card, in bytes (0
    /// disables). The expert arena is a *cache*, so it is the right thing
    /// to give up when a co-tenant needs room: `slots_per_layer` is clamped
    /// to whatever still leaves this much free.
    pub reserve_bytes: u64,
}

impl Default for PagerConfig {
    fn default() -> Self {
        Self {
            slots_per_layer: 32,
            io_threads: 4,
            decay_interval: 256,
            direct: true,
            ram_bytes: 0,
            reserve_bytes: 0,
        }
    }
}

/// Host-RAM expert tier: one anonymous NORESERVE mapping sliced into
/// blob-span slabs; bookkeeping is a single-pool ExpertCache whose expert
/// id folds (layer, expert) together.
struct RamTier {
    cache: Mutex<ExpertCache>,
    base: *mut u8,
    bytes: usize,
    span: usize,
    experts_per_layer: u32,
    hits: AtomicU64,
}

// Safety: slab contents are only read after publish under the pin
// discipline; the mapping itself is plain anonymous memory.
unsafe impl Send for RamTier {}
unsafe impl Sync for RamTier {}

impl RamTier {
    fn new(bytes: u64, span: usize, num_layers: u16, experts_per_layer: u16) -> Option<Self> {
        let slots = (bytes as usize / span) as u32;
        // The folded (layer, expert) key must fit the cache's u16 id space.
        let total = num_layers as u32 * experts_per_layer as u32;
        if slots == 0 || total > u16::MAX as u32 {
            return None;
        }
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                slots as usize * span,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return None;
        }
        Some(Self {
            cache: Mutex::new(ExpertCache::new(1, slots, slots.max(64) * 4)),
            base: base as *mut u8,
            bytes: slots as usize * span,
            span,
            experts_per_layer: experts_per_layer as u32,
            hits: AtomicU64::new(0),
        })
    }

    fn key(&self, layer: u16, expert: u16) -> u16 {
        (layer as u32 * self.experts_per_layer + expert as u32) as u16
    }

    fn slab(&self, slot: u32) -> *mut u8 {
        unsafe { self.base.add(slot as usize * self.span) }
    }

    /// Copy the expert into `dst` on hit.
    fn get(&self, layer: u16, expert: u16, dst: &mut [u8]) -> bool {
        let key = self.key(layer, expert);
        let slot = match self.cache.lock().unwrap().lookup_ready(0, key) {
            Some(s) => s,
            None => return false,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(self.slab(slot), dst.as_mut_ptr(), self.span);
        }
        self.cache.lock().unwrap().release(0, slot);
        self.hits.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Write-allocate after a disk read; best-effort.
    fn put(&self, layer: u16, expert: u16, src: &[u8]) {
        let key = self.key(layer, expert);
        let slot = {
            let mut c = self.cache.lock().unwrap();
            match c.acquire(0, key) {
                Lookup::Miss { slot, .. } => slot,
                Lookup::Hit(s) => {
                    c.release(0, s);
                    return;
                }
                Lookup::Stalled => return,
            }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.slab(slot), self.span);
        }
        let mut c = self.cache.lock().unwrap();
        c.publish(0, slot);
        c.release(0, slot);
    }
}

impl Drop for RamTier {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base as *mut _, self.bytes) };
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
    ram: Option<RamTier>,
    /// Per-(layer, expert) fetch counts — the profile that drives pre-warm.
    fetch_counts: Vec<AtomicU64>,
    experts_per_layer: u16,
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
    pack_path: PathBuf,
    direct: bool,
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

        // Clamp the arena to what the card can spare. CUDA gives no
        // cross-process pressure signal, so a co-tenant's allocation simply
        // fails against memory we are holding as cache; leaving a reserve is
        // the only way to be a good neighbour by default.
        let mut cfg = cfg;
        if cfg.reserve_bytes > 0 {
            let (free, total) = cuda.mem_info()?;
            let per_slot = layers as u64 * span as u64;
            let spare = free.saturating_sub(cfg.reserve_bytes);
            let fits = (spare / per_slot.max(1)) as u32;
            if fits < cfg.slots_per_layer {
                if fits == 0 {
                    bail!(
                        "cannot honour a {} MB VRAM reserve: {} MB free of {} MB, and one slot \
                         per layer needs {} MB",
                        cfg.reserve_bytes / 1_000_000,
                        free / 1_000_000,
                        total / 1_000_000,
                        per_slot / 1_000_000
                    );
                }
                eprintln!(
                    "vram reserve: {} slots/layer -> {fits} (leaving {} MB free of {} MB)",
                    cfg.slots_per_layer,
                    cfg.reserve_bytes / 1_000_000,
                    total / 1_000_000
                );
                cfg.slots_per_layer = fits;
            }
        }
        let cfg = cfg;
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
            ram: RamTier::new(cfg.ram_bytes, span, meta.num_layers, meta.experts_per_layer),
            fetch_counts: (0..meta.num_layers as usize * meta.experts_per_layer as usize)
                .map(|_| AtomicU64::new(0))
                .collect(),
            experts_per_layer: meta.experts_per_layer,
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

        Ok(Self {
            cuda,
            shared,
            index,
            pack_path: pack.to_path_buf(),
            direct: cfg.direct,
            tx: Some(tx),
            workers,
            arenas,
        })
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
                        // Every slot in the layer is pinned. The only
                        // releases that can arrive without us doing anything
                        // come from fills still in flight; with none in
                        // flight this wait would never end (a caller holding
                        // handles across a forward pass). Silently hanging
                        // the server is the worst failure mode available, so
                        // wait a grace period and then say what happened.
                        let (guard, res) =
                            self.shared.cv.wait_timeout(st, STALL_GRACE).unwrap();
                        st = guard;
                        if res.timed_out() {
                            let base = self.shared.idx(layer, 0);
                            let n = self.shared.slots_per_layer as usize;
                            let in_flight =
                                st.fill[base..base + n].iter().any(|f| *f == Fill::InFlight);
                            if !in_flight {
                                bail!(
                                    "expert cache deadlock: all {n} slots of layer {layer} are \
                                     pinned with no fetch in flight — expert handles were held \
                                     across a forward pass"
                                );
                            }
                        }
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

    /// Per-(layer, expert) fetch counts, row-major — a workload profile.
    pub fn expert_stats(&self) -> Vec<u64> {
        self.shared.fetch_counts.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// Preload the RAM tier with the highest-count experts from a profile
    /// (row-major layer*epl+expert counts). Reads sequentially by pack
    /// order for disk efficiency; stops when the tier stops accepting.
    /// No-op without a RAM tier.
    pub fn prewarm(&self, counts: &[u64]) -> Result<usize> {
        let Some(ram) = self.shared.ram.as_ref() else {
            return Ok(0);
        };
        let epl = self.shared.experts_per_layer as usize;
        let capacity = (ram.bytes / ram.span) as usize;
        let mut ranked: Vec<(u64, usize)> = counts
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(i, c)| (*c, i))
            .collect();
        ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        ranked.truncate(capacity);
        // Pack-order for near-sequential reads.
        let mut idxs: Vec<usize> = ranked.into_iter().map(|(_, i)| i).collect();
        idxs.sort_unstable();

        let mut reader = open_reader(&self.pack_path, self.direct)?;
        let mut buf = llmpager_core::pack::AlignedBuf::new(self.shared.span);
        let mut loaded = 0usize;
        for i in idxs {
            let (layer, expert) = ((i / epl) as u16, (i % epl) as u16);
            reader.read_blob_into(layer, expert, buf.as_mut())?;
            ram.put(layer, expert, &buf.as_ref()[..self.shared.span]);
            loaded += 1;
        }
        Ok(loaded)
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
            ram_hits: self
                .shared
                .ram
                .as_ref()
                .map_or(0, |r| r.hits.load(Ordering::Relaxed)),
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
    /// Fetches served from the host-RAM tier (subset of `fetches`).
    pub ram_hits: u64,
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
    let mut reader = open_reader(pack, direct)?;
    let mut pin = cuda.alloc_pinned(shared.span)?;
    let stream = cuda.stream()?;

    loop {
        let job = match rx.lock().unwrap().recv() {
            Ok(j) => j,
            Err(_) => break, // channel closed: pager dropped
        };
        let t0 = Instant::now();
        let idx = shared.idx(job.layer, job.slot);
        let span = shared.span;
        let ram_hit = shared
            .ram
            .as_ref()
            .map_or(false, |r| r.get(job.layer, job.expert, &mut pin.as_mut()[..span]));
        let len = if ram_hit {
            reader.entry(job.layer, job.expert).nbytes as usize
        } else {
            let n = reader.read_blob_into(job.layer, job.expert, pin.as_mut())?;
            if let Some(r) = &shared.ram {
                r.put(job.layer, job.expert, &pin.as_ref()[..span]);
            }
            n
        };
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
        shared.fetch_counts
            [job.layer as usize * shared.experts_per_layer as usize + job.expert as usize]
            .fetch_add(1, Ordering::Relaxed);

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
