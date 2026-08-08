//! Per-layer aged-LFU cache bookkeeping for routed experts.
//!
//! Each transformer layer gets `slots_per_layer` slots. A slot holds one
//! expert's weights (the device buffer itself lives in the GPU layer; here a
//! slot is just an index). Eviction picks the unpinned slot with the lowest
//! frequency counter; counters halve every `decay_interval` insertions so
//! long-dead experts cannot pin the cache forever. Pinning (refcounts) keeps
//! an expert resident while a fetch or a forward pass is using it.

use std::collections::HashMap;

pub type Layer = u16;
pub type Expert = u16;
/// Slot index within one layer's slot array, `0..slots_per_layer`.
pub type Slot = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup {
    /// Expert already cached; slot is pinned for the caller.
    Hit(Slot),
    /// Expert must be fetched into `slot`; previous occupant (if any) was
    /// evicted. Slot is pinned for the caller; call `publish` once filled.
    Miss { slot: Slot, evicted: Option<Expert> },
    /// All slots in the layer are pinned; caller must retry after unpinning.
    Stalled,
}

#[derive(Debug, Clone, Copy)]
struct SlotState {
    expert: Option<Expert>,
    freq: u32,
    pins: u32,
    /// False while a miss is in flight: the slot is claimed (and pinned) but
    /// the expert's weights are not yet valid for other readers.
    ready: bool,
}

struct LayerCache {
    slots: Vec<SlotState>,
    /// expert -> slot, for experts currently resident (ready or in flight).
    map: HashMap<Expert, Slot>,
    inserts_since_decay: u32,
}

pub struct ExpertCache {
    layers: Vec<LayerCache>,
    slots_per_layer: u32,
    decay_interval: u32,
    hits: u64,
    misses: u64,
}

impl ExpertCache {
    pub fn new(num_layers: u16, slots_per_layer: u32, decay_interval: u32) -> Self {
        assert!(slots_per_layer > 0 && decay_interval > 0);
        let layers = (0..num_layers)
            .map(|_| LayerCache {
                slots: vec![
                    SlotState { expert: None, freq: 0, pins: 0, ready: false };
                    slots_per_layer as usize
                ],
                map: HashMap::new(),
                inserts_since_decay: 0,
            })
            .collect();
        Self { layers, slots_per_layer, decay_interval, hits: 0, misses: 0 }
    }

    pub fn slots_per_layer(&self) -> u32 {
        self.slots_per_layer
    }

    /// Look up `expert` in `layer`, pinning the returned slot.
    ///
    /// On `Hit` the slot may still be in flight (`publish` not yet called by
    /// the fetching party); callers coordinate readiness via their own fetch
    /// tracking, keyed by slot. On `Miss` the caller owns the fetch.
    pub fn acquire(&mut self, layer: Layer, expert: Expert) -> Lookup {
        let lc = &mut self.layers[layer as usize];
        if let Some(&slot) = lc.map.get(&expert) {
            let s = &mut lc.slots[slot as usize];
            s.freq = s.freq.saturating_add(1);
            s.pins += 1;
            self.hits += 1;
            return Lookup::Hit(slot);
        }

        // Victim: unpinned slot with lowest (freq, prefer empty).
        let mut victim: Option<Slot> = None;
        let mut victim_key = (u32::MAX, false); // (freq, occupied)
        for (i, s) in lc.slots.iter().enumerate() {
            if s.pins > 0 {
                continue;
            }
            let key = (s.freq, s.expert.is_some());
            if victim.is_none() || key < victim_key {
                victim = Some(i as Slot);
                victim_key = key;
            }
        }
        let Some(slot) = victim else {
            return Lookup::Stalled;
        };

        let evicted = lc.slots[slot as usize].expert;
        if let Some(old) = evicted {
            lc.map.remove(&old);
        }
        lc.map.insert(expert, slot);
        lc.slots[slot as usize] =
            SlotState { expert: Some(expert), freq: 1, pins: 1, ready: false };
        self.misses += 1;

        lc.inserts_since_decay += 1;
        if lc.inserts_since_decay >= self.decay_interval {
            lc.inserts_since_decay = 0;
            for s in &mut lc.slots {
                s.freq /= 2;
            }
        }
        Lookup::Miss { slot, evicted }
    }

    /// Hit-only lookup: pin and return the slot if the expert is resident
    /// AND published; never inserts. Used by read-through tiers (e.g. the
    /// host-RAM expert cache) where a miss is handled out of band.
    pub fn lookup_ready(&mut self, layer: Layer, expert: Expert) -> Option<Slot> {
        let lc = &mut self.layers[layer as usize];
        let &slot = lc.map.get(&expert)?;
        let s = &mut lc.slots[slot as usize];
        if !s.ready {
            return None;
        }
        s.freq = s.freq.saturating_add(1);
        s.pins += 1;
        self.hits += 1;
        Some(slot)
    }

    /// Mark a miss-filled slot as holding valid weights.
    pub fn publish(&mut self, layer: Layer, slot: Slot) {
        self.layers[layer as usize].slots[slot as usize].ready = true;
    }

    pub fn is_ready(&self, layer: Layer, slot: Slot) -> bool {
        self.layers[layer as usize].slots[slot as usize].ready
    }

    /// Release one pin taken by `acquire`.
    pub fn release(&mut self, layer: Layer, slot: Slot) {
        let s = &mut self.layers[layer as usize].slots[slot as usize];
        assert!(s.pins > 0, "release without matching acquire");
        s.pins -= 1;
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_after_miss() {
        let mut c = ExpertCache::new(1, 4, 1000);
        let Lookup::Miss { slot, evicted } = c.acquire(0, 7) else {
            panic!("expected miss");
        };
        assert_eq!(evicted, None);
        c.publish(0, slot);
        c.release(0, slot);
        assert_eq!(c.acquire(0, 7), Lookup::Hit(slot));
        assert!(c.is_ready(0, slot));
        assert_eq!(c.stats(), (1, 1));
    }

    #[test]
    fn evicts_lowest_freq_unpinned() {
        let mut c = ExpertCache::new(1, 2, 1000);
        // Fill both slots; expert 1 gets extra hits.
        for e in [0u16, 1] {
            let Lookup::Miss { slot, .. } = c.acquire(0, e) else { panic!() };
            c.publish(0, slot);
            c.release(0, slot);
        }
        for _ in 0..3 {
            let Lookup::Hit(s) = c.acquire(0, 1) else { panic!() };
            c.release(0, s);
        }
        // Expert 2 must evict expert 0 (freq 1) not expert 1 (freq 4).
        let Lookup::Miss { evicted, .. } = c.acquire(0, 2) else { panic!() };
        assert_eq!(evicted, Some(0));
    }

    #[test]
    fn pinned_slots_are_not_victims() {
        let mut c = ExpertCache::new(1, 1, 1000);
        let Lookup::Miss { slot, .. } = c.acquire(0, 0) else { panic!() };
        // Slot still pinned: any other expert stalls.
        assert_eq!(c.acquire(0, 1), Lookup::Stalled);
        c.release(0, slot);
        let Lookup::Miss { evicted, .. } = c.acquire(0, 1) else { panic!() };
        assert_eq!(evicted, Some(0));
    }

    #[test]
    fn decay_halves_frequencies() {
        let mut c = ExpertCache::new(1, 2, 2);
        let Lookup::Miss { slot, .. } = c.acquire(0, 0) else { panic!() };
        c.release(0, slot);
        for _ in 0..7 {
            let Lookup::Hit(s) = c.acquire(0, 0) else { panic!() };
            c.release(0, s);
        }
        // freq(expert 0) is now 8. Two inserts trigger one decay (interval 2).
        let Lookup::Miss { slot, .. } = c.acquire(0, 1) else { panic!() };
        c.release(0, slot);
        let Lookup::Miss { slot, .. } = c.acquire(0, 2) else { panic!() };
        c.release(0, slot);
        // After decay, expert 0's freq is 4; still the hottest, so a further
        // insert evicts one of the newcomers, not expert 0.
        let Lookup::Miss { evicted, .. } = c.acquire(0, 3) else { panic!() };
        assert_ne!(evicted, Some(0));
    }

    #[test]
    fn lookup_ready_never_inserts() {
        let mut c = ExpertCache::new(1, 2, 1000);
        assert_eq!(c.lookup_ready(0, 3), None);
        let Lookup::Miss { slot, .. } = c.acquire(0, 3) else { panic!() };
        // In flight: not ready yet.
        assert_eq!(c.lookup_ready(0, 3), None);
        c.publish(0, slot);
        c.release(0, slot);
        let got = c.lookup_ready(0, 3).expect("ready hit");
        assert_eq!(got, slot);
        c.release(0, got);
    }

    #[test]
    fn layers_are_independent() {
        let mut c = ExpertCache::new(2, 1, 1000);
        let Lookup::Miss { slot: s0, .. } = c.acquire(0, 5) else { panic!() };
        let Lookup::Miss { slot: s1, .. } = c.acquire(1, 5) else { panic!() };
        c.release(0, s0);
        c.release(1, s1);
        let Lookup::Hit(_) = c.acquire(0, 5) else { panic!() };
        let Lookup::Hit(_) = c.acquire(1, 5) else { panic!() };
    }
}
