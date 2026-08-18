//! Aged-LFU cache bookkeeping for routed experts, keyed by a flat expert id.
//!
//! Experts are identified by a single `ExpertId`; a `(layer, expert)` pair
//! folds into one as `layer * ids_per_partition + expert`. Slots are drawn
//! from *partitions*: with one partition per layer the cache behaves exactly
//! like the per-layer arrays it replaced, and with a single partition every
//! layer competes for one global pool, so layers that route to many experts
//! naturally claim more slots than layers that route to few.
//!
//! A slot holds one expert's weights (the device buffer itself lives in the
//! GPU layer; here a slot is just an index, global across partitions).
//! Eviction picks the unpinned slot in the same partition with the lowest
//! frequency counter; counters halve every `decay_interval` insertions so
//! long-dead experts cannot pin the cache forever. Pinning (refcounts) keeps
//! an expert resident while a fetch or a forward pass is using it.

use std::collections::HashMap;

pub type Layer = u16;
pub type Expert = u16;
/// Flat expert identifier: `layer * ids_per_partition + expert` for packs
/// with a layer structure, or just an index for a flat expert population.
pub type ExpertId = u32;
/// Slot index, global across every partition.
pub type Slot = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup {
    /// Expert already cached; slot is pinned for the caller.
    Hit(Slot),
    /// Expert must be fetched into `slot`; previous occupant (if any) was
    /// evicted. Slot is pinned for the caller; call `publish` once filled.
    Miss { slot: Slot, evicted: Option<ExpertId> },
    /// Every slot the id could use is pinned; caller must retry after
    /// unpinning.
    Stalled,
}

#[derive(Debug, Clone, Copy)]
struct SlotState {
    id: Option<ExpertId>,
    freq: u32,
    pins: u32,
    /// False while a miss is in flight: the slot is claimed (and pinned) but
    /// the expert's weights are not yet valid for other readers.
    ready: bool,
}

struct Partition {
    /// Global slot indices this partition draws from: `base .. base + len`.
    base: u32,
    len: u32,
    /// id -> global slot, for experts currently resident (ready or in flight).
    map: HashMap<ExpertId, Slot>,
    inserts_since_decay: u32,
}

pub struct ExpertCache {
    slots: Vec<SlotState>,
    parts: Vec<Partition>,
    /// Ids per partition; also the fold factor for `(layer, expert)`.
    ids_per_partition: u32,
    decay_interval: u32,
    hits: u64,
    misses: u64,
}

impl ExpertCache {
    /// One partition per layer, `slots_per_layer` slots each — the layout
    /// used by packs with a layer structure.
    pub fn new(num_layers: u16, slots_per_layer: u32, decay_interval: u32) -> Self {
        Self::partitioned(num_layers as u32, u32::MAX, slots_per_layer, decay_interval)
    }

    /// `partitions` slot pools of `slots_per_partition` each, with ids folded
    /// as `id / ids_per_partition` to choose a pool. `partitions == 1` gives
    /// one global pool shared by every layer.
    pub fn partitioned(
        partitions: u32,
        ids_per_partition: u32,
        slots_per_partition: u32,
        decay_interval: u32,
    ) -> Self {
        assert!(partitions > 0 && slots_per_partition > 0 && decay_interval > 0);
        let total = partitions as usize * slots_per_partition as usize;
        let parts = (0..partitions)
            .map(|p| Partition {
                base: p * slots_per_partition,
                len: slots_per_partition,
                map: HashMap::new(),
                inserts_since_decay: 0,
            })
            .collect();
        Self {
            slots: vec![SlotState { id: None, freq: 0, pins: 0, ready: false }; total],
            parts,
            ids_per_partition,
            decay_interval,
            hits: 0,
            misses: 0,
        }
    }

    /// Fold a `(layer, expert)` pair into a flat id. `experts_per_layer` must
    /// match the `ids_per_partition` the cache was built with.
    pub fn fold(layer: Layer, expert: Expert, experts_per_layer: u32) -> ExpertId {
        layer as u32 * experts_per_layer + expert as u32
    }

    /// Slots available to a single partition. With one partition per layer
    /// this is the per-layer cache size.
    pub fn slots_per_partition(&self) -> u32 {
        self.parts[0].len
    }

    pub fn total_slots(&self) -> u32 {
        self.slots.len() as u32
    }

    fn part_of(&self, id: ExpertId) -> usize {
        if self.parts.len() == 1 {
            return 0;
        }
        ((id / self.ids_per_partition) as usize).min(self.parts.len() - 1)
    }

    /// Look up `id`, pinning the returned slot.
    ///
    /// On `Hit` the slot may still be in flight (`publish` not yet called by
    /// the fetching party); callers coordinate readiness via their own fetch
    /// tracking, keyed by slot. On `Miss` the caller owns the fetch.
    pub fn acquire(&mut self, id: ExpertId) -> Lookup {
        let p = self.part_of(id);
        if let Some(&slot) = self.parts[p].map.get(&id) {
            let s = &mut self.slots[slot as usize];
            s.freq = s.freq.saturating_add(1);
            s.pins += 1;
            self.hits += 1;
            return Lookup::Hit(slot);
        }

        // Victim: unpinned slot in this partition with lowest (freq, prefer
        // empty).
        let (base, len) = (self.parts[p].base, self.parts[p].len);
        let mut victim: Option<Slot> = None;
        let mut victim_key = (u32::MAX, false); // (freq, occupied)
        for i in base..base + len {
            let s = &self.slots[i as usize];
            if s.pins > 0 {
                continue;
            }
            let key = (s.freq, s.id.is_some());
            if victim.is_none() || key < victim_key {
                victim = Some(i);
                victim_key = key;
            }
        }
        let Some(slot) = victim else {
            return Lookup::Stalled;
        };

        let evicted = self.slots[slot as usize].id;
        if let Some(old) = evicted {
            self.parts[p].map.remove(&old);
        }
        self.parts[p].map.insert(id, slot);
        self.slots[slot as usize] =
            SlotState { id: Some(id), freq: 1, pins: 1, ready: false };
        self.misses += 1;

        self.parts[p].inserts_since_decay += 1;
        if self.parts[p].inserts_since_decay >= self.decay_interval {
            self.parts[p].inserts_since_decay = 0;
            for i in base..base + len {
                self.slots[i as usize].freq /= 2;
            }
        }
        Lookup::Miss { slot, evicted }
    }

    /// Hit-only lookup: pin and return the slot if the expert is resident
    /// AND published; never inserts. Used by read-through tiers (e.g. the
    /// host-RAM expert cache) where a miss is handled out of band.
    pub fn lookup_ready(&mut self, id: ExpertId) -> Option<Slot> {
        let p = self.part_of(id);
        let &slot = self.parts[p].map.get(&id)?;
        let s = &mut self.slots[slot as usize];
        if !s.ready {
            return None;
        }
        s.freq = s.freq.saturating_add(1);
        s.pins += 1;
        self.hits += 1;
        Some(slot)
    }

    /// Mark a miss-filled slot as holding valid weights.
    pub fn publish(&mut self, slot: Slot) {
        self.slots[slot as usize].ready = true;
    }

    pub fn is_ready(&self, slot: Slot) -> bool {
        self.slots[slot as usize].ready
    }

    /// Release one pin taken by `acquire`.
    pub fn release(&mut self, slot: Slot) {
        let s = &mut self.slots[slot as usize];
        assert!(s.pins > 0, "release without matching acquire");
        s.pins -= 1;
    }

    /// True when every slot the id could use is pinned *and* none of them is
    /// mid-fetch — nothing will free itself, so a waiter would wait forever.
    pub fn is_wedged(&self, id: ExpertId) -> bool {
        let p = self.part_of(id);
        let (base, len) = (self.parts[p].base, self.parts[p].len);
        (base..base + len).all(|i| {
            let s = &self.slots[i as usize];
            s.pins > 0 && s.ready
        })
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
        let Lookup::Miss { slot, evicted } = c.acquire(7) else {
            panic!("expected miss");
        };
        assert_eq!(evicted, None);
        c.publish(slot);
        c.release(slot);
        assert_eq!(c.acquire(7), Lookup::Hit(slot));
        assert!(c.is_ready(slot));
        assert_eq!(c.stats(), (1, 1));
    }

    #[test]
    fn evicts_lowest_freq_unpinned() {
        let mut c = ExpertCache::new(1, 2, 1000);
        // Fill both slots; expert 1 gets extra hits.
        for e in [0u32, 1] {
            let Lookup::Miss { slot, .. } = c.acquire(e) else { panic!() };
            c.publish(slot);
            c.release(slot);
        }
        for _ in 0..3 {
            let Lookup::Hit(s) = c.acquire(1) else { panic!() };
            c.release(s);
        }
        // Expert 2 must evict expert 0 (freq 1) not expert 1 (freq 4).
        let Lookup::Miss { evicted, .. } = c.acquire(2) else { panic!() };
        assert_eq!(evicted, Some(0));
    }

    #[test]
    fn pinned_slots_are_not_victims() {
        let mut c = ExpertCache::new(1, 1, 1000);
        let Lookup::Miss { slot, .. } = c.acquire(0) else { panic!() };
        // Slot still pinned: any other expert stalls.
        assert_eq!(c.acquire(1), Lookup::Stalled);
        c.release(slot);
        let Lookup::Miss { evicted, .. } = c.acquire(1) else { panic!() };
        assert_eq!(evicted, Some(0));
    }

    #[test]
    fn decay_halves_frequencies() {
        let mut c = ExpertCache::new(1, 2, 2);
        let Lookup::Miss { slot, .. } = c.acquire(0) else { panic!() };
        c.release(slot);
        for _ in 0..7 {
            let Lookup::Hit(s) = c.acquire(0) else { panic!() };
            c.release(s);
        }
        // freq(expert 0) is now 8. Two inserts trigger one decay (interval 2).
        let Lookup::Miss { slot, .. } = c.acquire(1) else { panic!() };
        c.release(slot);
        let Lookup::Miss { slot, .. } = c.acquire(2) else { panic!() };
        c.release(slot);
        // After decay, expert 0's freq is 4; still the hottest, so a further
        // insert evicts one of the newcomers, not expert 0.
        let Lookup::Miss { evicted, .. } = c.acquire(3) else { panic!() };
        assert_ne!(evicted, Some(0));
    }

    #[test]
    fn lookup_ready_never_inserts() {
        let mut c = ExpertCache::new(1, 2, 1000);
        assert_eq!(c.lookup_ready(3), None);
        let Lookup::Miss { slot, .. } = c.acquire(3) else { panic!() };
        // In flight: not ready yet.
        assert_eq!(c.lookup_ready(3), None);
        c.publish(slot);
        c.release(slot);
        let got = c.lookup_ready(3).expect("ready hit");
        assert_eq!(got, slot);
        c.release(got);
    }

    #[test]
    fn layers_are_independent() {
        // Two partitions of one slot each, 128 ids per layer.
        let mut c = ExpertCache::partitioned(2, 128, 1, 1000);
        let a = ExpertCache::fold(0, 5, 128);
        let b = ExpertCache::fold(1, 5, 128);
        let Lookup::Miss { slot: s0, .. } = c.acquire(a) else { panic!() };
        let Lookup::Miss { slot: s1, .. } = c.acquire(b) else { panic!() };
        assert_ne!(s0, s1, "different layers must not share a slot");
        c.release(s0);
        c.release(s1);
        let Lookup::Hit(_) = c.acquire(a) else { panic!() };
        let Lookup::Hit(_) = c.acquire(b) else { panic!() };
    }

    #[test]
    fn fold_round_trips_layer_and_expert() {
        assert_eq!(ExpertCache::fold(0, 0, 128), 0);
        assert_eq!(ExpertCache::fold(0, 127, 128), 127);
        assert_eq!(ExpertCache::fold(1, 0, 128), 128);
        assert_eq!(ExpertCache::fold(47, 127, 128), 47 * 128 + 127);
    }

    #[test]
    fn one_partition_lets_a_hot_layer_take_more_slots() {
        // One global pool of 4 slots shared by two layers of 128 experts.
        let mut c = ExpertCache::partitioned(1, 128, 4, 1000);
        // Layer 0 routes to three experts, repeatedly; layer 1 to one.
        for _ in 0..3 {
            for e in 0..3u16 {
                let id = ExpertCache::fold(0, e, 128);
                let slot = match c.acquire(id) {
                    Lookup::Hit(s) => s,
                    Lookup::Miss { slot, .. } => {
                        c.publish(slot);
                        slot
                    }
                    Lookup::Stalled => panic!("stalled"),
                };
                c.release(slot);
            }
        }
        let id1 = ExpertCache::fold(1, 0, 128);
        let Lookup::Miss { slot, .. } = c.acquire(id1) else { panic!() };
        c.publish(slot);
        c.release(slot);
        // All four are resident together: layer 0 holds three of the shared
        // slots, which per-layer partitioning could never allow.
        for e in 0..3u16 {
            let id = ExpertCache::fold(0, e, 128);
            assert!(matches!(c.acquire(id), Lookup::Hit(_)), "layer 0 expert {e} evicted");
        }
        assert!(matches!(c.acquire(id1), Lookup::Hit(_)));
    }

    #[test]
    fn wedged_only_when_everything_is_pinned_and_ready() {
        let mut c = ExpertCache::new(1, 2, 1000);
        assert!(!c.is_wedged(0));
        let Lookup::Miss { slot: a, .. } = c.acquire(0) else { panic!() };
        let Lookup::Miss { slot: b, .. } = c.acquire(1) else { panic!() };
        // Both pinned but still in flight: a fetch will release them.
        assert!(!c.is_wedged(0));
        c.publish(a);
        c.publish(b);
        assert!(c.is_wedged(0));
        c.release(a);
        assert!(!c.is_wedged(0));
    }
}
