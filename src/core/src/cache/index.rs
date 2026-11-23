use congee::CongeeArc;
use std::{
    fmt::{Debug, Formatter},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use crate::cache::{
    cached_batch::{CacheEntry, CachedBatchType},
    utils::EntryID,
};
use crate::sync::RwLock;

/// What the index holds for a key: the entry, and the identity of the source
/// it was cached for.
///
/// **An entry is found only under the identity it was inserted with.** Every
/// lookup that serves data goes through [`ArtIndex::get_for`], which treats
/// a slot of any other identity as absent, and every disk object carries the
/// identity in its header, which a read checks.
///
/// A key does not name its source uniquely: callers pack fields into fixed
/// widths, so two sources can compute the same key. The identity is what tells
/// them apart, so a read under any other identity must treat the slot as
/// absent.
#[derive(Debug)]
pub(crate) struct Slot {
    pub(crate) identity: u64,
    pub(crate) entry: CacheEntry,
}

impl Slot {
    pub(crate) fn belongs_to(&self, identity: u64) -> bool {
        self.identity == identity
    }
}

/// The value stored in the tree: a [`Slot`] behind a cell the index empties
/// the moment it gives the slot up.
///
/// `CongeeArc` frees a removed or replaced value through crossbeam-epoch's
/// deferred destruction: it clones the `Arc` and drops the clone only when a
/// later pin collects that epoch's garbage — up to 64 objects per thread wait
/// in a thread-local bag, and the global queue drains 8 bags per 128 pins.
/// With multi-megabyte arrays as values, that kept every evicted entry alive
/// for an unbounded, budget-invisible stretch: a tier reporting at its limit
/// while the process held several times that.
///
/// Taking the slot out of the cell when the index gives it up means the
/// deferred drop reclaims only an empty shell, and the array dies with the
/// last caller-held reference.
struct Cell {
    slot: RwLock<Option<Arc<Slot>>>,
}

impl Cell {
    fn new(slot: Slot) -> Self {
        Self {
            slot: RwLock::new(Some(Arc::new(slot))),
        }
    }

    fn load(&self) -> Option<Arc<Slot>> {
        self.slot.read().unwrap().clone()
    }

    fn take(&self) -> Option<Arc<Slot>> {
        self.slot.write().unwrap().take()
    }
}

pub(crate) struct ArtIndex {
    art: CongeeArc<EntryID, Cell>,
    entry_count: AtomicUsize,
    identity_mismatches: AtomicU64,
    holdings: Holdings,
}

/// Entries and bytes the index holds per form, kept as slots go in and out.
///
/// A slot's entry never changes once installed, so adding it when the tree
/// takes it and subtracting it when the tree gives it up (to exactly one
/// insert or removal) keeps these exact without walking the tree.
#[derive(Default)]
struct Holdings {
    entries: [AtomicUsize; 4],
    bytes: [AtomicUsize; 4],
}

impl Holdings {
    fn form(kind: CachedBatchType) -> usize {
        match kind {
            CachedBatchType::MemoryArrow => 0,
            CachedBatchType::MemoryLiquid => 1,
            CachedBatchType::DiskArrow => 2,
            CachedBatchType::DiskLiquid => 3,
        }
    }

    /// The bytes `entry` counts: its memory, or a disk record's disk share.
    fn bytes_of(entry: &CacheEntry) -> usize {
        match entry {
            CacheEntry::DiskLiquid { disk_bytes, .. }
            | CacheEntry::DiskArrow { disk_bytes, .. } => *disk_bytes,
            CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_) => entry.memory_usage_bytes(),
        }
    }

    fn add(&self, entry: &CacheEntry) {
        let form = Self::form(CachedBatchType::from(entry));
        if let (Some(entries), Some(bytes)) = (self.entries.get(form), self.bytes.get(form)) {
            entries.fetch_add(1, Ordering::Relaxed);
            bytes.fetch_add(Self::bytes_of(entry), Ordering::Relaxed);
        }
    }

    fn remove(&self, entry: &CacheEntry) {
        let form = Self::form(CachedBatchType::from(entry));
        let less = |counter: &AtomicUsize, by: usize| {
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                Some(held.saturating_sub(by))
            });
        };
        if let (Some(entries), Some(bytes)) = (self.entries.get(form), self.bytes.get(form)) {
            less(entries, 1);
            less(bytes, Self::bytes_of(entry));
        }
    }

    fn get(&self, kind: CachedBatchType) -> (usize, usize) {
        let form = Self::form(kind);
        let load = |counters: &[AtomicUsize; 4]| {
            counters
                .get(form)
                .map_or(0, |counter| counter.load(Ordering::Relaxed))
        };
        (load(&self.entries), load(&self.bytes))
    }

    fn clear(&self) {
        for counter in self.entries.iter().chain(&self.bytes) {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

impl Debug for ArtIndex {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArtIndex {
    pub(crate) fn new() -> Self {
        Self {
            art: CongeeArc::new(),
            entry_count: AtomicUsize::new(0),
            identity_mismatches: AtomicU64::new(0),
            holdings: Holdings::default(),
        }
    }

    /// The slot under `entry_id`, whichever identity holds it.
    pub(crate) fn get(&self, entry_id: &EntryID) -> Option<Arc<Slot>> {
        let guard = self.art.pin();
        // An empty cell means the slot was removed or replaced between the
        // tree lookup and the load. A remove reading as a miss is as if it had
        // won the race outright, but after a replace the key is still present
        // under a new cell, so look it up once more rather than report a
        // cached entry as absent.
        let cell = self.art.get(*entry_id, &guard)?;
        if let Some(slot) = cell.load() {
            return Some(slot);
        }
        self.art.get(*entry_id, &guard)?.load()
    }

    /// The slot under `entry_id`, only if `identity` holds it. A slot held by
    /// another identity is counted in [`Self::identity_mismatches`].
    pub(crate) fn get_for(&self, entry_id: &EntryID, identity: u64) -> Option<Arc<Slot>> {
        let slot = self.get(entry_id)?;
        if !slot.belongs_to(identity) {
            self.identity_mismatches.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(slot)
    }

    /// How many lookups found their key held by another identity.
    pub(crate) fn identity_mismatches(&self) -> u64 {
        self.identity_mismatches.load(Ordering::Relaxed)
    }

    pub(crate) fn contains(&self, entry_id: &EntryID, identity: u64) -> bool {
        self.get_for(entry_id, identity).is_some()
    }

    /// Store `batch` under `entry_id` for `identity`, replacing whatever held
    /// the key, including a slot held by another identity.
    ///
    /// Returns the slot this insert replaced. The tree hands a replaced slot
    /// to exactly one insert or removal, so whatever the slot held is the
    /// caller's alone to give back.
    ///
    /// The tree refuses an insert only when it cannot allocate a node; the
    /// batch is then handed back, nothing changed, rather than the process
    /// aborting.
    pub(crate) fn insert(
        &self,
        entry_id: &EntryID,
        identity: u64,
        batch: CacheEntry,
    ) -> Result<Option<Arc<Slot>>, CacheEntry> {
        let guard = self.art.pin();
        self.holdings.add(&batch);
        // A handle on the batch to give back if the tree refuses it; its
        // arrays are shared, not copied.
        let kept = batch.clone();
        let slot = Slot {
            identity,
            entry: batch,
        };
        let Ok(existing) = self
            .art
            .insert(*entry_id, Arc::new(Cell::new(slot)), &guard)
        else {
            self.holdings.remove(&kept);
            return Err(kept);
        };
        let replaced = match existing {
            Some(replaced) => replaced.take(),
            None => {
                self.entry_count.fetch_add(1, Ordering::Relaxed);
                None
            }
        };
        if let Some(replaced) = &replaced {
            self.holdings.remove(&replaced.entry);
        }
        Ok(replaced)
    }

    /// [`Self::insert`], only over a slot `over` holds for: the slot the
    /// caller read before it built `batch`, a transcode, spill or promotion
    /// of what it read.
    ///
    /// The check and the replacement are one step, so a removal or another
    /// write of the key since the caller read it wins and is never undone:
    /// the batch is handed back, nothing changed. That holds for every key,
    /// whatever lock its caller holds.
    pub(crate) fn replace_if(
        &self,
        entry_id: &EntryID,
        identity: u64,
        batch: CacheEntry,
        over: impl Fn(&Slot) -> bool,
    ) -> Result<Option<Arc<Slot>>, CacheEntry> {
        let guard = self.art.pin();
        // Counted before it can be seen, as an insert counts it, so a removal
        // that follows at once never subtracts what was not added.
        self.holdings.add(&batch);
        let mut matched = false;
        let old = self.art.compute_if_present(
            *entry_id,
            |cell| {
                // The tree calls this again when its node changed under it, so
                // the last call decides.
                matched = cell.load().is_some_and(|slot| over(&slot));
                if matched {
                    Some(Arc::new(Cell::new(Slot {
                        identity,
                        entry: batch.clone(),
                    })))
                } else {
                    Some(cell)
                }
            },
            &guard,
        );
        if !matched || old.is_none() {
            self.holdings.remove(&batch);
            return Err(batch);
        }
        let replaced = old.and_then(|old| old.take());
        if let Some(replaced) = &replaced {
            self.holdings.remove(&replaced.entry);
        }
        Ok(replaced)
    }

    pub(crate) fn remove(&self, entry_id: &EntryID) -> Option<Arc<Slot>> {
        let guard = self.art.pin();
        let removed = self.art.remove(*entry_id, &guard)?;
        self.entry_count.fetch_sub(1, Ordering::Relaxed);
        let removed = removed.take();
        if let Some(removed) = &removed {
            self.holdings.remove(&removed.entry);
        }
        removed
    }

    /// Remove the slot under `entry_id` only if `pred` holds for it.
    ///
    /// The check and the removal are one step: a slot installed after `pred`
    /// looked is not removed unseen, since the tree rejects the removal and
    /// `pred` is asked again about the slot now there.
    pub(crate) fn remove_if(
        &self,
        entry_id: &EntryID,
        pred: impl Fn(&Slot) -> bool,
    ) -> Option<Arc<Slot>> {
        let guard = self.art.pin();
        let mut matched = false;
        let removed = self.art.compute_if_present(
            *entry_id,
            |cell| {
                // The tree calls this again when its node changed under it, so
                // the last call decides.
                matched = cell.load().is_some_and(|slot| pred(&slot));
                if matched { None } else { Some(cell) }
            },
            &guard,
        )?;
        if !matched {
            return None;
        }
        self.entry_count.fetch_sub(1, Ordering::Relaxed);
        let removed = removed.take();
        if let Some(removed) = &removed {
            self.holdings.remove(&removed.entry);
        }
        removed
    }

    pub(crate) fn reset(&self) {
        for k in self.art.keys() {
            self.remove(&k);
        }
        self.entry_count.store(0, Ordering::Relaxed);
        self.holdings.clear();
    }

    /// How many entries of `kind` the index holds, and their bytes.
    pub(crate) fn holdings(&self, kind: CachedBatchType) -> (usize, usize) {
        self.holdings.get(kind)
    }

    pub(crate) fn for_each(&self, mut f: impl FnMut(&EntryID, &Slot)) {
        for id in self.art.keys() {
            if let Some(slot) = self.get(&id) {
                f(&id, &slot);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn keys(&self) -> Vec<EntryID> {
        self.art.keys()
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::cached_batch::CacheEntry;
    use crate::cache::utils::create_test_array;

    use super::*;

    #[test]
    fn test_get_and_contains() {
        let store = ArtIndex::new();
        let entry_id1: EntryID = EntryID::from(1);
        let entry_id2: EntryID = EntryID::from(2);
        let array1 = create_test_array(100);

        // Initially, entries should not be cached
        assert!(!store.contains(&entry_id1, 0));
        assert!(!store.contains(&entry_id2, 0));
        assert!(store.get(&entry_id1).is_none());

        // Insert an entry and verify it's cached
        {
            store.insert(&entry_id1, 0, array1.clone()).unwrap();
        }

        assert!(store.contains(&entry_id1, 0));
        assert!(!store.contains(&entry_id2, 0));

        // Get should return the cached value
        match store.get_for(&entry_id1, 0) {
            Some(slot) => match &slot.entry {
                CacheEntry::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
                _ => panic!("Expected ArrowMemory batch"),
            },
            None => panic!("Expected ArrowMemory batch"),
        }
    }

    #[test]
    fn test_reset() {
        let store = ArtIndex::new();
        let entry_id: EntryID = EntryID::from(1);
        let array = create_test_array(100);

        store.insert(&entry_id, 0, array.clone()).unwrap();

        let entry_id: EntryID = EntryID::from(1);
        assert!(store.contains(&entry_id, 0));

        store.reset();
        let entry_id: EntryID = EntryID::from(1);
        assert!(!store.contains(&entry_id, 0));
    }

    /// The array behind a removed or replaced entry must die with the last
    /// caller-held reference, not wait for epoch garbage collection.
    #[test]
    fn removed_and_replaced_entries_are_released_immediately() {
        let store = ArtIndex::new();
        let id = EntryID::from(1);

        let first = create_test_array(100);
        let CacheEntry::MemoryArrow(first_array) = &first else {
            unreachable!()
        };
        let weak_first = Arc::downgrade(first_array);
        store.insert(&id, 0, first).unwrap();
        store.insert(&id, 0, create_test_array(200)).unwrap();
        assert!(
            weak_first.upgrade().is_none(),
            "replaced entry still alive: held by the index's deferred drop"
        );

        let second = store.get(&id).unwrap();
        let removed = store.remove(&id).unwrap();
        let CacheEntry::MemoryArrow(second_array) = &removed.entry else {
            unreachable!()
        };
        let weak_second = Arc::downgrade(second_array);
        drop((second, removed));
        assert!(
            weak_second.upgrade().is_none(),
            "removed entry still alive: held by the index's deferred drop"
        );
        assert_eq!(store.entry_count(), 0);
    }

    #[test]
    fn a_slot_is_visible_only_to_its_identity() {
        let store = ArtIndex::new();
        let entry_id = EntryID::from(1);
        store.insert(&entry_id, 7, create_test_array(10)).unwrap();

        assert!(store.contains(&entry_id, 7));
        assert!(!store.contains(&entry_id, 8));
        assert!(store.get_for(&entry_id, 8).is_none());
        assert_eq!(store.identity_mismatches(), 2);

        store.insert(&entry_id, 8, create_test_array(20)).unwrap();
        assert!(!store.contains(&entry_id, 7));
        assert!(store.contains(&entry_id, 8));
        assert_eq!(store.entry_count(), 1);
    }

    #[test]
    fn remove_if_removes_only_a_matching_slot() {
        let store = ArtIndex::new();
        let entry_id = EntryID::from(1);
        store.insert(&entry_id, 7, create_test_array(10)).unwrap();

        assert!(
            store
                .remove_if(&entry_id, |slot| slot.identity == 8)
                .is_none()
        );
        assert!(store.contains(&entry_id, 7));
        assert_eq!(store.entry_count(), 1);

        let removed = store
            .remove_if(&entry_id, |slot| slot.identity == 7)
            .expect("matching slot is removed");
        assert_eq!(removed.identity, 7);
        assert!(store.get(&entry_id).is_none());
        assert_eq!(store.entry_count(), 0);
        assert!(store.remove_if(&entry_id, |_| true).is_none());
    }

    #[test]
    fn replace_if_never_brings_back_a_removed_slot() {
        let store = ArtIndex::new();
        let entry_id = EntryID::from(1);
        store.insert(&entry_id, 7, create_test_array(10)).unwrap();
        let held = |kind| store.holdings(kind).0;

        // Over a slot of another identity: refused, nothing changed.
        assert!(
            store
                .replace_if(&entry_id, 7, create_test_array(20), |slot| slot.identity
                    == 8)
                .is_err()
        );
        assert_eq!(held(CachedBatchType::MemoryArrow), 1);

        // Over the slot it read: replaced, still one entry.
        let replaced = store
            .replace_if(&entry_id, 7, create_test_array(20), |slot| {
                slot.identity == 7
            })
            .expect("a matching slot is replaced");
        assert_eq!(replaced.map(|slot| slot.identity), Some(7));
        assert_eq!(store.entry_count(), 1);
        assert_eq!(held(CachedBatchType::MemoryArrow), 1);

        // Removed since: refused, and the key stays empty.
        store.remove(&entry_id);
        assert!(
            store
                .replace_if(&entry_id, 7, create_test_array(30), |_| true)
                .is_err()
        );
        assert!(store.get(&entry_id).is_none());
        assert_eq!(store.entry_count(), 0);
        assert_eq!(held(CachedBatchType::MemoryArrow), 0);
    }
}
