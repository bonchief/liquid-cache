//! Where the store's objects lie in its file, and which of them reads and
//! writes still hold.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::cache::utils::EntryID;

/// The page every extent starts on and is sized in. `O_DIRECT` needs both
/// offsets and lengths aligned to the device's logical block, which a page
/// covers.
pub(super) const PAGE: u64 = 4096;

/// The store's map from entries to extents, and the file's free space.
///
/// Its methods keep these invariants, and nothing else changes its fields:
///
/// - **Bounded.** Every extent ends at or below `capacity`, so the file is
///   never written past it.
/// - **Disjoint.** No two extents overlap, and no extent overlaps free space.
///   A freed extent merges with free neighbours, and free space that reaches
///   `tail` folds into it.
/// - **No reuse under a read.** An extent is never handed out while a read or
///   a write pins it: a removed or replaced extent returns to free space only
///   when its last pin goes. A put of another key therefore never lands under
///   a read in flight.
#[derive(Debug)]
pub(super) struct Extents {
    /// The extent each entry's current object lies in, by offset.
    named: HashMap<EntryID, u64>,
    /// Every extent not free: named ones, and ones a pin still holds.
    held: HashMap<u64, Extent>,
    free: FreeSpace,
    /// Everything from here to `capacity` is free.
    tail: u64,
    capacity: u64,
    /// The bytes `held` covers.
    used: u64,
}

#[derive(Debug)]
struct Extent {
    len: u64,
    /// Reads and writes in flight over the extent.
    pins: u64,
    /// Whether an entry names it in `named`.
    named: bool,
}

impl Extents {
    /// No extents, and `capacity` bytes free, rounded down to whole pages.
    pub(super) fn new(capacity: u64) -> Self {
        Self {
            named: HashMap::new(),
            held: HashMap::new(),
            free: FreeSpace::default(),
            tail: 0,
            capacity: capacity - capacity % PAGE,
            used: 0,
        }
    }

    pub(super) fn capacity(&self) -> u64 {
        self.capacity
    }

    pub(super) fn used(&self) -> u64 {
        self.used
    }

    pub(super) fn len(&self) -> usize {
        self.named.len()
    }

    /// A fresh extent of `len` bytes, a whole number of pages, unnamed and
    /// pinned once for the write that fills it. `None` when no free span
    /// below the capacity is that long.
    pub(super) fn allocate(&mut self, len: u64) -> Option<u64> {
        if len == 0 || len % PAGE != 0 {
            return None;
        }
        let at = match self.free.take(len) {
            Some(at) => at,
            None => {
                let end = self
                    .tail
                    .checked_add(len)
                    .filter(|end| *end <= self.capacity)?;
                let at = self.tail;
                self.tail = end;
                at
            }
        };
        self.held.insert(
            at,
            Extent {
                len,
                pins: 1,
                named: false,
            },
        );
        self.used += len;
        Some(at)
    }

    /// Name the extent at `at`, which a pin holds, as `entry`'s object. The
    /// extent `entry` named before is unnamed, and freed once unpinned.
    pub(super) fn name(&mut self, entry: EntryID, at: u64) {
        let Some(extent) = self.held.get_mut(&at) else {
            return;
        };
        extent.named = true;
        if let Some(old) = self.named.insert(entry, at)
            && old != at
        {
            self.unname(old);
        }
    }

    /// Forget `entry`'s object; whether it had one.
    pub(super) fn remove(&mut self, entry: &EntryID) -> bool {
        match self.named.remove(entry) {
            Some(at) => {
                self.unname(at);
                true
            }
            None => false,
        }
    }

    /// Pin the extent `entry` names, for a read: its offset and length.
    pub(super) fn pin_named(&mut self, entry: &EntryID) -> Option<(u64, u64)> {
        let at = *self.named.get(entry)?;
        let extent = self.held.get_mut(&at)?;
        extent.pins += 1;
        Some((at, extent.len))
    }

    /// Pin the extent at `at` once more. Only called while another pin holds
    /// it, so it is still there.
    pub(super) fn pin(&mut self, at: u64) {
        if let Some(extent) = self.held.get_mut(&at) {
            extent.pins += 1;
        }
    }

    /// Give one pin on `at` back; the extent is freed if that was its last
    /// and no entry names it.
    pub(super) fn unpin(&mut self, at: u64) {
        let Some(extent) = self.held.get_mut(&at) else {
            return;
        };
        extent.pins = extent.pins.saturating_sub(1);
        if extent.pins == 0 && !extent.named {
            self.release(at);
        }
    }

    fn unname(&mut self, at: u64) {
        let Some(extent) = self.held.get_mut(&at) else {
            return;
        };
        extent.named = false;
        if extent.pins == 0 {
            self.release(at);
        }
    }

    fn release(&mut self, at: u64) {
        if let Some(extent) = self.held.remove(&at) {
            self.used -= extent.len;
            self.free.release(at, extent.len, &mut self.tail);
        }
    }
}

/// Free spans below the tail, by offset and by length, for a best fit.
#[derive(Debug, Default)]
struct FreeSpace {
    by_offset: BTreeMap<u64, u64>,
    by_len: BTreeSet<(u64, u64)>,
}

impl FreeSpace {
    /// The offset of the shortest free span at least `len` long, of which
    /// the first `len` bytes are taken and the rest stays free.
    fn take(&mut self, len: u64) -> Option<u64> {
        let (have, at) = *self.by_len.range((len, 0)..).next()?;
        self.remove(at, have);
        if have > len {
            self.insert(at + len, have - len);
        }
        Some(at)
    }

    /// Free `len` bytes at `at`, merged with the free spans on either side;
    /// a span that then ends at `tail` moves the tail down instead.
    fn release(&mut self, mut at: u64, mut len: u64, tail: &mut u64) {
        if let Some((&before, &size)) = self.by_offset.range(..at).next_back()
            && before + size == at
        {
            self.remove(before, size);
            at = before;
            len += size;
        }
        if let Some(&size) = self.by_offset.get(&(at + len)) {
            self.remove(at + len, size);
            len += size;
        }
        if at + len == *tail {
            *tail = at;
        } else {
            self.insert(at, len);
        }
    }

    fn insert(&mut self, at: u64, len: u64) {
        self.by_offset.insert(at, len);
        self.by_len.insert((len, at));
    }

    fn remove(&mut self, at: u64, len: u64) {
        self.by_offset.remove(&at);
        self.by_len.remove(&(len, at));
    }
}
