use std::collections::{HashMap, VecDeque};

use crate::{
    cache::{CachePolicy, EntryID, cached_batch::CachedBatchType},
    sync::Mutex,
};

use super::ChunkCounts;

/// The queue a chunk waits in. Memory chunks holding any Arrow batch go
/// first, being the cheapest to rebuild, then memory chunks holding only
/// Liquid ones; chunks on disk wait in their own queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueueKind {
    Arrow,
    Liquid,
    Disk,
}

impl QueueKind {
    const MEMORY: [Self; 2] = [Self::Arrow, Self::Liquid];

    fn index(self) -> usize {
        match self {
            Self::Arrow => 0,
            Self::Liquid => 1,
            Self::Disk => 2,
        }
    }

    fn tier(self) -> Tier {
        match self {
            Self::Arrow | Self::Liquid => Tier::Memory,
            Self::Disk => Tier::Disk,
        }
    }
}

/// Where a batch is held. A chunk has one record per tier, so a batch read
/// from disk and promoted arrives in memory cold: a disk hit never marks the
/// memory record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Tier {
    Memory,
    Disk,
}

impl Tier {
    fn of(kind: CachedBatchType) -> Self {
        match kind {
            CachedBatchType::MemoryArrow | CachedBatchType::MemoryLiquid => Self::Memory,
            CachedBatchType::DiskArrow | CachedBatchType::DiskLiquid => Self::Disk,
        }
    }

    fn other(self) -> Self {
        match self {
            Self::Memory => Self::Disk,
            Self::Disk => Self::Memory,
        }
    }
}

/// One chunk's batches in one tier.
#[derive(Debug)]
struct ChunkRecord {
    queue: QueueKind,
    /// Which of the queue's tokens for this chunk is live; the others are
    /// left from moves and second chances, and skipped.
    stamp: u64,
    /// The CLOCK reference bit: set by a hit on any of the chunk's batches,
    /// cleared when a victim search passes over the chunk once.
    referenced: bool,
    /// The chunk's batches in this tier, each with whether it is Arrow.
    batches: Vec<(EntryID, bool)>,
}

impl ChunkRecord {
    fn queue_for(tier: Tier, batches: &[(EntryID, bool)]) -> QueueKind {
        match tier {
            Tier::Disk => QueueKind::Disk,
            Tier::Memory if batches.iter().any(|(_, arrow)| *arrow) => QueueKind::Arrow,
            Tier::Memory => QueueKind::Liquid,
        }
    }
}

/// What removing one batch from a chunk record left.
enum Removal {
    Absent,
    Emptied(QueueKind),
    Moved(QueueKind),
    Kept,
}

/// The policy's state: every chunk record, and a queue of tokens per
/// [`QueueKind`] naming the records in the order a CLOCK hand visits them.
///
/// Its methods keep these invariants:
///
/// - **One record per chunk and tier.** A batch is in at most one record:
///   filing it in one tier takes it out of the other.
/// - **One live token per record**, in the queue its kind names, carrying
///   the record's stamp. A token whose record is gone, moved or given a
///   second chance is stale and skipped; stale tokens are dropped when a
///   queue holds more than twice its live records.
/// - **`counts` is the number of records per queue.**
#[derive(Debug)]
struct ChunkClock {
    /// Clears an entry id's batch bits, leaving its chunk.
    chunk_mask: usize,
    records: HashMap<(EntryID, Tier), ChunkRecord>,
    queues: [VecDeque<(EntryID, u64)>; 3],
    counts: [usize; 3],
    next_stamp: u64,
}

impl Default for ChunkClock {
    fn default() -> Self {
        Self {
            // Until told otherwise, every entry is a chunk of its own.
            chunk_mask: usize::MAX,
            records: HashMap::new(),
            queues: Default::default(),
            counts: [0; 3],
            next_stamp: 0,
        }
    }
}

impl ChunkClock {
    /// Stale tokens a queue may hold beyond twice its live ones before they
    /// are swept.
    const SLACK: usize = 1024;

    fn chunk_of(&self, entry: &EntryID) -> EntryID {
        EntryID::from(usize::from(*entry) & self.chunk_mask)
    }

    fn stamp(&mut self) -> u64 {
        let stamp = self.next_stamp;
        self.next_stamp = stamp.wrapping_add(1);
        stamp
    }

    /// Put a token for `chunk` at the back of `queue`, and sweep the queue's
    /// stale tokens once they outnumber its live ones by far.
    fn push(&mut self, queue: QueueKind, chunk: EntryID, stamp: u64) {
        let tier = queue.tier();
        let live = self.counts[queue.index()];
        let tokens = &mut self.queues[queue.index()];
        tokens.push_back((chunk, stamp));
        if tokens.len() > live.saturating_mul(2).saturating_add(Self::SLACK) {
            let records = &self.records;
            tokens.retain(|(chunk, stamp)| {
                records
                    .get(&(*chunk, tier))
                    .is_some_and(|record| record.stamp == *stamp && record.queue == queue)
            });
        }
    }

    /// Move the record under `key` to the back of `queue`.
    fn move_to(&mut self, key: (EntryID, Tier), queue: QueueKind) {
        let stamp = self.stamp();
        let Some(record) = self.records.get_mut(&key) else {
            return;
        };
        let old = std::mem::replace(&mut record.queue, queue);
        record.stamp = stamp;
        self.counts[old.index()] = self.counts[old.index()].saturating_sub(1);
        self.counts[queue.index()] += 1;
        self.push(queue, key.0, stamp);
    }

    fn insert(&mut self, entry: EntryID, kind: CachedBatchType) {
        let tier = Tier::of(kind);
        let arrow = kind == CachedBatchType::MemoryArrow;
        let chunk = self.chunk_of(&entry);
        self.remove_from(chunk, tier.other(), &entry);
        let key = (chunk, tier);
        let moved = match self.records.get_mut(&key) {
            Some(record) => {
                match record.batches.iter_mut().find(|(id, _)| *id == entry) {
                    Some(batch) => batch.1 = arrow,
                    None => record.batches.push((entry, arrow)),
                }
                let queue = ChunkRecord::queue_for(tier, &record.batches);
                (queue != record.queue).then_some(queue)
            }
            None => {
                let queue = ChunkRecord::queue_for(tier, &[(entry, arrow)]);
                let stamp = self.stamp();
                self.records.insert(
                    key,
                    ChunkRecord {
                        queue,
                        stamp,
                        // A chunk just filled into memory has its chance
                        // before the hand takes it: one already read again
                        // since its fill is no likelier to be read next.
                        referenced: tier == Tier::Memory,
                        batches: vec![(entry, arrow)],
                    },
                );
                self.counts[queue.index()] += 1;
                self.push(queue, chunk, stamp);
                None
            }
        };
        if let Some(queue) = moved {
            self.move_to(key, queue);
        }
    }

    /// Take `entry` out of its chunk's record in `tier`.
    fn remove_from(&mut self, chunk: EntryID, tier: Tier, entry: &EntryID) {
        let key = (chunk, tier);
        let removal = match self.records.get_mut(&key) {
            None => Removal::Absent,
            Some(record) => {
                let before = record.batches.len();
                record.batches.retain(|(id, _)| id != entry);
                if record.batches.len() == before {
                    Removal::Absent
                } else if record.batches.is_empty() {
                    Removal::Emptied(record.queue)
                } else {
                    let queue = ChunkRecord::queue_for(tier, &record.batches);
                    if queue == record.queue {
                        Removal::Kept
                    } else {
                        Removal::Moved(queue)
                    }
                }
            }
        };
        match removal {
            Removal::Absent | Removal::Kept => {}
            Removal::Emptied(queue) => {
                self.records.remove(&key);
                self.counts[queue.index()] = self.counts[queue.index()].saturating_sub(1);
            }
            Removal::Moved(queue) => self.move_to(key, queue),
        }
    }

    fn remove(&mut self, entry: &EntryID) {
        let chunk = self.chunk_of(entry);
        self.remove_from(chunk, Tier::Memory, entry);
        self.remove_from(chunk, Tier::Disk, entry);
    }

    fn access(&mut self, entry: &EntryID, kind: CachedBatchType) {
        let key = (self.chunk_of(entry), Tier::of(kind));
        if let Some(record) = self.records.get_mut(&key)
            && record.batches.iter().any(|(id, _)| id == entry)
        {
            record.referenced = true;
        }
    }

    /// Whole chunks from `queues`, in that order, until they hold at least
    /// `want` batches. Each chunk's record leaves the policy with its
    /// batches.
    ///
    /// CLOCK: a referenced chunk at the hand is passed over once, its bit
    /// cleared and its token moved to the back. A chunk read once is thus
    /// evicted before any chunk read again since it was filed, whatever its
    /// age. Each pass clears a bit, so the search ends.
    fn victims(&mut self, queues: &[QueueKind], want: usize) -> Vec<EntryID> {
        let mut victims = Vec::new();
        for &queue in queues {
            while victims.len() < want {
                let Some((chunk, stamp)) = self.queues[queue.index()].pop_front() else {
                    break;
                };
                let key = (chunk, queue.tier());
                let fresh = self.stamp();
                let Some(record) = self.records.get_mut(&key) else {
                    continue;
                };
                if record.stamp != stamp || record.queue != queue {
                    continue;
                }
                if record.referenced {
                    record.referenced = false;
                    record.stamp = fresh;
                    self.queues[queue.index()].push_back((chunk, fresh));
                    continue;
                }
                if let Some(record) = self.records.remove(&key) {
                    self.counts[queue.index()] = self.counts[queue.index()].saturating_sub(1);
                    victims.extend(record.batches.into_iter().map(|(id, _)| id));
                }
            }
        }
        victims
    }

    fn counts(&self) -> ChunkCounts {
        ChunkCounts {
            memory_arrow: self.counts[QueueKind::Arrow.index()],
            memory_liquid: self.counts[QueueKind::Liquid.index()],
            disk: self.counts[QueueKind::Disk.index()],
        }
    }
}

/// Cache policy that evicts whole column chunks by CLOCK, memory chunks
/// holding Arrow first, then Liquid ones, and disk chunks in their own
/// queue.
///
/// **A chunk is evicted whole or not at all.** The policy files batches by
/// their chunk (the entry id with the batch bits the cache names in
/// [`CachePolicy::group_by_chunk`] cleared), and a victim search hands out
/// every batch a chunk holds in the tier searched, never one alone: a hole
/// in a chunk would leave the rest of it held but unreadable as a whole.
/// The cache moves those batches to disk, or drops them, together.
///
/// A hit on any batch sets its chunk's reference bit in the tier it was
/// served from; the search passes over a referenced chunk once. **A chunk
/// filed in memory starts referenced**, so the hand passes a chunk just
/// filled once before it can take it, as it would one read since: a chunk
/// just filled is taken only after every chunk filed before it in its
/// queue and not read since. A chunk filed on disk starts cold.
#[derive(Debug, Default)]
pub struct LiquidPolicy {
    inner: Mutex<ChunkClock>,
}

impl LiquidPolicy {
    /// Create a new [`LiquidPolicy`]. Every entry is its own chunk until the
    /// cache groups them ([`CachePolicy::group_by_chunk`]).
    pub fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> crate::sync::MutexGuard<'_, ChunkClock> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl CachePolicy for LiquidPolicy {
    fn group_by_chunk(&self, batch_bits: u32) {
        let mut state = self.state();
        state.chunk_mask = usize::MAX.checked_shl(batch_bits).unwrap_or(0);
    }

    fn notify_insert(&self, entry_id: &EntryID, batch_type: CachedBatchType) {
        self.state().insert(*entry_id, batch_type);
    }

    fn find_memory_victim(&self, cnt: usize) -> Vec<EntryID> {
        if cnt == 0 {
            return vec![];
        }
        self.state().victims(&QueueKind::MEMORY, cnt)
    }

    fn find_disk_victim(&self, cnt: usize) -> Vec<EntryID> {
        if cnt == 0 {
            return vec![];
        }
        self.state().victims(&[QueueKind::Disk], cnt)
    }

    fn notify_access(&self, entry_id: &EntryID, batch_type: CachedBatchType) {
        self.state().access(entry_id, batch_type);
    }

    fn notify_remove(&self, entry_id: &EntryID) {
        self.state().remove(entry_id);
    }

    fn chunk_counts(&self) -> ChunkCounts {
        self.state().counts()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::utils::EntryID;

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    #[test]
    fn test_fifo_within_each_queue() {
        let policy = LiquidPolicy::new();

        let arrow_a = entry(1);
        let arrow_b = entry(2);
        let liquid_a = entry(3);
        let liquid_b = entry(4);

        policy.notify_insert(&arrow_a, CachedBatchType::MemoryArrow);
        policy.notify_insert(&arrow_b, CachedBatchType::MemoryArrow);
        policy.notify_insert(&liquid_a, CachedBatchType::MemoryLiquid);
        policy.notify_insert(&liquid_b, CachedBatchType::MemoryLiquid);

        assert_eq!(policy.find_memory_victim(1), vec![arrow_a]);
        assert_eq!(policy.find_memory_victim(2), vec![arrow_b, liquid_a]);
        assert_eq!(policy.find_memory_victim(1), vec![liquid_b]);
    }

    #[test]
    fn test_queue_priority_order() {
        let policy = LiquidPolicy::new();

        let arrow_entry = entry(1);
        let liquid_entry = entry(2);

        policy.notify_insert(&liquid_entry, CachedBatchType::MemoryLiquid);
        policy.notify_insert(&arrow_entry, CachedBatchType::MemoryArrow);

        // Request more victims than available to ensure we only get what exists.
        let victims = policy.find_memory_victim(5);
        assert_eq!(victims, vec![arrow_entry, liquid_entry]);
    }

    #[test]
    fn test_zero_victim_request_returns_empty() {
        let policy = LiquidPolicy::new();

        policy.notify_insert(&entry(1), CachedBatchType::MemoryArrow);
        assert!(policy.find_memory_victim(0).is_empty());
    }

    #[test]
    fn test_disk_entries_not_evicted() {
        let policy = LiquidPolicy::new();

        let disk_entry = entry(1);
        let arrow_entry = entry(2);
        let liquid_entry = entry(3);

        policy.notify_insert(&disk_entry, CachedBatchType::DiskArrow);
        policy.notify_insert(&arrow_entry, CachedBatchType::MemoryArrow);
        policy.notify_insert(&liquid_entry, CachedBatchType::MemoryLiquid);

        let victims = policy.find_memory_victim(5);
        assert_eq!(victims, vec![arrow_entry, liquid_entry]);

        // Only the disk entry remains and should still not be evicted.
        assert!(policy.find_memory_victim(1).is_empty());
    }

    #[test]
    fn test_disk_victims_and_remove() {
        let policy = LiquidPolicy::new();
        let disk_old = entry(1);
        let disk_new = entry(2);

        policy.notify_insert(&disk_old, CachedBatchType::DiskArrow);
        policy.notify_insert(&disk_new, CachedBatchType::DiskLiquid);

        assert_eq!(policy.find_disk_victim(1), vec![disk_old]);
        policy.notify_remove(&disk_new);
        assert!(policy.find_disk_victim(1).is_empty());
    }

    #[test]
    fn test_a_fill_is_not_the_first_victim_beside_an_older_read_chunk() {
        let policy = LiquidPolicy::new();
        let older = entry(1);
        let filled = entry(2);

        policy.notify_insert(&older, CachedBatchType::MemoryLiquid);
        policy.notify_access(&older, CachedBatchType::MemoryLiquid);
        policy.notify_insert(&filled, CachedBatchType::MemoryLiquid);

        assert_eq!(policy.find_memory_victim(1), vec![older]);
        assert_eq!(policy.find_memory_victim(1), vec![filled]);
    }

    #[test]
    fn test_a_disk_chunk_starts_cold() {
        let policy = LiquidPolicy::new();
        let first = entry(1);
        let second = entry(2);

        policy.notify_insert(&first, CachedBatchType::DiskLiquid);
        policy.notify_insert(&second, CachedBatchType::DiskLiquid);
        policy.notify_access(&first, CachedBatchType::DiskLiquid);

        assert_eq!(policy.find_disk_victim(1), vec![second]);
    }

    #[test]
    fn test_reinsert_handles_cross_queue_move() {
        let policy = LiquidPolicy::new();

        let entry_id = entry(42);

        policy.notify_insert(&entry_id, CachedBatchType::MemoryArrow);
        policy.notify_insert(&entry_id, CachedBatchType::MemoryLiquid);

        let victims = policy.find_memory_victim(2);
        assert_eq!(victims, vec![entry_id]);
    }
}
