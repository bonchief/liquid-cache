//! Chunks read from disk again since they were filed there, offered for
//! promotion back to memory ([`super::LiquidCache::promote_cold`]).

use std::collections::VecDeque;

use ahash::AHashSet;

use crate::cache::utils::EntryID;
use crate::sync::Mutex;

/// Chunks offered for promotion, each named by an entry of it and the
/// identity it is held for, oldest first: at most [`Self::QUEUED`] at once,
/// each once. A read offers a chunk ([`Self::offer`]) and never waits; an
/// owner takes them one at a time ([`Self::next`]) and promotes them off the
/// read's path. An offer past the bound is dropped, as a repeat hit of the
/// same chunk offers it again later.
#[derive(Debug, Default)]
pub struct Promotions {
    queued: Mutex<Queued>,
    ready: tokio::sync::Notify,
}

#[derive(Debug, Default)]
struct Queued {
    order: VecDeque<(EntryID, EntryID, u64)>,
    named: AHashSet<(EntryID, u64)>,
    /// Taken by [`Promotions::next`], not yet done.
    taken: usize,
}

/// One chunk to promote, taken off the queue; it counts as pending until
/// dropped.
#[derive(Debug)]
pub struct Promotion<'a> {
    promotions: &'a Promotions,
    /// An entry of the chunk, as the read that offered it named it.
    pub entry: EntryID,
    /// The identity the chunk is held for.
    pub identity: u64,
}

impl Drop for Promotion<'_> {
    fn drop(&mut self) {
        let mut queued = self.promotions.lock();
        queued.taken = queued.taken.saturating_sub(1);
    }
}

impl Promotions {
    /// The most chunks offered and not yet taken.
    pub const QUEUED: usize = 1024;

    fn lock(&self) -> crate::sync::MutexGuard<'_, Queued> {
        self.queued
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Offer the chunk of `entry` held for `identity`, unless it is queued
    /// already or the queue is full. Never waits.
    pub(crate) fn offer(&self, chunk: EntryID, entry: EntryID, identity: u64) {
        let mut queued = self.lock();
        if queued.order.len() >= Self::QUEUED || !queued.named.insert((chunk, identity)) {
            return;
        }
        queued.order.push_back((chunk, entry, identity));
        drop(queued);
        self.ready.notify_one();
    }

    /// The oldest chunk offered, once there is one. Once taken it can be
    /// offered again.
    pub async fn next(&self) -> Promotion<'_> {
        loop {
            let ready = self.ready.notified();
            {
                let mut queued = self.lock();
                if let Some((chunk, entry, identity)) = queued.order.pop_front() {
                    queued.named.remove(&(chunk, identity));
                    queued.taken += 1;
                    return Promotion {
                        promotions: self,
                        entry,
                        identity,
                    };
                }
            }
            ready.await;
        }
    }

    /// Chunks offered and not yet promoted or refused: queued, or taken and
    /// not yet dropped.
    pub fn pending(&self) -> usize {
        let queued = self.lock();
        queued.order.len() + queued.taken
    }
}
