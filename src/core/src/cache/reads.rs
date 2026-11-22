//! The memory that disk reads and fills hold outside the cache's budget.
//!
//! A read of a disk record fetches its chunk's whole object into one buffer
//! and decodes its batches from it, and a read that keeps those batches for
//! the rest of a row group holds them until it is done. A fill decodes and
//! encodes a whole chunk before inserting it. None of that memory is an
//! entry of the cache, so none of it is in the memory budget; [`ReadBudget`]
//! bounds it instead.

use crate::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// **What disk reads and fills hold in memory at once is at most
/// [`Self::max`] bytes**, with one exception: a fill's decode is charged by
/// its estimate ([`ReadBudget::try_take`] before, [`ReadCharge::grow`]
/// after), and while it runs it may hold more than its estimate.
///
/// Each read or fill takes a [`ReadCharge`] for what it will hold before it
/// allocates, and gives it back when the charge drops: a read once its
/// buffer and every batch decoded from it are let go, a fill once its chunk
/// is inserted or refused. A charge that does not fit is refused at once;
/// nothing waits for room, so a read that holds a charge never waits on
/// another. A refused read is a miss: the caller reads the rows from
/// Parquet. [`Self::refused`] counts the refusals.
#[derive(Debug)]
pub struct ReadBudget {
    max: AtomicUsize,
    used: AtomicUsize,
    refused: AtomicU64,
}

impl ReadBudget {
    /// What a cache built without [`Self::set_max`] lets reads and fills hold:
    /// 256 MiB, a few chunk objects of tens of megabytes at once.
    pub const DEFAULT_MAX: usize = 256 << 20;

    pub(crate) fn new(max: usize) -> Arc<Self> {
        Arc::new(Self {
            max: AtomicUsize::new(max),
            used: AtomicUsize::new(0),
            refused: AtomicU64::new(0),
        })
    }

    /// The most bytes reads and fills hold at once.
    pub fn max(&self) -> usize {
        self.max.load(Ordering::Relaxed)
    }

    /// Change [`Self::max`]. Charges already taken are kept; new ones fit
    /// under the new bound.
    pub fn set_max(&self, max: usize) {
        self.max.store(max, Ordering::Relaxed);
    }

    /// The bytes charges hold now.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// Charges refused so far.
    pub fn refused(&self) -> u64 {
        self.refused.load(Ordering::Relaxed)
    }

    /// A charge of `bytes`, if it fits under [`Self::max`] with what is held.
    pub fn try_take(self: &Arc<Self>, bytes: usize) -> Option<ReadCharge> {
        if self.add(bytes) {
            return Some(ReadCharge {
                budget: Arc::clone(self),
                bytes: AtomicUsize::new(bytes),
            });
        }
        self.refused.fetch_add(1, Ordering::Relaxed);
        None
    }

    fn add(&self, bytes: usize) -> bool {
        let max = self.max();
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|total| *total <= max)
            })
            .is_ok()
    }
}

/// Bytes held against a [`ReadBudget`], given back when dropped.
#[derive(Debug)]
pub struct ReadCharge {
    budget: Arc<ReadBudget>,
    bytes: AtomicUsize,
}

impl ReadCharge {
    /// The bytes this charge holds.
    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Hold `to` bytes rather than what the charge holds, if it is more and
    /// the rest fits; whether the charge now holds at least `to`.
    pub fn grow(&self, to: usize) -> bool {
        let held = self.bytes();
        if to <= held {
            return true;
        }
        if !self.budget.add(to - held) {
            self.budget.refused.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        self.bytes.fetch_add(to - held, Ordering::Relaxed);
        true
    }
}

impl Drop for ReadCharge {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes(), Ordering::AcqRel);
    }
}
