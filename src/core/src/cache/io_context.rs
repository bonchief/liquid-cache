use std::fmt::Debug;

use ahash::AHashMap;

use crate::cache::{CacheExpression, utils::EntryID};
use crate::sync::{Arc, RwLock};

/// Per-entry metadata used by the cache.
///
/// This trait covers only the metadata side of the cache: where to find a
/// batch's compressor and lineage expressions. All actual byte IO goes through the
/// [`crate::cache::DiskStore`] held by the cache itself.
pub trait EntryMetadata: Debug + Send + Sync {
    /// Add a lineage expression for an entry.
    fn add_lineage(&self, _entry_id: &EntryID, _expression: Arc<CacheExpression>) {
        // Do nothing by default
    }

    /// Get the lineage expression for an entry.
    /// If None, the entry will be evicted to disk entirely.
    /// The expression records how the column is used by query plans and may inform
    /// encoding decisions without discarding any values.
    fn lineage(&self, _entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        None
    }
}

/// A default implementation of [`EntryMetadata`].
///
/// Lineage expressions are stored in a flat map keyed by [`EntryID`].
#[derive(Debug, Default)]
pub struct DefaultCacheMetadata {
    lineages: RwLock<AHashMap<EntryID, Arc<CacheExpression>>>,
}

impl DefaultCacheMetadata {
    /// Create a new instance of [`DefaultCacheMetadata`].
    pub fn new() -> Self {
        Self {
            lineages: RwLock::new(AHashMap::new()),
        }
    }
}

impl EntryMetadata for DefaultCacheMetadata {
    fn add_lineage(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        let mut guard = self.lineages.write().unwrap();
        guard.insert(*entry_id, expression);
    }

    fn lineage(&self, entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        let guard = self.lineages.read().unwrap();
        guard.get(entry_id).cloned()
    }
}
