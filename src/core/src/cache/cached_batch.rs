//! Cached batch types.

use std::{fmt::Display, sync::Arc};

use arrow::array::ArrayRef;
use arrow_schema::DataType;

use crate::liquid_array::LiquidArrayRef;

/// A cached entry storing data in various formats.
#[derive(Debug, Clone)]
pub enum CacheEntry {
    /// Cached batch in memory as Arrow array.
    MemoryArrow(ArrayRef),
    /// Cached batch in memory as liquid array.
    MemoryLiquid(LiquidArrayRef),
    /// Cached batch on disk as liquid array.
    DiskLiquid {
        /// Original Arrow data type.
        data_type: DataType,
        /// The disk budget the record holds: its share of the extent its
        /// object takes in the store. Callers building a record pass the
        /// payload's length; the write that installs it sets its share.
        disk_bytes: usize,
    },
    /// Cached batch on disk as Arrow array.
    DiskArrow {
        /// Original Arrow data type.
        data_type: DataType,
        /// The disk budget the record holds: its share of the extent its
        /// object takes in the store. Callers building a record pass the
        /// payload's length; the write that installs it sets its share.
        disk_bytes: usize,
    },
}

impl CacheEntry {
    /// Construct a cached batch stored as an in-memory Arrow array.
    pub fn memory_arrow(array: ArrayRef) -> Self {
        Self::MemoryArrow(array)
    }

    /// Construct a cached batch stored as an in-memory Liquid array.
    pub fn memory_liquid(array: LiquidArrayRef) -> Self {
        Self::MemoryLiquid(array)
    }

    /// Construct a cached batch stored on disk as Liquid bytes.
    pub fn disk_liquid(data_type: DataType, disk_bytes: usize) -> Self {
        Self::DiskLiquid {
            data_type,
            disk_bytes,
        }
    }

    /// Construct a cached batch stored on disk as Arrow bytes.
    pub fn disk_arrow(data_type: DataType, disk_bytes: usize) -> Self {
        Self::DiskArrow {
            data_type,
            disk_bytes,
        }
    }

    /// This record holding `bytes` of the disk budget; a memory entry is
    /// returned as it is.
    pub(crate) fn charged_at(self, bytes: usize) -> Self {
        match self {
            Self::DiskLiquid { data_type, .. } => Self::DiskLiquid {
                data_type,
                disk_bytes: bytes,
            },
            Self::DiskArrow { data_type, .. } => Self::DiskArrow {
                data_type,
                disk_bytes: bytes,
            },
            memory @ (Self::MemoryArrow(_) | Self::MemoryLiquid(_)) => memory,
        }
    }

    /// The memory a disk record is charged: its index slot, its policy
    /// record and its extent in the disk store, each a few dozen bytes, with
    /// room for the maps' own overhead.
    ///
    /// A disk record holds no array, but a full disk tier holds millions of
    /// them, so their metadata is charged to the memory budget like any
    /// entry's memory and bounded by it. A memory entry's own record is not
    /// charged separately: next to the batch of rows its array holds, it is
    /// small.
    pub const DISK_RECORD_BYTES: usize = 256;

    /// The memory the cache charges the entry: its array's, or for a disk
    /// record [`Self::DISK_RECORD_BYTES`].
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => array.get_array_memory_size(),
            Self::MemoryLiquid(array) => array.get_array_memory_size(),
            Self::DiskLiquid { .. } | Self::DiskArrow { .. } => Self::DISK_RECORD_BYTES,
        }
    }

    /// Reference count (if any) of the backing storage.
    pub fn reference_count(&self) -> usize {
        match self {
            Self::MemoryArrow(array) => Arc::strong_count(array),
            Self::MemoryLiquid(array) => Arc::strong_count(array),
            Self::DiskLiquid { .. } | Self::DiskArrow { .. } => 0,
        }
    }
}

impl Display for CacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryArrow(_) => write!(f, "MemoryArrow"),
            Self::MemoryLiquid(_) => write!(f, "MemoryLiquid"),
            Self::DiskLiquid { .. } => write!(f, "DiskLiquid"),
            Self::DiskArrow { .. } => write!(f, "DiskArrow"),
        }
    }
}

/// The type of the cached batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum CachedBatchType {
    /// Cached batch in memory as Arrow array.
    MemoryArrow,
    /// Cached batch in memory as liquid array.
    MemoryLiquid,
    /// Cached batch on disk as liquid array.
    DiskLiquid,
    /// Cached batch on disk as Arrow array.
    DiskArrow,
}

impl From<&CacheEntry> for CachedBatchType {
    fn from(batch: &CacheEntry) -> Self {
        match batch {
            CacheEntry::MemoryArrow(_) => Self::MemoryArrow,
            CacheEntry::MemoryLiquid(_) => Self::MemoryLiquid,
            CacheEntry::DiskLiquid { .. } => Self::DiskLiquid,
            CacheEntry::DiskArrow { .. } => Self::DiskArrow,
        }
    }
}
