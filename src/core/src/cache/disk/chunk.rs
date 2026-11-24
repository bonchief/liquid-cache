//! The payload of an object: a directory of the batches it holds, then each
//! batch's bytes.
//!
//! Layout, little-endian, offsets from the payload's start:
//!
//! - batch count (4 bytes) and 4 zero bytes;
//! - per batch, 32 bytes: entry id (8), form (8, of which the low byte is
//!   the caller's tag), offset (8), length (8);
//! - the batches' bytes, each starting on a multiple of [`ALIGN`], so a
//!   decoder that wants aligned buffers can slice them in place.

use std::ops::Range;

use bytes::Bytes;

use super::DiskError;
use crate::cache::utils::EntryID;

/// What every batch's bytes start on, relative to the payload. The payload
/// starts on a page, after the 64-byte header.
const ALIGN: usize = 64;
const HEAD: usize = 8;
const ENTRY: usize = 32;

/// One batch of an object: the entry it is stored under, a tag the caller
/// gives its form by, and its bytes.
#[derive(Debug, Clone)]
pub struct StoredBatch {
    entry: EntryID,
    form: u8,
    bytes: Bytes,
}

impl StoredBatch {
    /// `bytes` of `entry`, in the form the caller calls `form`.
    pub fn new(entry: EntryID, form: u8, bytes: Bytes) -> Self {
        Self { entry, form, bytes }
    }

    /// The entry the batch is stored under.
    pub fn entry(&self) -> EntryID {
        self.entry
    }

    /// The tag the batch was stored with.
    pub fn form(&self) -> u8 {
        self.form
    }

    /// The batch's bytes.
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }
}

/// Where each batch of a payload lies.
#[derive(Debug, Clone)]
pub(super) struct Directory {
    batches: Vec<(EntryID, u8, Range<usize>)>,
}

impl Directory {
    /// Where batches of `lens` bytes go, and the payload's length.
    fn layout(lens: impl ExactSizeIterator<Item = usize>) -> Option<(Vec<Range<usize>>, usize)> {
        let mut at = ENTRY
            .checked_mul(lens.len())?
            .checked_add(HEAD)?
            .checked_next_multiple_of(ALIGN)?;
        let mut end = at;
        let mut ranges = Vec::with_capacity(lens.len());
        for len in lens {
            end = at.checked_add(len)?;
            ranges.push(at..end);
            at = end.checked_next_multiple_of(ALIGN)?;
        }
        Some((ranges, end))
    }

    /// The length of the payload holding batches of `lens` bytes.
    pub(super) fn payload_len(lens: impl ExactSizeIterator<Item = usize>) -> Option<usize> {
        Self::layout(lens).map(|(_, len)| len)
    }

    /// Write `batches` and their directory into the start of `body`, which
    /// is zeroed; the payload's length.
    pub(super) fn write(batches: &[StoredBatch], body: &mut [u8]) -> Result<usize, DiskError> {
        let (ranges, len) =
            Self::layout(batches.iter().map(|batch| batch.bytes.len())).ok_or(DiskError::Full)?;
        let count = u32::try_from(batches.len()).map_err(|_| DiskError::Full)?;
        Self::put(body, 0, &count.to_le_bytes())?;
        for (index, (batch, range)) in batches.iter().zip(&ranges).enumerate() {
            let at = HEAD + index * ENTRY;
            let fields = [
                usize::from(batch.entry) as u64,
                u64::from(batch.form),
                range.start as u64,
                batch.bytes.len() as u64,
            ];
            for (word, field) in fields.into_iter().enumerate() {
                Self::put(body, at + word * 8, &field.to_le_bytes())?;
            }
            Self::put(body, range.start, &batch.bytes)?;
        }
        Ok(len)
    }

    fn put(body: &mut [u8], at: usize, bytes: &[u8]) -> Result<(), DiskError> {
        at.checked_add(bytes.len())
            .and_then(|end| body.get_mut(at..end))
            .ok_or(DiskError::Full)?
            .copy_from_slice(bytes);
        Ok(())
    }

    /// The directory at the start of `payload`, every range of it within
    /// the payload; a payload that does not hold one is corrupt.
    pub(super) fn read(payload: &[u8]) -> Result<Self, DiskError> {
        let word = |at: usize| -> Result<u64, DiskError> {
            at.checked_add(8)
                .and_then(|end| payload.get(at..end))
                .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
                .map(u64::from_le_bytes)
                .ok_or(DiskError::Corrupt)
        };
        let count = payload
            .get(..4)
            .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
            .map(u32::from_le_bytes)
            .ok_or(DiskError::Corrupt)? as usize;
        if count > payload.len().saturating_sub(HEAD) / ENTRY {
            return Err(DiskError::Corrupt);
        }
        let mut batches = Vec::with_capacity(count);
        for index in 0..count {
            let at = HEAD + index * ENTRY;
            let entry = usize::try_from(word(at)?).map_err(|_| DiskError::Corrupt)?;
            let form = u8::try_from(word(at + 8)?).map_err(|_| DiskError::Corrupt)?;
            let start = usize::try_from(word(at + 16)?).map_err(|_| DiskError::Corrupt)?;
            let len = usize::try_from(word(at + 24)?).map_err(|_| DiskError::Corrupt)?;
            let range = start
                .checked_add(len)
                .filter(|end| *end <= payload.len())
                .map(|end| start..end)
                .ok_or(DiskError::Corrupt)?;
            batches.push((EntryID::from(entry), form, range));
        }
        Ok(Self { batches })
    }

    pub(super) fn holds(&self, entry: &EntryID) -> bool {
        self.batches.iter().any(|(id, _, _)| id == entry)
    }
}

/// An object read back whole: its payload, and where each of its batches
/// lies in it. Slicing a batch out copies nothing; the payload lives as long
/// as any slice of it.
#[derive(Debug, Clone)]
pub struct ChunkObject {
    payload: Bytes,
    directory: Directory,
}

impl ChunkObject {
    pub(super) fn new(payload: Bytes, directory: Directory) -> Self {
        Self { payload, directory }
    }

    /// The form and bytes of `entry`'s batch, if the object holds it.
    pub fn batch(&self, entry: &EntryID) -> Option<(u8, Bytes)> {
        self.directory
            .batches
            .iter()
            .find(|(id, _, _)| id == entry)
            .map(|(_, form, range)| (*form, self.payload.slice(range.clone())))
    }

    /// Every batch the object holds: entry, form and bytes.
    pub fn batches(&self) -> impl Iterator<Item = (EntryID, u8, Bytes)> + '_ {
        self.directory
            .batches
            .iter()
            .map(|(id, form, range)| (*id, *form, self.payload.slice(range.clone())))
    }

    pub(super) fn members(&self) -> usize {
        self.directory.batches.len()
    }

    /// The bytes the object's payload holds.
    pub fn len(&self) -> usize {
        self.payload.len()
    }

    /// Whether the object holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.payload.is_empty()
    }
}
