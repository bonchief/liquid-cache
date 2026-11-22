//! The disk tier's store: one sparse file holding each object in an extent of
//! whole pages, for the life of one process.
//!
//! It replaces t4 0.1.9's key-value store. A cache emptied at every start
//! needs none of t4's write-ahead log, which grew with every put and remove,
//! nor its durability; and a t4 read could return bytes a later put of
//! another key wrote into the slot it was reading. What this store promises
//! is stated on the types that keep it: [`DiskStore`] for the file and its
//! objects, `Extents` for space and pins, `Header` for what a read
//! accepts.
//!
//! # Layout
//!
//! An object is one extent: a 64-byte `Header` and the payload right after
//! it, padded to a whole number of pages. The payload holds one column
//! chunk's batches, or as many of them as were written together: a
//! directory naming each batch's entry, form, offset and length, then the
//! batches' bytes (see `chunk`). The object is keyed by the chunk (the entry
//! id with its batch bits cleared) and every batch's entry names the same
//! extent, so a read of any of them fetches the whole chunk in one `pread`,
//! and the extent is freed once no entry names it and no read pins it.

mod chunk;
mod extents;
mod io;

use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::cache::utils::EntryID;
use chunk::Directory;
pub use chunk::{ChunkObject, StoredBatch};
use extents::{Extents, PAGE};
pub use io::{DiskFile, FileIo};
use io::{PageBuffer, Pool};

/// What a [`DiskStore`] operation can fail with.
#[derive(Debug)]
pub enum DiskError {
    /// The store holds no object for the entry.
    NotFound,
    /// The object stored for the entry was written for another identity: the
    /// entry's key changed hands. A plain miss.
    Foreign,
    /// The object's header does not name the entry, or its length or
    /// checksum does not match what was read. A miss, and a fault.
    Corrupt,
    /// No free extent below the store's capacity is long enough.
    Full,
    /// The file could not be read or written.
    Io(std::io::Error),
    /// The store's I/O threads are gone.
    Closed,
    /// The caller of [`DiskStore::get_chunk_charged`] declined the read: no
    /// room to hold what it would allocate.
    Refused,
}

impl std::fmt::Display for DiskError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => out.write_str("the disk tier holds no object for this entry"),
            Self::Foreign => {
                out.write_str("the disk tier's object for this entry is another source's")
            }
            Self::Corrupt => out.write_str("the disk tier's object does not match its header"),
            Self::Full => out.write_str("the disk tier has no free extent long enough"),
            Self::Io(error) => write!(out, "disk tier I/O failed: {error}"),
            Self::Closed => out.write_str("the disk tier's I/O threads have stopped"),
            Self::Refused => out.write_str("the disk tier read was refused: no room to hold it"),
        }
    }
}

impl std::error::Error for DiskError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DiskError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// What [`DiskOptions::with_file`] puts in front of the real file.
type Layer = Arc<dyn Fn(FileIo) -> Arc<dyn DiskFile> + Send + Sync>;

/// How a [`DiskStore`] is mounted.
#[derive(Clone)]
pub struct DiskOptions {
    capacity: u64,
    direct: bool,
    threads: usize,
    writers: usize,
    layer: Option<Layer>,
}

impl std::fmt::Debug for DiskOptions {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("DiskOptions")
            .field("capacity", &self.capacity)
            .field("direct", &self.direct)
            .field("threads", &self.threads)
            .field("writers", &self.writers)
            .field("layered", &self.layer.is_some())
            .finish()
    }
}

impl DiskOptions {
    /// What a store mounted with [`Default`] may hold.
    pub const DEFAULT_CAPACITY: u64 = 16 << 30;

    /// A store of at most `capacity` bytes (rounded down to whole pages),
    /// read and written through the page cache, read on four I/O threads
    /// and written on one.
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            direct: false,
            threads: 4,
            writers: 1,
            layer: None,
        }
    }

    /// A store for a cache whose disk budget is `budget` bytes: its file
    /// `budget` plus an eighth, and at least 64 MiB more.
    ///
    /// The store's bytes in use can pass the budget the cache charges: an
    /// extent stays whole while any batch of its chunk still names it, a
    /// replaced extent stays while a read pins it (at most what the read
    /// budget lets reads hold), and free space can lie in pieces too short
    /// for an object. The cache drops disk victims when the store is full
    /// all the same, so headroom only spares those drops. The file is
    /// sparse: the headroom takes no blocks until written, but the
    /// filesystem must have room for it.
    pub fn for_budget(budget: u64) -> Self {
        Self::new(budget.saturating_add((budget / 8).max(64 << 20)))
    }

    /// Bypass the page cache: `O_DIRECT` on Linux, `F_NOCACHE` on macOS.
    /// Filesystems that refuse `O_DIRECT`, such as tmpfs on older kernels,
    /// then fail the mount. Buffered, reads keep their pages cached and
    /// every write is written back and dropped from the cache (Linux only).
    pub fn with_direct(mut self, direct: bool) -> Self {
        self.direct = direct;
        self
    }

    /// The most reads in flight at once.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// The most writes in flight at once. Writes, with their writeback,
    /// run on threads of their own, so a spill never queues ahead of a
    /// query's read.
    pub fn with_writers(mut self, writers: usize) -> Self {
        self.writers = writers.max(1);
        self
    }

    /// Do the store's I/O through what `layer` makes of the real file, once
    /// it is opened and sized, instead of through the file itself. The
    /// store's every read, write and [`DiskFile::sync`] then goes through
    /// it, so an implementation that wraps the file and fails some calls
    /// drives the store's own error paths. Without it the store uses the
    /// [`FileIo`] as it is.
    pub fn with_file(
        mut self,
        layer: impl Fn(FileIo) -> Arc<dyn DiskFile> + Send + Sync + 'static,
    ) -> Self {
        self.layer = Some(Arc::new(layer));
        self
    }
}

impl Default for DiskOptions {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAPACITY)
    }
}

/// Which object a put or get is for: the chunk it holds batches of (the
/// entry id with its batch bits cleared), and the identity of the source it
/// was cached for. Both are written in the object's header and checked on
/// every read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectKey {
    entry: EntryID,
    identity: u64,
}

impl ObjectKey {
    /// The object of `chunk` cached for `identity`.
    pub fn new(chunk: EntryID, identity: u64) -> Self {
        Self {
            entry: chunk,
            identity,
        }
    }
}

/// One process's disk tier. Clones share it.
///
/// - **One file, never past its capacity.** Mounting truncates the file and
///   sets its length to [`DiskOptions::new`]'s capacity, so it is sparse and
///   cannot grow beyond it; every extent lies below that length. Nothing
///   survives a restart: there is no log, no `O_DSYNC`, and the owner deletes
///   the directory at start.
/// - **Each object is charged what it takes.** [`Self::footprint`] is the
///   extent an object of a given payload occupies, header and padding
///   included.
/// - **A read returns only its own object's bytes.** It returns an object
///   only if the extent's header names the requested chunk and identity, the
///   payload's length and xxh3 match the header, and the payload's directory
///   lists the entry asked for. Any other outcome is an error, and
///   [`Self::mismatches`] counts those whose header or directory did not
///   match.
/// - **No reuse under a read.** A read pins its extent from the index lookup
///   until the `pread` is done, even if the caller drops the read; an
///   extent is freed only once no entry names it and its last pin goes.
/// - **Shared extents.** Every batch of an object names its extent. Removing
///   one batch's entry leaves the extent to the others; its disk budget
///   share, which the cache releases with the batch's record, is then
///   counted free while the extent still takes it, until the chunk's other
///   batches go. The file's length still bounds what the store holds.
#[derive(Clone, Debug)]
pub struct DiskStore {
    shared: Arc<Shared>,
    /// Runs reads.
    pool: Arc<Pool>,
    /// Runs writes and their writeback, apart from reads.
    writers: Arc<Pool>,
}

#[derive(Debug)]
struct Shared {
    file: Arc<dyn DiskFile>,
    extents: Mutex<Extents>,
    mismatches: AtomicU64,
}

impl DiskStore {
    /// Mount an empty store at `path`, creating the file or discarding what
    /// it held.
    pub fn mount(path: impl AsRef<Path>, options: DiskOptions) -> Result<Self, DiskError> {
        let file = FileIo::open(path.as_ref(), options.direct)?;
        let extents = Extents::new(options.capacity);
        file.set_len(extents.capacity())?;
        let file: Arc<dyn DiskFile> = match &options.layer {
            Some(layer) => layer(file),
            None => Arc::new(file),
        };
        let pool = Pool::start("liquid-disk-read", options.threads)?;
        let writers = Pool::start("liquid-disk-write", options.writers)?;
        Ok(Self {
            shared: Arc::new(Shared {
                file,
                extents: Mutex::new(extents),
                mismatches: AtomicU64::new(0),
            }),
            pool: Arc::new(pool),
            writers: Arc::new(writers),
        })
    }

    /// The bytes an object of `payload` bytes takes in the file: its header
    /// and payload rounded up to whole pages. What the cache charges it.
    pub const fn footprint(payload: usize) -> usize {
        let page = PAGE as usize;
        payload
            .saturating_add(Header::LEN)
            .div_ceil(page)
            .saturating_mul(page)
    }

    /// The bytes an object holding batches of `lens` bytes takes in the
    /// file: its header, directory and batches, rounded up to whole pages.
    /// `None` past what an object can address.
    pub fn chunk_footprint(lens: impl ExactSizeIterator<Item = usize>) -> Option<usize> {
        Directory::payload_len(lens).map(Self::footprint)
    }

    /// Store `batches` as one object of `chunk`, in one extent, and name it
    /// as each batch's entry's object, replacing what each entry named.
    ///
    /// The object is written into a fresh extent, never over a current one,
    /// and the entries are pointed at it only once the write succeeds. A
    /// failed or dropped put leaves every entry's previous object as it was
    /// and frees its extent.
    pub async fn put_chunk(
        &self,
        chunk: ObjectKey,
        batches: Vec<StoredBatch>,
    ) -> Result<(), DiskError> {
        let len = Self::chunk_footprint(batches.iter().map(|batch| batch.bytes().len()))
            .ok_or(DiskError::Full)?;
        let entries: Vec<EntryID> = batches.iter().map(StoredBatch::entry).collect();
        let at = u64::try_from(len)
            .ok()
            .and_then(|len| self.shared.extents().allocate(len))
            .ok_or(DiskError::Full)?;
        let hold = Hold {
            shared: Arc::clone(&self.shared),
            at,
        };
        let writing = hold.again();
        let shared = Arc::clone(&self.shared);
        self.writers
            .run(move || {
                let _writing = writing;
                shared.write(chunk, &batches, at, len)
            })
            .await??;
        {
            let mut extents = self.shared.extents();
            for entry in entries {
                extents.name(entry, at);
            }
        }
        drop(hold);
        Ok(())
    }

    /// The object `entry` names, read whole in one `pread`, if it is
    /// `chunk`'s and lists `entry`.
    pub async fn get_chunk(
        &self,
        entry: EntryID,
        chunk: ObjectKey,
    ) -> Result<ChunkObject, DiskError> {
        self.get_chunk_charged(entry, chunk, |_| Some(()))
            .await
            .map(|(object, ())| object)
    }

    /// [`Self::get_chunk`], once `charge` has agreed to the bytes the read
    /// allocates (its buffer: the object's extent and a page to align it),
    /// before it allocates them: what `charge` returned, with the object.
    /// `None` from `charge` refuses the read ([`DiskError::Refused`]).
    pub async fn get_chunk_charged<T>(
        &self,
        entry: EntryID,
        chunk: ObjectKey,
        charge: impl FnOnce(usize) -> Option<T>,
    ) -> Result<(ChunkObject, T), DiskError> {
        let (at, len) = self
            .shared
            .extents()
            .pin_named(&entry)
            .ok_or(DiskError::NotFound)?;
        let hold = Hold {
            shared: Arc::clone(&self.shared),
            at,
        };
        let held = usize::try_from(len)
            .ok()
            .and_then(|len| len.checked_add(PAGE as usize))
            .ok_or(DiskError::Corrupt)?;
        let charged = charge(held).ok_or(DiskError::Refused)?;
        let shared = Arc::clone(&self.shared);
        let (buffer, payload) = self
            .pool
            .run(move || {
                let _reading = hold;
                shared.read(chunk, at, len)
            })
            .await??;
        let payload = buffer.into_bytes(payload);
        let directory = match Directory::read(&payload) {
            Ok(directory) if directory.holds(&entry) => directory,
            Ok(_) | Err(_) => {
                self.shared.mismatches.fetch_add(1, Ordering::Relaxed);
                return Err(DiskError::Corrupt);
            }
        };
        Ok((ChunkObject::new(payload, directory), charged))
    }

    /// Forget `entry`'s object; whether it had one. Its extent is freed once
    /// no other entry names it and no read pins it.
    pub fn remove(&self, entry: &EntryID) -> bool {
        self.shared.extents().remove(entry)
    }

    /// How many entries name an object.
    pub fn len(&self) -> usize {
        self.shared.extents().len()
    }

    /// Whether no entry names an object.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes of the file in extents: named objects, and replaced ones a
    /// read still pins.
    pub fn used_bytes(&self) -> u64 {
        self.shared.extents().used()
    }

    /// The most bytes the file holds: its length.
    pub fn capacity(&self) -> u64 {
        self.shared.extents().capacity()
    }

    /// Reads, so far, whose extent's header or checksum did not match the
    /// object asked for. An object of another identity is a plain miss and
    /// is not counted.
    pub fn mismatches(&self) -> u64 {
        self.shared.mismatches.load(Ordering::Relaxed)
    }
}

impl Shared {
    fn extents(&self) -> MutexGuard<'_, Extents> {
        self.extents.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Write `key`'s header, and `batches` with their directory as its
    /// payload, into the extent of `len` bytes at `at`. Runs on an I/O
    /// thread.
    fn write(
        &self,
        key: ObjectKey,
        batches: &[StoredBatch],
        at: u64,
        len: usize,
    ) -> Result<(), DiskError> {
        let mut buffer = PageBuffer::zeroed(len);
        let extent = buffer.as_mut_slice();
        let (head, body) = extent
            .split_at_mut_checked(Header::LEN)
            .ok_or(DiskError::Full)?;
        let payload = Directory::write(batches, body)?;
        let payload = body.get(..payload).ok_or(DiskError::Full)?;
        Header::of(key, payload).write(head);
        self.file.write_all_at(buffer.as_slice(), at)?;
        self.file.sync(at, len as u64)?;
        Ok(())
    }

    /// Read the extent of `len` bytes at `at`, and check it holds `key`'s
    /// object: the buffer and where in it the payload lies. Runs on an I/O
    /// thread.
    fn read(
        &self,
        key: ObjectKey,
        at: u64,
        len: u64,
    ) -> Result<(PageBuffer, Range<usize>), DiskError> {
        let len = usize::try_from(len).map_err(|_| DiskError::Corrupt)?;
        let mut buffer = PageBuffer::zeroed(len);
        self.file.read_exact_at(buffer.as_mut_slice(), at)?;
        match Header::check(key, buffer.as_slice()) {
            Ok(payload) => Ok((buffer, payload)),
            // An object of another identity is the key changing hands, a
            // plain miss; only a header that does not match is a fault.
            Err(DiskError::Foreign) => Err(DiskError::Foreign),
            Err(error) => {
                self.mismatches.fetch_add(1, Ordering::Relaxed);
                Err(error)
            }
        }
    }
}

/// A pin on the extent at `at`, given back however the read or write that
/// holds it ends.
struct Hold {
    shared: Arc<Shared>,
    at: u64,
}

impl Hold {
    /// A second pin on the same extent.
    fn again(&self) -> Self {
        self.shared.extents().pin(self.at);
        Self {
            shared: Arc::clone(&self.shared),
            at: self.at,
        }
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.shared.extents().unpin(self.at);
    }
}

/// The first [`Header::LEN`] bytes of an extent: what object it holds and how
/// to check it.
///
/// Layout, little-endian: magic (8 bytes), chunk id (8), identity (8),
/// payload length (8), xxh3-64 of the payload (8), then zeros. A read
/// returns bytes only if the header's chunk id, identity, length and
/// checksum all match.
struct Header {
    key: ObjectKey,
    len: u64,
    checksum: u64,
}

impl Header {
    const LEN: usize = 64;
    const MAGIC: [u8; 8] = *b"LQDISK02";

    fn of(key: ObjectKey, payload: &[u8]) -> Self {
        Self {
            key,
            len: payload.len() as u64,
            checksum: twox_hash::XxHash3_64::oneshot(payload),
        }
    }

    /// Write the header into `head`, which is [`Self::LEN`] zeroed bytes.
    fn write(&self, head: &mut [u8]) {
        let fields = [
            u64::from_le_bytes(Self::MAGIC),
            usize::from(self.key.entry) as u64,
            self.key.identity,
            self.len,
            self.checksum,
        ];
        for (word, field) in head.chunks_exact_mut(8).zip(fields) {
            word.copy_from_slice(&field.to_le_bytes());
        }
    }

    /// Where `extent`'s payload lies, if the extent holds `key`'s object.
    fn check(key: ObjectKey, extent: &[u8]) -> Result<Range<usize>, DiskError> {
        let head = extent.get(..Self::LEN).ok_or(DiskError::Corrupt)?;
        let mut words = head
            .chunks_exact(8)
            .map(|word| <[u8; 8]>::try_from(word).map(u64::from_le_bytes));
        let mut next = || words.next().and_then(Result::ok).ok_or(DiskError::Corrupt);
        let (magic, entry, identity, len, checksum) = (next()?, next()?, next()?, next()?, next()?);
        if magic != u64::from_le_bytes(Self::MAGIC) || entry != usize::from(key.entry) as u64 {
            return Err(DiskError::Corrupt);
        }
        if identity != key.identity {
            return Err(DiskError::Foreign);
        }
        let payload = usize::try_from(len)
            .ok()
            .and_then(|len| Self::LEN.checked_add(len))
            .filter(|end| *end <= extent.len())
            .map(|end| Self::LEN..end)
            .ok_or(DiskError::Corrupt)?;
        let bytes = extent.get(payload.clone()).ok_or(DiskError::Corrupt)?;
        if twox_hash::XxHash3_64::oneshot(bytes) != checksum {
            return Err(DiskError::Corrupt);
        }
        Ok(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::AtomicBool;

    /// Which calls of a [`Faulty`] file fail.
    #[derive(Debug, Default)]
    struct Faults {
        writes: AtomicBool,
        syncs: AtomicBool,
        reads: AtomicBool,
    }

    /// The real file, failing the calls `faults` names: writes with
    /// `ENOSPC`, syncs and reads with `EIO`.
    #[derive(Debug)]
    struct Faulty {
        file: FileIo,
        faults: Arc<Faults>,
    }

    impl Faulty {
        fn fail(flag: &AtomicBool, code: i32) -> std::io::Result<()> {
            match flag.load(Ordering::Relaxed) {
                true => Err(std::io::Error::from_raw_os_error(code)),
                false => Ok(()),
            }
        }
    }

    impl DiskFile for Faulty {
        fn read_exact_at(&self, buffer: &mut [u8], at: u64) -> std::io::Result<()> {
            Self::fail(&self.faults.reads, libc::EIO)?;
            self.file.read_exact_at(buffer, at)
        }

        fn write_all_at(&self, buffer: &[u8], at: u64) -> std::io::Result<()> {
            Self::fail(&self.faults.writes, libc::ENOSPC)?;
            self.file.write_all_at(buffer, at)
        }

        fn sync(&self, at: u64, len: u64) -> std::io::Result<()> {
            Self::fail(&self.faults.syncs, libc::EIO)?;
            self.file.sync(at, len)
        }
    }

    /// The first entry of chunk `chunk`, with 16 batch bits as the Parquet
    /// layer packs them.
    fn chunk(chunk: usize) -> EntryID {
        EntryID::from(chunk << 16)
    }

    fn batch(chunk: EntryID, batch: usize) -> EntryID {
        EntryID::from(usize::from(chunk) + batch)
    }

    fn batches(chunk: EntryID, count: usize, len: usize) -> Vec<StoredBatch> {
        (0..count)
            .map(|index| {
                let bytes = Bytes::from(vec![index as u8 + 1; len]);
                StoredBatch::new(batch(chunk, index), 0, bytes)
            })
            .collect()
    }

    fn mounted(capacity: u64) -> (tempfile::TempDir, std::path::PathBuf, DiskStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tier.disk");
        let store = DiskStore::mount(&path, DiskOptions::new(capacity)).unwrap();
        (dir, path, store)
    }

    #[test]
    fn an_extent_is_not_reused_while_a_read_pins_it() {
        let mut extents = Extents::new(4 * PAGE);
        let entry = chunk(1);
        let at = extents.allocate(PAGE).unwrap();
        extents.name(entry, at);
        extents.unpin(at);

        let (pinned, len) = extents.pin_named(&entry).unwrap();
        assert_eq!((pinned, len), (at, PAGE));
        assert!(extents.remove(&entry));
        let others: Vec<u64> = (0..3).map(|_| extents.allocate(PAGE).unwrap()).collect();
        assert!(!others.contains(&at), "a pinned extent was handed out");
        assert_eq!(
            extents.allocate(PAGE),
            None,
            "the pinned extent is still held"
        );

        extents.unpin(pinned);
        assert_eq!(extents.allocate(PAGE), Some(at));
    }

    #[test]
    fn a_shared_extent_is_freed_only_once_no_entry_names_it() {
        let mut extents = Extents::new(PAGE);
        let at = extents.allocate(PAGE).unwrap();
        extents.name(batch(chunk(1), 0), at);
        extents.name(batch(chunk(1), 1), at);
        extents.unpin(at);
        assert!(extents.remove(&batch(chunk(1), 0)));
        assert_eq!(extents.allocate(PAGE), None);
        assert!(extents.remove(&batch(chunk(1), 1)));
        assert_eq!(extents.used(), 0);
        assert_eq!(extents.allocate(PAGE), Some(at));
    }

    #[tokio::test]
    async fn a_chunk_is_read_whole_in_one_object() {
        let (_dir, _, store) = mounted(1 << 20);
        let key = ObjectKey::new(chunk(1), 7);
        store
            .put_chunk(key, batches(chunk(1), 3, 100))
            .await
            .unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.used_bytes(), PAGE);

        let object = store.get_chunk(batch(chunk(1), 2), key).await.unwrap();
        let read: Vec<(EntryID, u8, Bytes)> = object.batches().collect();
        assert_eq!(read.len(), 3);
        for (index, (entry, form, bytes)) in read.into_iter().enumerate() {
            assert_eq!((entry, form), (batch(chunk(1), index), 0));
            assert_eq!(bytes, Bytes::from(vec![index as u8 + 1; 100]));
        }
        assert_eq!(store.mismatches(), 0);
    }

    #[tokio::test]
    async fn a_header_that_does_not_match_is_a_miss() {
        let (_dir, path, store) = mounted(1 << 20);
        let key = ObjectKey::new(chunk(1), 7);
        let entry = batch(chunk(1), 0);
        store
            .put_chunk(key, batches(chunk(1), 2, 100))
            .await
            .unwrap();

        // Another chunk's key: the header names chunk 1.
        let other = ObjectKey::new(chunk(2), 7);
        assert!(matches!(
            store.get_chunk(entry, other).await,
            Err(DiskError::Corrupt)
        ));
        assert_eq!(store.mismatches(), 1);
        // Another identity: the key changed hands, a plain miss.
        let foreign = ObjectKey::new(chunk(1), 8);
        assert!(matches!(
            store.get_chunk(entry, foreign).await,
            Err(DiskError::Foreign)
        ));
        assert_eq!(store.mismatches(), 1);
        // An entry no object is named for.
        assert!(matches!(
            store.get_chunk(batch(chunk(1), 5), key).await,
            Err(DiskError::NotFound)
        ));

        // The object is the file's first extent: its header at offset 0,
        // the payload length at 24, the payload from 64.
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all_at(&u64::MAX.to_le_bytes(), 24).unwrap();
        assert!(matches!(
            store.get_chunk(entry, key).await,
            Err(DiskError::Corrupt)
        ));
        assert_eq!(store.mismatches(), 2);

        let (_dir, path, store) = mounted(1 << 20);
        store
            .put_chunk(key, batches(chunk(1), 2, 100))
            .await
            .unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all_at(&[0xEE], Header::LEN as u64 + 200)
            .unwrap();
        assert!(matches!(
            store.get_chunk(entry, key).await,
            Err(DiskError::Corrupt)
        ));
        assert_eq!(store.mismatches(), 1);
    }

    #[tokio::test]
    async fn a_failing_file_fails_the_call_and_keeps_what_the_store_held() {
        let dir = tempfile::tempdir().unwrap();
        let faults = Arc::new(Faults::default());
        let layered = Arc::clone(&faults);
        let options = DiskOptions::new(1 << 20).with_file(move |file| {
            Arc::new(Faulty {
                file,
                faults: Arc::clone(&layered),
            })
        });
        let store = DiskStore::mount(dir.path().join("tier.disk"), options).unwrap();
        let key = ObjectKey::new(chunk(1), 7);
        let entry = batch(chunk(1), 0);
        store
            .put_chunk(key, batches(chunk(1), 2, 100))
            .await
            .unwrap();

        // A write or its sync that fails names nothing and frees its extent.
        for flag in [&faults.writes, &faults.syncs] {
            flag.store(true, Ordering::Relaxed);
            let other = ObjectKey::new(chunk(2), 7);
            let put = store.put_chunk(other, batches(chunk(2), 1, 100)).await;
            assert!(matches!(put, Err(DiskError::Io(_))), "{put:?}");
            flag.store(false, Ordering::Relaxed);
            assert_eq!(store.len(), 2);
            assert_eq!(store.used_bytes(), PAGE);
        }
        let error = store
            .put_chunk(key, batches(chunk(1), 2, 100))
            .await
            .map(|_| ());
        assert!(error.is_ok());

        // A read that fails is an I/O error, not a mismatch.
        faults.reads.store(true, Ordering::Relaxed);
        let read = store.get_chunk(entry, key).await;
        match read {
            Err(DiskError::Io(error)) => assert_eq!(error.raw_os_error(), Some(libc::EIO)),
            other => panic!("expected EIO, got {:?}", other.map(|_| ())),
        }
        assert_eq!(store.mismatches(), 0);
        faults.reads.store(false, Ordering::Relaxed);
        assert!(store.get_chunk(entry, key).await.is_ok());
    }

    /// The real file, whose writeback waits until `open` is set.
    #[derive(Debug)]
    struct SlowWriteback {
        file: FileIo,
        open: Arc<AtomicBool>,
    }

    impl DiskFile for SlowWriteback {
        fn read_exact_at(&self, buffer: &mut [u8], at: u64) -> std::io::Result<()> {
            self.file.read_exact_at(buffer, at)
        }

        fn write_all_at(&self, buffer: &[u8], at: u64) -> std::io::Result<()> {
            self.file.write_all_at(buffer, at)
        }

        fn sync(&self, at: u64, len: u64) -> std::io::Result<()> {
            while !self.open.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            self.file.sync(at, len)
        }
    }

    #[tokio::test]
    async fn a_read_never_queues_behind_a_write_s_writeback() {
        let dir = tempfile::tempdir().unwrap();
        let open = Arc::new(AtomicBool::new(true));
        let gate = Arc::clone(&open);
        let options = DiskOptions::new(1 << 20)
            .with_threads(1)
            .with_writers(1)
            .with_file(move |file| {
                Arc::new(SlowWriteback {
                    file,
                    open: Arc::clone(&gate),
                })
            });
        let store = DiskStore::mount(dir.path().join("tier.disk"), options).unwrap();
        let key = ObjectKey::new(chunk(1), 7);
        store
            .put_chunk(key, batches(chunk(1), 2, 100))
            .await
            .unwrap();

        // A spill stuck in writeback holds the one writer thread.
        open.store(false, Ordering::Release);
        let writing = store.clone();
        let spill = tokio::spawn(async move {
            let other = ObjectKey::new(chunk(2), 7);
            writing.put_chunk(other, batches(chunk(2), 1, 100)).await
        });
        // Let the spill reach its thread first.
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        // The read thread is free: the read completes while it waits.
        let read = store.get_chunk(batch(chunk(1), 0), key).await;
        assert!(read.is_ok());
        assert!(!spill.is_finished());
        open.store(true, Ordering::Release);
        spill.await.unwrap().unwrap();
    }

    /// The real file, whose reads wait until `open` is set.
    #[derive(Debug)]
    struct SlowRead {
        file: FileIo,
        open: Arc<AtomicBool>,
        reading: Arc<AtomicBool>,
    }

    impl DiskFile for SlowRead {
        fn read_exact_at(&self, buffer: &mut [u8], at: u64) -> std::io::Result<()> {
            self.reading.store(true, Ordering::Release);
            while !self.open.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            self.file.read_exact_at(buffer, at)
        }

        fn write_all_at(&self, buffer: &[u8], at: u64) -> std::io::Result<()> {
            self.file.write_all_at(buffer, at)
        }

        fn sync(&self, at: u64, len: u64) -> std::io::Result<()> {
            self.file.sync(at, len)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_read_keeps_its_extent_until_its_pread_ends() {
        let dir = tempfile::tempdir().unwrap();
        let (open, reading) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        );
        let (gate, seen) = (Arc::clone(&open), Arc::clone(&reading));
        let options = DiskOptions::new(2 * PAGE).with_file(move |file| {
            Arc::new(SlowRead {
                file,
                open: Arc::clone(&gate),
                reading: Arc::clone(&seen),
            })
        });
        let store = DiskStore::mount(dir.path().join("tier.disk"), options).unwrap();
        let first = ObjectKey::new(chunk(1), 7);
        store
            .put_chunk(first, batches(chunk(1), 1, 100))
            .await
            .unwrap();

        // A read starts its pread, then its caller drops it.
        let pinned = store.clone();
        let read = tokio::spawn(async move { pinned.get_chunk(batch(chunk(1), 0), first).await });
        for _ in 0..1_000_000 {
            if reading.load(Ordering::Acquire) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            reading.load(Ordering::Acquire),
            "the read never reached its pread"
        );
        read.abort();
        assert!(read.await.unwrap_err().is_cancelled());
        assert!(store.remove(&batch(chunk(1), 0)));

        // Puts race the read: the one free page goes, the pinned one never.
        let second = ObjectKey::new(chunk(2), 7);
        store
            .put_chunk(second, batches(chunk(2), 1, 100))
            .await
            .unwrap();
        let third = ObjectKey::new(chunk(3), 7);
        assert!(matches!(
            store.put_chunk(third, batches(chunk(3), 1, 100)).await,
            Err(DiskError::Full)
        ));
        assert_eq!(store.used_bytes(), 2 * PAGE);

        // Once the pread ends, the pin goes and the extent is free.
        open.store(true, Ordering::Release);
        for _ in 0..1_000_000 {
            if store.used_bytes() == PAGE {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(store.used_bytes(), PAGE, "the dropped read's pin stayed");
        store
            .put_chunk(third, batches(chunk(3), 1, 100))
            .await
            .unwrap();
        let object = store.get_chunk(batch(chunk(2), 0), second).await.unwrap();
        let (_, _, bytes) = object.batches().next().unwrap();
        assert_eq!(bytes, Bytes::from(vec![1; 100]));
    }

    #[tokio::test]
    async fn a_full_store_refuses_a_put_and_keeps_what_it_held() {
        let (_dir, _, store) = mounted(2 * PAGE);
        let key = ObjectKey::new(chunk(1), 7);
        store
            .put_chunk(key, batches(chunk(1), 1, 100))
            .await
            .unwrap();
        // Three pages do not fit the one page left.
        let big = ObjectKey::new(chunk(2), 7);
        assert!(matches!(
            store
                .put_chunk(big, batches(chunk(2), 1, 2 * PAGE as usize))
                .await,
            Err(DiskError::Full)
        ));
        assert_eq!(store.used_bytes(), PAGE);
        assert_eq!(store.len(), 1);
        assert!(store.get_chunk(batch(chunk(1), 0), key).await.is_ok());
    }

    #[test]
    fn a_payload_that_is_not_a_directory_is_corrupt() {
        for payload in [
            &[][..],
            &[1, 0, 0, 0][..],
            &[0xFF; 64][..],
            // One batch claimed, its range past the payload's end.
            &[
                1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0,
                0, 0, 0, 0, 0, 0xFF, 0, 0, 0, 0, 0, 0, 0,
            ][..],
        ] {
            assert!(matches!(Directory::read(payload), Err(DiskError::Corrupt)));
        }
    }
}
