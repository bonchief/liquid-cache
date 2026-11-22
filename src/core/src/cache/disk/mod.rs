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

use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::cache::utils::EntryID;
use chunk::Directory;
pub use chunk::{ChunkObject, StoredBatch};
use extents::{Extents, PAGE};
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

/// How a [`DiskStore`] is mounted.
#[derive(Clone, Copy, Debug)]
pub struct DiskOptions {
    capacity: u64,
    direct: bool,
    threads: usize,
}

impl DiskOptions {
    /// What a store mounted with [`Default`] may hold.
    pub const DEFAULT_CAPACITY: u64 = 16 << 30;

    /// A store of at most `capacity` bytes (rounded down to whole pages),
    /// read and written through the page cache, on four I/O threads.
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            direct: false,
            threads: 4,
        }
    }

    /// Bypass the page cache: `O_DIRECT` on Linux, `F_NOCACHE` on macOS.
    /// Filesystems that refuse `O_DIRECT`, such as tmpfs on older kernels,
    /// then fail the mount. Buffered, reads keep their pages cached and
    /// every write is written back and dropped from the cache (Linux only).
    pub fn with_direct(mut self, direct: bool) -> Self {
        self.direct = direct;
        self
    }

    /// The most reads and writes in flight at once.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
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
    pool: Arc<Pool>,
}

#[derive(Debug)]
struct Shared {
    file: File,
    direct: bool,
    extents: Mutex<Extents>,
    mismatches: AtomicU64,
}

impl DiskStore {
    /// Mount an empty store at `path`, creating the file or discarding what
    /// it held.
    pub fn mount(path: impl AsRef<Path>, options: DiskOptions) -> Result<Self, DiskError> {
        let file = Shared::open(path.as_ref(), options.direct)?;
        let extents = Extents::new(options.capacity);
        file.set_len(extents.capacity())?;
        let pool = Pool::start(options.threads)?;
        Ok(Self {
            shared: Arc::new(Shared {
                file,
                direct: options.direct,
                extents: Mutex::new(extents),
                mismatches: AtomicU64::new(0),
            }),
            pool: Arc::new(pool),
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
        self.pool
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
        let (at, len) = self
            .shared
            .extents()
            .pin_named(&entry)
            .ok_or(DiskError::NotFound)?;
        let hold = Hold {
            shared: Arc::clone(&self.shared),
            at,
        };
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
        Ok(ChunkObject::new(payload, directory))
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
    /// object asked for.
    pub fn mismatches(&self) -> u64 {
        self.shared.mismatches.load(Ordering::Relaxed)
    }
}

impl Shared {
    fn open(path: &Path, direct: bool) -> Result<File, DiskError> {
        let mut open = std::fs::OpenOptions::new();
        open.read(true).write(true).create(true).truncate(true);
        #[cfg(target_os = "linux")]
        if direct {
            use std::os::unix::fs::OpenOptionsExt;
            open.custom_flags(libc::O_DIRECT);
        }
        let file = open.open(path)?;
        #[cfg(target_os = "macos")]
        if direct {
            use std::os::fd::AsRawFd;
            // SAFETY: `file` was just opened, so its descriptor is valid, and
            // the call reads no memory of this process.
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = direct;
        Ok(file)
    }

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
        if !self.direct {
            io::drop_written_pages(&self.file, at, len as u64)?;
        }
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
