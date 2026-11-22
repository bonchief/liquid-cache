//! The store's file and its I/O threads, and the page-aligned buffers they
//! read and write.

use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError, mpsc};

use bytes::Bytes;

use super::DiskError;
use super::extents::PAGE;

type Job = Box<dyn FnOnce() + Send>;

/// The positioned I/O a [`super::DiskStore`] does on its file, and nothing
/// else: every read and write of an object goes through it, on the store's
/// I/O threads.
///
/// [`FileIo`] is the store's real file and what [`super::DiskOptions`] mounts
/// by default; [`super::DiskOptions::with_file`] puts another implementation
/// in front of it. Whatever error an implementation returns takes the path a
/// failing `pread`, `pwrite` or writeback of the real file takes
/// ([`super::DiskError::Io`]), so an implementation that returns `EIO` or
/// `ENOSPC` exercises exactly what a failing disk does.
pub trait DiskFile: Send + Sync + std::fmt::Debug {
    /// Fill `buffer` from the file's bytes at `at`, wholly or fail.
    fn read_exact_at(&self, buffer: &mut [u8], at: u64) -> std::io::Result<()>;

    /// Write all of `buffer` into the file at `at`, or fail.
    fn write_all_at(&self, buffer: &[u8], at: u64) -> std::io::Result<()>;

    /// Called once each object's write of `len` bytes at `at` has succeeded,
    /// before the object is named: settle the write. Not a durability
    /// barrier; see [`FileIo`].
    fn sync(&self, at: u64, len: u64) -> std::io::Result<()>;
}

/// The store's real file: `pread` and `pwrite` on one descriptor.
///
/// Its [`DiskFile::sync`] writes a buffered write's dirty pages back and
/// drops them from the page cache (Linux only), so a write leaves nothing
/// cached; under direct I/O it does nothing. No metadata is synced: the
/// store is emptied at every start.
#[derive(Debug)]
pub struct FileIo {
    file: File,
    direct: bool,
}

impl FileIo {
    /// Open the file at `path`, creating it or discarding what it held:
    /// `O_DIRECT` on Linux and `F_NOCACHE` on macOS when `direct`.
    pub fn open(path: &Path, direct: bool) -> std::io::Result<Self> {
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
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(Self { file, direct })
    }

    /// Set the file's length: sparse, so it takes no blocks until written.
    pub(super) fn set_len(&self, len: u64) -> std::io::Result<()> {
        self.file.set_len(len)
    }
}

impl DiskFile for FileIo {
    fn read_exact_at(&self, buffer: &mut [u8], at: u64) -> std::io::Result<()> {
        self.file.read_exact_at(buffer, at)
    }

    fn write_all_at(&self, buffer: &[u8], at: u64) -> std::io::Result<()> {
        self.file.write_all_at(buffer, at)
    }

    fn sync(&self, at: u64, len: u64) -> std::io::Result<()> {
        if self.direct {
            return Ok(());
        }
        drop_written_pages(&self.file, at, len)
    }
}

/// A fixed set of threads that run the store's `pread`s, or its `pwrite`s
/// and their writeback: a store has one of each, so a write waiting on
/// writeback never holds up a read.
///
/// At most as many operations as there are threads touch the file at once;
/// the rest wait in the queue. io_uring is not used: the seccomp profile the
/// compute pods run under refuses it. The threads stop once every handle to
/// the pool is gone and the queue has drained.
#[derive(Debug)]
pub(super) struct Pool {
    jobs: mpsc::Sender<Job>,
}

impl Pool {
    pub(super) fn start(name: &str, threads: usize) -> std::io::Result<Self> {
        let (jobs, queue) = mpsc::channel::<Job>();
        let queue = Arc::new(Mutex::new(queue));
        for index in 0..threads.max(1) {
            let queue = Arc::clone(&queue);
            std::thread::Builder::new()
                .name(format!("{name}-{index}"))
                .spawn(move || Self::serve(&queue))?;
        }
        Ok(Self { jobs })
    }

    fn serve(queue: &Mutex<mpsc::Receiver<Job>>) {
        loop {
            // The queue's lock is let go before the job runs, so the threads
            // run their jobs side by side.
            let next = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
            match next {
                Ok(job) => job(),
                Err(_) => return,
            }
        }
    }

    /// Run `work` on one of the pool's threads.
    ///
    /// Dropping the returned future does not stop `work`; whatever `work`
    /// owns, such as an extent's pin, is held until it finishes.
    pub(super) async fn run<T: Send + 'static>(
        &self,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<T, DiskError> {
        let (sent, received) = futures::channel::oneshot::channel();
        self.jobs
            .send(Box::new(move || {
                let _ = sent.send(work());
            }))
            .map_err(|_| DiskError::Closed)?;
        received.await.map_err(|_| DiskError::Closed)
    }
}

/// A zeroed buffer of whole pages whose start is page-aligned, as `O_DIRECT`
/// reads and writes need.
pub(super) struct PageBuffer {
    bytes: Vec<u8>,
    start: usize,
    len: usize,
}

impl PageBuffer {
    const ALIGN: usize = PAGE as usize;

    pub(super) fn zeroed(len: usize) -> Self {
        let bytes = vec![0; len.saturating_add(Self::ALIGN)];
        // `align_offset` may report that it cannot align; the buffer is then
        // used unaligned, and an `O_DIRECT` call on it fails rather than
        // reading or writing out of bounds.
        let start = match bytes.as_ptr().align_offset(Self::ALIGN) {
            start if start < Self::ALIGN => start,
            _ => 0,
        };
        let len = len.min(bytes.len() - start);
        Self { bytes, start, len }
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        &self.bytes[self.start..self.start + self.len]
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes[self.start..self.start + self.len]
    }

    /// The bytes at `range` of the buffer, without copying them. `range`
    /// must lie within the buffer, as [`super::Header::check`] makes sure.
    pub(super) fn into_bytes(self, range: Range<usize>) -> Bytes {
        let end = self.start + range.end.min(self.len);
        let start = (self.start + range.start).min(end);
        Bytes::from(self.bytes).slice(start..end)
    }
}

/// Write the file's dirty pages at `at..at + len` back and drop them from
/// the page cache, so a buffered write leaves nothing cached. This waits for
/// writeback but is not a durability barrier: no metadata is synced, and the
/// store is emptied at every start anyway.
#[cfg(target_os = "linux")]
fn drop_written_pages(file: &File, at: u64, len: u64) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let (Ok(offset), Ok(length)) = (libc::off64_t::try_from(at), libc::off64_t::try_from(len))
    else {
        return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput));
    };
    let flags = libc::SYNC_FILE_RANGE_WAIT_BEFORE
        | libc::SYNC_FILE_RANGE_WRITE
        | libc::SYNC_FILE_RANGE_WAIT_AFTER;
    // SAFETY: the descriptor stays open for the life of `file`, and neither
    // call reads or writes memory of this process.
    unsafe {
        if libc::sync_file_range(file.as_raw_fd(), offset, length, flags) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // Advice only: pages it fails to drop cost cache, not correctness.
        libc::posix_fadvise(file.as_raw_fd(), offset, length, libc::POSIX_FADV_DONTNEED);
    }
    Ok(())
}

/// Elsewhere there is no portable way to drop written pages; buffered writes
/// stay cached. Only Linux serves the tier.
#[cfg(not(target_os = "linux"))]
fn drop_written_pages(_file: &File, _at: u64, _len: u64) -> std::io::Result<()> {
    Ok(())
}
