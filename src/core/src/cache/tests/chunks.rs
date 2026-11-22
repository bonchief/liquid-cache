//! Column chunks move between tiers whole: evicted, written and dropped
//! together, never one batch alone.

use crate::cache::{
    CachedBatchType, DiskOptions, DiskStore, EntryID, Evict, LiquidCache, LiquidCacheBuilder,
    LiquidPolicy,
};
use arrow::array::{Array, ArrayRef, Int64Array};
use std::sync::Arc;

/// Batch bits, as the Parquet layer packs entry ids.
const CHUNK_BITS: u32 = 16;
const BATCHES: usize = 3;

fn entry(chunk: usize, batch: usize) -> EntryID {
    EntryID::from((chunk << CHUNK_BITS) | batch)
}

/// An array of 1,024 values, 8 KiB in memory as Arrow.
fn array(seed: i64) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(
        (0..1024).map(|value| value * 7919 + seed),
    ))
}

fn batch_bytes() -> usize {
    array(0).get_array_memory_size()
}

/// A cache holding `BATCHES` batches in memory, and `disk` bytes on disk,
/// that moves Arrow batches straight to disk.
async fn cache(disk: usize, store: Option<DiskStore>) -> Arc<LiquidCache> {
    let builder = LiquidCacheBuilder::new()
        .with_max_memory_bytes(BATCHES * batch_bytes() + batch_bytes() / 2)
        .with_max_disk_bytes(disk)
        .with_chunk_bits(CHUNK_BITS)
        .with_eviction_policy(Box::new(Evict))
        .with_cache_policy(Box::new(LiquidPolicy::new()));
    match store {
        Some(store) => builder.with_store(store).build().await,
        None => builder.build().await,
    }
}

async fn insert_chunk(cache: &Arc<LiquidCache>, chunk: usize) {
    for batch in 0..BATCHES {
        cache
            .insert(entry(chunk, batch), 0, array((chunk * 10 + batch) as i64))
            .await
            .unwrap();
    }
}

/// Where each batch of `chunk` is: `Some(true)` on disk, `Some(false)` in
/// memory, `None` not held.
fn tiers(cache: &LiquidCache, chunk: usize) -> Vec<Option<bool>> {
    (0..BATCHES)
        .map(|batch| {
            let entry = entry(chunk, batch);
            cache
                .contains(&entry, 0)
                .then(|| cache.is_on_disk(&entry, 0))
        })
        .collect()
}

#[tokio::test]
async fn a_chunk_is_spilled_whole_in_one_object() {
    let cache = cache(1 << 30, None).await;
    insert_chunk(&cache, 1).await;
    assert_eq!(tiers(&cache, 1), vec![Some(false); BATCHES]);

    // Room for one batch more takes all of chunk 1 to disk, not one batch.
    cache.insert(entry(2, 0), 0, array(20)).await.unwrap();
    assert_eq!(tiers(&cache, 1), vec![Some(true); BATCHES]);
    assert!(!cache.is_on_disk(&entry(2, 0), 0));
    let counters = cache.counters();
    assert_eq!(counters.chunks_spilled, 1);
    assert_eq!(counters.disk_arrow.entries, BATCHES);
    assert_eq!(counters.chunks.disk, 1);

    // One read brings back every batch of the chunk, and moves nothing.
    let read = cache.read_chunk(&entry(1, 1), 0).await;
    assert_eq!(read.len(), BATCHES);
    for (id, batch) in read {
        assert_eq!(
            CachedBatchType::from(batch.as_ref()),
            CachedBatchType::MemoryArrow
        );
        let index = usize::from(id) & ((1 << CHUNK_BITS) - 1);
        let held = cache.get(&id, 0).read().await.unwrap();
        assert_eq!(held.as_ref(), array((10 + index) as i64).as_ref());
    }
    assert_eq!(tiers(&cache, 1), vec![Some(true); BATCHES]);
}

#[tokio::test]
async fn a_chunk_is_dropped_from_disk_whole() {
    // Measure what one chunk's object is charged, then give the disk tier
    // exactly that.
    let measure = cache(1 << 30, None).await;
    insert_chunk(&measure, 1).await;
    measure.insert(entry(2, 0), 0, array(20)).await.unwrap();
    let one_chunk = measure.stats().disk_usage_bytes;
    assert!(one_chunk > 0);

    let cache = cache(one_chunk, None).await;
    insert_chunk(&cache, 1).await;
    insert_chunk(&cache, 2).await;
    assert_eq!(tiers(&cache, 1), vec![Some(true); BATCHES]);
    // Chunk 2 spills in its turn; chunk 1 leaves the disk whole for it.
    insert_chunk(&cache, 3).await;
    assert_eq!(tiers(&cache, 1), vec![None; BATCHES]);
    assert_eq!(tiers(&cache, 2), vec![Some(true); BATCHES]);
    assert_eq!(tiers(&cache, 3), vec![Some(false); BATCHES]);
    assert_eq!(cache.counters().chunks_dropped_from_disk, 1);
    assert!(cache.stats().disk_usage_bytes <= one_chunk);
}

#[tokio::test]
async fn a_full_disk_drops_the_chunk_from_memory() {
    let cache = cache(1, None).await;
    insert_chunk(&cache, 1).await;
    cache.insert(entry(2, 0), 0, array(20)).await.unwrap();
    assert_eq!(tiers(&cache, 1), vec![None; BATCHES]);
    assert_eq!(tiers(&cache, 2)[0], Some(false));
    let counters = cache.counters();
    assert_eq!(counters.chunks_dropped_from_memory, 1);
    assert_eq!(counters.chunks_spilled, 0);
    assert_eq!(cache.stats().disk_usage_bytes, 0);
    assert!(cache.budget().memory_usage_bytes() <= cache.config().max_memory_bytes());
}

#[tokio::test]
async fn a_failed_write_drops_the_chunk_and_gives_its_disk_back() {
    // The budget allows the write; the store's file, one page long, cannot
    // take the object.
    let directory = tempfile::tempdir().unwrap();
    let store =
        DiskStore::mount(directory.path().join("tier.disk"), DiskOptions::new(4096)).unwrap();
    let cache = cache(1 << 30, Some(store.clone())).await;
    insert_chunk(&cache, 1).await;
    cache.insert(entry(2, 0), 0, array(20)).await.unwrap();

    assert_eq!(tiers(&cache, 1), vec![None; BATCHES]);
    assert_eq!(tiers(&cache, 2)[0], Some(false));
    assert_eq!(cache.stats().runtime.disk_write_failures, 1);
    assert_eq!(cache.stats().disk_usage_bytes, 0);
    assert!(store.is_empty());
    assert_eq!(store.used_bytes(), 0);

    // The cache goes on working after the failure.
    cache.insert(entry(2, 1), 0, array(21)).await.unwrap();
    let held = cache.get(&entry(2, 1), 0).read().await.unwrap();
    assert_eq!(held.as_ref(), array(21).as_ref());
}

/// A store whose real file fails every write with `ENOSPC` while `writes`
/// is set, and every read with `EIO` while `reads` is.
fn faulty_store(
    directory: &std::path::Path,
) -> (
    DiskStore,
    Arc<std::sync::atomic::AtomicBool>,
    Arc<std::sync::atomic::AtomicBool>,
) {
    use crate::cache::{DiskFile, FileIo};
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct Faulty {
        file: FileIo,
        writes: Arc<AtomicBool>,
        reads: Arc<AtomicBool>,
    }

    fn fail(flag: &AtomicBool, code: i32) -> std::io::Result<()> {
        match flag.load(Ordering::Relaxed) {
            true => Err(std::io::Error::from_raw_os_error(code)),
            false => Ok(()),
        }
    }

    impl DiskFile for Faulty {
        fn read_exact_at(&self, buffer: &mut [u8], at: u64) -> std::io::Result<()> {
            fail(&self.reads, libc::EIO)?;
            self.file.read_exact_at(buffer, at)
        }

        fn write_all_at(&self, buffer: &[u8], at: u64) -> std::io::Result<()> {
            fail(&self.writes, libc::ENOSPC)?;
            self.file.write_all_at(buffer, at)
        }

        fn sync(&self, at: u64, len: u64) -> std::io::Result<()> {
            self.file.sync(at, len)
        }
    }

    let (writes, reads) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let (w, r) = (Arc::clone(&writes), Arc::clone(&reads));
    let options = DiskOptions::new(1 << 20).with_file(move |file| {
        Arc::new(Faulty {
            file,
            writes: Arc::clone(&w),
            reads: Arc::clone(&r),
        })
    });
    let store = DiskStore::mount(directory.join("tier.disk"), options).unwrap();
    (store, writes, reads)
}

#[tokio::test]
async fn an_injected_disk_fault_takes_a_real_fault_s_path() {
    use std::sync::atomic::Ordering;
    let directory = tempfile::tempdir().unwrap();
    let (store, writes, _) = faulty_store(directory.path());
    let cache = cache(1 << 30, Some(store.clone())).await;

    // ENOSPC on the spill: the chunk is dropped and its disk given back.
    writes.store(true, Ordering::Relaxed);
    insert_chunk(&cache, 1).await;
    cache.insert(entry(2, 0), 0, array(20)).await.unwrap();
    assert_eq!(tiers(&cache, 1), vec![None; BATCHES]);
    assert_eq!(cache.stats().runtime.disk_write_failures, 1);
    assert_eq!(cache.stats().disk_usage_bytes, 0);
    assert!(store.is_empty());
    writes.store(false, Ordering::Relaxed);

    // EIO on a read of a spilled chunk: a miss, counted, and the chunk's
    // record dropped; the process and the cache go on.
    let directory = tempfile::tempdir().unwrap();
    let (store, _, reads) = faulty_store(directory.path());
    let cache = self::cache(1 << 30, Some(store)).await;
    insert_chunk(&cache, 1).await;
    cache.insert(entry(2, 0), 0, array(20)).await.unwrap();
    assert_eq!(tiers(&cache, 1), vec![Some(true); BATCHES]);
    let (spilled, on_disk) = (1, 1);
    reads.store(true, Ordering::Relaxed);
    assert!(
        cache
            .read_chunk(&entry(spilled, on_disk), 0)
            .await
            .is_empty()
    );
    assert_eq!(cache.stats().runtime.disk_read_failures, 1);
    assert!(!cache.contains(&entry(spilled, on_disk), 0));
    reads.store(false, Ordering::Relaxed);
    cache.insert(entry(4, 0), 0, array(40)).await.unwrap();
    let held = cache.get(&entry(4, 0), 0).read().await.unwrap();
    assert_eq!(held.as_ref(), array(40).as_ref());
}
