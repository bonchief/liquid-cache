//! Filling the cache with whole column chunks from their Parquet bytes, off
//! any query's read path.
//!
//! A filler fetches one column chunk's bytes ([`chunk_span`]), and
//! [`LiquidCacheParquet::fill`](super::LiquidCacheParquet::fill) decodes them
//! as the reader would, with the same schema coercion, encodes each batch on
//! the caller's thread and inserts the chunk whole.

use std::ops::Range;
use std::sync::Arc;

use arrow_schema::{ArrowError, SchemaRef};
use bytes::{Buf, Bytes};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::{ChunkReader, Length};

/// Why a fill inserted nothing.
#[derive(Debug)]
pub enum FillError {
    /// The metadata has no such row group or column, or its column chunk's
    /// offsets do not describe a span of the file.
    NoSuchChunk,
    /// The bytes given are not the chunk's whole span ([`chunk_span`]).
    Partial,
    /// The bytes did not decode as the chunk.
    Decode(ParquetError),
    /// A decoded batch was not the one column asked for.
    Arrow(ArrowError),
    /// The cache refused the chunk; none of it was kept.
    Full,
    /// The file's entries are cached decoded to other types than `schema`
    /// decodes it to: a read registered it first with another table schema.
    Schema,
}

impl std::fmt::Display for FillError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchChunk => write!(out, "the file has no such column chunk"),
            Self::Partial => write!(out, "the bytes are not the column chunk's whole span"),
            Self::Decode(error) => write!(out, "the column chunk did not decode: {error}"),
            Self::Arrow(error) => write!(out, "a decoded batch was malformed: {error}"),
            Self::Full => write!(out, "the cache has no room for the column chunk"),
            Self::Schema => write!(
                out,
                "the file is cached decoded to other types than this fill's schema"
            ),
        }
    }
}

impl std::error::Error for FillError {}

/// The byte span, in the file, of column `column` (its index in the file's
/// Arrow schema) of row group `row_group`: from the first byte of its first
/// leaf's pages to the last byte of its last leaf's. A nested column is
/// several column chunks, one per leaf, and the span covers all of them.
/// `None` when the metadata has no such chunk or its offsets are negative or
/// overflow.
pub fn chunk_span(
    metadata: &ParquetMetaData,
    row_group: usize,
    column: usize,
) -> Option<Range<u64>> {
    let group = metadata.row_groups().get(row_group)?;
    let schema = metadata.file_metadata().schema_descr();
    let mut span: Option<Range<u64>> = None;
    for (leaf, chunk) in group.columns().iter().enumerate() {
        if leaf >= schema.num_columns() || schema.get_column_root_idx(leaf) != column {
            continue;
        }
        let start = chunk
            .dictionary_page_offset()
            .unwrap_or_else(|| chunk.data_page_offset());
        let start = u64::try_from(start).ok()?;
        let len = u64::try_from(chunk.compressed_size()).ok()?;
        let end = start.checked_add(len)?;
        span = Some(match span {
            Some(span) => span.start.min(start)..span.end.max(end),
            None => start..end,
        });
    }
    span
}

/// One column chunk's bytes, read by the Parquet reader as if they were the
/// file: offsets are file offsets, and only the span the bytes cover reads.
struct ChunkBytes {
    start: u64,
    bytes: Bytes,
}

impl ChunkBytes {
    /// The bytes from file offset `start` for `len` bytes, or an error if
    /// they lie outside the span held.
    fn slice(&self, start: u64, len: Option<usize>) -> Result<Bytes, ParquetError> {
        let outside = || {
            ParquetError::EOF(format!(
                "read at offset {start} lies outside the column chunk's {} bytes at {}",
                self.bytes.len(),
                self.start
            ))
        };
        let from = start
            .checked_sub(self.start)
            .and_then(|from| usize::try_from(from).ok())
            .filter(|from| *from <= self.bytes.len())
            .ok_or_else(outside)?;
        let to = match len {
            Some(len) => from
                .checked_add(len)
                .filter(|to| *to <= self.bytes.len())
                .ok_or_else(outside)?,
            None => self.bytes.len(),
        };
        Ok(self.bytes.slice(from..to))
    }
}

impl Length for ChunkBytes {
    fn len(&self) -> u64 {
        self.start.saturating_add(self.bytes.len() as u64)
    }
}

impl ChunkReader for ChunkBytes {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> Result<Self::T, ParquetError> {
        Ok(self.slice(start, None)?.reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes, ParquetError> {
        self.slice(start, Some(length))
    }
}

/// `schema` coerced as the reader coerces the file's: the file's own Arrow
/// schema with the table's view types where the table reads views, so the
/// cache holds the type a read asks for.
pub(crate) fn reader_metadata(
    table: &SchemaRef,
    metadata: Arc<ParquetMetaData>,
) -> Result<ArrowReaderMetadata, ParquetError> {
    let plain = ArrowReaderMetadata::try_new(Arc::clone(&metadata), ArrowReaderOptions::new())?;
    match datafusion::datasource::physical_plan::parquet::apply_file_schema_type_coercions(
        table,
        plain.schema(),
    ) {
        Some(merged) => ArrowReaderMetadata::try_new(
            metadata,
            ArrowReaderOptions::new().with_schema(Arc::new(merged)),
        ),
        None => Ok(plain),
    }
}

/// Every batch of column `column` of row group `row_group`, decoded from
/// `bytes`, the chunk's whole span, `batch_size` rows each.
pub(crate) fn decode(
    reader: &ArrowReaderMetadata,
    row_group: usize,
    column: usize,
    bytes: Bytes,
    batch_size: usize,
) -> Result<Vec<arrow::array::ArrayRef>, FillError> {
    let span = chunk_span(reader.metadata(), row_group, column).ok_or(FillError::NoSuchChunk)?;
    if span.end - span.start != bytes.len() as u64 {
        return Err(FillError::Partial);
    }
    let input = ChunkBytes {
        start: span.start,
        bytes,
    };
    let mask = ProjectionMask::roots(reader.parquet_schema(), [column]);
    let batches = ParquetRecordBatchReaderBuilder::new_with_metadata(input, reader.clone())
        .with_projection(mask)
        .with_row_groups(vec![row_group])
        .with_batch_size(batch_size)
        .build()
        .map_err(FillError::Decode)?;
    let mut arrays = Vec::new();
    for batch in batches {
        let batch = batch.map_err(FillError::Arrow)?;
        let array = batch.columns().first().cloned().ok_or_else(|| {
            FillError::Arrow(ArrowError::SchemaError(
                "a decoded batch holds no column".to_string(),
            ))
        })?;
        arrays.push(array);
    }
    Ok(arrays)
}
