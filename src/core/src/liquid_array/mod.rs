//! Vortex-backed encoded array used by LiquidCache.

mod array;

use std::sync::Arc;

use arrow::{
    array::{ArrayRef, BooleanArray, cast::AsArray},
    record_batch::RecordBatch,
};
use arrow_schema::{ArrowError, Field, Schema};

use crate::cache::LiquidExpr;

pub use array::LiquidArray;

/// A reference to a Liquid array.
pub type LiquidArrayRef = Arc<LiquidArray>;

/// Evaluate a validated Liquid predicate on an Arrow array. A predicate that
/// does not evaluate on it, or does not yield booleans, is an error.
pub fn eval_predicate_on_array(
    array: ArrayRef,
    predicate: &LiquidExpr,
) -> Result<BooleanArray, ArrowError> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "liquid_predicate_col",
        array.data_type().clone(),
        true,
    )]));
    let record_batch = RecordBatch::try_new(schema, vec![array])?;
    let external =
        |error: datafusion_common::DataFusionError| ArrowError::ExternalError(Box::new(error));
    let result = predicate
        .physical_expr()
        .evaluate(&record_batch)
        .map_err(external)?;
    let array = result
        .into_array(record_batch.num_rows())
        .map_err(external)?;
    array.as_boolean_opt().cloned().ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!(
            "a Liquid predicate yielded {}, not booleans",
            array.data_type()
        ))
    })
}
