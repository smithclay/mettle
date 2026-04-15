//! Shared leaf helpers: column/literal conversions, filter application,
//! expression evaluation, CPU-side record-batch sorting, and the
//! `ColumnLookup`/`JoinResult` late-materialization types.

use std::error::Error;
use std::sync::Arc;

use duckdb::arrow::array::{
    ArrayRef, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use duckdb::arrow::datatypes::DataType;
use duckdb::arrow::record_batch::RecordBatch;

use crate::config::BufferConfig;
use crate::metal::{self, GpuVector, ScalarCompareOp, ScalarProjectOp};
use crate::pinned::{PinnedColumn, PinnedTable};

use super::ir::{AggFunc, AggregateStateKind, GpuBinOp, GpuExpr, GpuFilter, GpuLiteral, GpuOrderBy};

/// Abstraction for looking up pinned columns by name. Each column comes
/// with its own index array for late materialization — the caller gathers
/// from the original pinned data using these indices.
pub(super) trait ColumnLookup {
    fn get_column(&self, name: &str) -> Option<(&PinnedColumn, &[usize])>;
    fn row_count(&self) -> usize;
    fn gpu_relation_bindings(&self) -> Option<(&[(String, Arc<PinnedTable>)], &[Vec<usize>])> {
        None
    }
}

/// Late-materialized join result: references to original pinned tables +
/// per-table row index arrays. All index arrays have the same length
/// (the join result cardinality). Columns are only gathered at the point
/// of consumption (GPU kernels, output construction).
pub(super) struct JoinResult {
    /// (alias, original pinned table reference) per source table.
    pub(super) tables: Vec<(String, Arc<PinnedTable>)>,
    /// Per-table row index arrays. `table_indices[t][i]` is the row index
    /// into `tables[t].1` for result row `i`. All arrays have the same length.
    pub(super) table_indices: Vec<Vec<usize>>,
}

impl JoinResult {
    pub(super) fn row_count(&self) -> usize {
        self.table_indices.first().map(|v| v.len()).unwrap_or(0)
    }

    fn resolve_column(&self, name: &str) -> Option<(usize, &PinnedColumn)> {
        resolve_column_gpu(name, &self.tables)
    }
}

pub(super) struct JoinResultLookup<'a>(pub(super) &'a JoinResult);

impl ColumnLookup for JoinResultLookup<'_> {
    fn get_column(&self, name: &str) -> Option<(&PinnedColumn, &[usize])> {
        let (table_idx, col) = self.0.resolve_column(name)?;
        Some((col, &self.0.table_indices[table_idx]))
    }
    fn row_count(&self) -> usize {
        self.0.row_count()
    }
    fn gpu_relation_bindings(&self) -> Option<(&[(String, Arc<PinnedTable>)], &[Vec<usize>])> {
        Some((&self.0.tables, &self.0.table_indices))
    }
}

/// Resolve a column name to (table_index, column) across a set of table refs.
pub(super) fn resolve_column_gpu<'a>(
    name: &str,
    table_refs: &'a [(String, Arc<PinnedTable>)],
) -> Option<(usize, &'a PinnedColumn)> {
    let name_lower = name.to_lowercase();
    for (t, (alias, table)) in table_refs.iter().enumerate() {
        let prefix = format!("{}.", alias.to_lowercase());
        if let Some(col_name) = name_lower.strip_prefix(&prefix) {
            if let Some(col) = table.column(col_name) {
                return Some((t, col));
            }
        }
        if let Some(col) = table.column(&name_lower) {
            return Some((t, col));
        }
    }
    None
}

pub(super) fn aggregate_state_kind(func: AggFunc) -> AggregateStateKind {
    match func {
        AggFunc::Count => AggregateStateKind::CountU64,
        AggFunc::Sum => AggregateStateKind::SumF64,
        AggFunc::Min => AggregateStateKind::MinF64,
        AggFunc::Max => AggregateStateKind::MaxF64,
        AggFunc::Avg => AggregateStateKind::AvgSumCount,
    }
}

/// Gather a column as f32 GpuVector through a table's index array.
pub(super) fn gather_column_as_f32_gv(
    col: &PinnedColumn,
    indices: &GpuVector,
    config: &BufferConfig,
) -> Result<Option<GpuVector>, Box<dyn Error>> {
    let col_gpu = metal::column_to_gpu_buffer(col, config)?;
    match col {
        PinnedColumn::Float32(_) | PinnedColumn::Float64(_) => {
            Ok(Some(metal::gather_f32_gv(&col_gpu, indices, config)?))
        }
        PinnedColumn::Int32(_) | PinnedColumn::Date32(_) | PinnedColumn::Int64(_) => Ok(Some(
            metal::gather_i32_as_f32_gv(&col_gpu, indices, config)?,
        )),
        _ => Ok(None),
    }
}

pub(super) fn gather_group_key_as_i32_gv(
    col: &PinnedColumn,
    indices: &GpuVector,
    config: &BufferConfig,
) -> Result<Option<GpuVector>, Box<dyn Error>> {
    let col_gpu = metal::column_to_gpu_buffer(col, config)?;
    match col {
        PinnedColumn::Int32(_) | PinnedColumn::Date32(_) | PinnedColumn::DictionaryI32 { .. } => {
            Ok(Some(metal::gather_i32_gv(&col_gpu, indices, config)?))
        }
        PinnedColumn::Int64(_) | PinnedColumn::Float32(_) | PinnedColumn::Float64(_) => Ok(None),
    }
}

/// Recursively evaluate a GpuExpr on GPU, returning a GpuVector of f32 values.
pub(super) fn evaluate_expr_gv(
    expr: &GpuExpr,
    table_refs: &[(String, Arc<PinnedTable>)],
    gv_indices: &[GpuVector],
    config: &BufferConfig,
) -> Result<Option<GpuVector>, Box<dyn Error>> {
    match expr {
        GpuExpr::Column(name) => {
            let (tidx, col) = match resolve_column_gpu(name, table_refs) {
                Some(r) => r,
                None => return Ok(None),
            };
            gather_column_as_f32_gv(col, &gv_indices[tidx], config)
        }
        GpuExpr::BinaryOp { op, left, right } => {
            let metal_op = match op {
                GpuBinOp::Add => ScalarProjectOp::Add,
                GpuBinOp::Mul => ScalarProjectOp::Multiply,
                GpuBinOp::Sub => ScalarProjectOp::Subtract,
            };
            // Fast path: Column op Literal or Literal op Column
            if let (GpuExpr::Column(_), GpuExpr::Literal(lit)) = (left.as_ref(), right.as_ref()) {
                let left_gv = match evaluate_expr_gv(left, table_refs, gv_indices, config)? {
                    Some(gv) => gv,
                    None => return Ok(None),
                };
                return Ok(Some(metal::project_f32_scalar_gv(
                    &left_gv,
                    literal_as_f32(lit),
                    metal_op,
                    config,
                )?));
            }
            if let (GpuExpr::Literal(lit), GpuExpr::Column(_)) = (left.as_ref(), right.as_ref()) {
                let right_gv = match evaluate_expr_gv(right, table_refs, gv_indices, config)? {
                    Some(gv) => gv,
                    None => return Ok(None),
                };
                let flipped_op = match op {
                    GpuBinOp::Sub => ScalarProjectOp::SubtractFrom,
                    _ => metal_op,
                };
                return Ok(Some(metal::project_f32_scalar_gv(
                    &right_gv,
                    literal_as_f32(lit),
                    flipped_op,
                    config,
                )?));
            }
            // General: evaluate both sides
            let left_gv = match evaluate_expr_gv(left, table_refs, gv_indices, config)? {
                Some(gv) => gv,
                None => return Ok(None),
            };
            let right_gv = match evaluate_expr_gv(right, table_refs, gv_indices, config)? {
                Some(gv) => gv,
                None => return Ok(None),
            };
            Ok(Some(metal::elementwise_f32_gv(
                &left_gv, &right_gv, metal_op, config,
            )?))
        }
        GpuExpr::AggCall { arg, .. } => evaluate_expr_gv(arg, table_refs, gv_indices, config),
        GpuExpr::Literal(_) => Ok(None), // Literals handled at BinaryOp level
    }
}

pub(super) fn upload_indices_as_gv(
    table_indices: &[Vec<usize>],
    config: &BufferConfig,
) -> Result<Vec<GpuVector>, Box<dyn Error>> {
    table_indices
        .iter()
        .map(|indices| {
            let gpu_indices: Vec<u32> = indices
                .iter()
                .map(|&idx| u32::try_from(idx).map_err(|_| "row index overflowed u32".to_owned()))
                .collect::<Result<_, _>>()?;
            metal::upload_gv(&gpu_indices, config).map_err(Box::<dyn Error>::from)
        })
        .collect()
}

pub(super) fn column_name_only(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

pub(super) fn refs_eq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// Recursively evaluate a GpuExpr against pinned columns, producing f64 values.
/// Column indices come from the lookup (late materialization).
pub(super) fn evaluate_expr(
    expr: &GpuExpr,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Vec<f64>, Box<dyn Error>> {
    match expr {
        GpuExpr::Column(col_name) => {
            let (col, indices) = lookup
                .get_column(col_name)
                .ok_or_else(|| format!("pinned table missing column {col_name:?}"))?;
            Ok(gather_column_as_f64(col, indices))
        }
        GpuExpr::Literal(lit) => {
            let val = literal_as_f64(lit);
            Ok(vec![val; lookup.row_count()])
        }
        GpuExpr::BinaryOp { op, left, right } => {
            if let Some(result) = try_scalar_projection(op, left, right, lookup, config)? {
                return Ok(result);
            }
            let left_vals = evaluate_expr(left, lookup, config)?;
            let right_vals = evaluate_expr(right, lookup, config)?;
            let metal_op = match op {
                GpuBinOp::Add => ScalarProjectOp::Add,
                GpuBinOp::Mul => ScalarProjectOp::Multiply,
                GpuBinOp::Sub => ScalarProjectOp::Subtract,
            };
            Ok(metal::elementwise_f64(
                &left_vals,
                &right_vals,
                metal_op,
                config,
            )?)
        }
        GpuExpr::AggCall { .. } => {
            Err("cannot evaluate aggregate call as a scalar expression".into())
        }
    }
}

pub(super) fn gather_column_as_f64(col: &PinnedColumn, indices: &[usize]) -> Vec<f64> {
    match col {
        PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
            indices.iter().map(|&i| v[i] as f64).collect()
        }
        PinnedColumn::Int64(v) => indices.iter().map(|&i| v[i] as f64).collect(),
        PinnedColumn::Float32(v) => indices.iter().map(|&i| v[i] as f64).collect(),
        PinnedColumn::Float64(v) => indices.iter().map(|&i| v[i]).collect(),
        PinnedColumn::DictionaryI32 { .. } => indices.iter().map(|_| 0.0).collect(),
    }
}

/// Fast path for BinaryOp(op, Column, Literal) or BinaryOp(op, Literal, Column).
/// Uses the scalar projection kernel directly (native f32, no literal broadcast,
/// no f64→f32→f64 round-trip for the literal side).
fn try_scalar_projection(
    op: &GpuBinOp,
    left: &GpuExpr,
    right: &GpuExpr,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Option<Vec<f64>>, Box<dyn Error>> {
    let (col_name, scalar_val, scalar_on_left) = match (left, right) {
        (GpuExpr::Column(c), GpuExpr::Literal(lit)) => (c.as_str(), literal_as_f64(lit), false),
        (GpuExpr::Literal(lit), GpuExpr::Column(c)) => (c.as_str(), literal_as_f64(lit), true),
        _ => return Ok(None),
    };

    let (col, indices) = match lookup.get_column(col_name) {
        Some(pair) => pair,
        None => return Ok(None),
    };

    let metal_op = match (op, scalar_on_left) {
        (GpuBinOp::Add, _) => ScalarProjectOp::Add,
        (GpuBinOp::Mul, _) => ScalarProjectOp::Multiply,
        (GpuBinOp::Sub, false) => ScalarProjectOp::Subtract,
        (GpuBinOp::Sub, true) => ScalarProjectOp::SubtractFrom,
    };

    let values_f32: Vec<f32> = match col {
        PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
            indices.iter().map(|&i| v[i] as f32).collect()
        }
        PinnedColumn::Int64(v) => indices.iter().map(|&i| v[i] as f32).collect(),
        PinnedColumn::Float32(v) => indices.iter().map(|&i| v[i]).collect(),
        PinnedColumn::Float64(v) => indices.iter().map(|&i| v[i] as f32).collect(),
        PinnedColumn::DictionaryI32 { .. } => return Ok(None),
    };

    let result_f32 = metal::project_f32_values(&values_f32, scalar_val as f32, metal_op, config)?;
    Ok(Some(result_f32.iter().map(|&v| v as f64).collect()))
}

pub(super) fn column_arrow_type(col: &PinnedColumn) -> DataType {
    match col {
        PinnedColumn::Int32(_) => DataType::Int32,
        PinnedColumn::Int64(_) => DataType::Int64,
        PinnedColumn::Float32(_) => DataType::Float32,
        PinnedColumn::Float64(_) => DataType::Float64,
        PinnedColumn::Date32(_) => DataType::Date32,
        PinnedColumn::DictionaryI32 { .. } => DataType::Utf8,
    }
}

pub(super) fn gather_column(col: &PinnedColumn, indices: &[usize]) -> ArrayRef {
    match col {
        PinnedColumn::Int32(v) => Arc::new(Int32Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::Int64(v) => Arc::new(Int64Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::Float32(v) => Arc::new(Float32Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::Float64(v) => Arc::new(Float64Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::Date32(v) => Arc::new(Date32Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::DictionaryI32 { codes, values, .. } => {
            let arr: Vec<&str> = indices
                .iter()
                .filter_map(|&i| {
                    usize::try_from(codes[i])
                        .ok()
                        .and_then(|code| values.get(code))
                        .map(String::as_str)
                })
                .collect();
            Arc::new(duckdb::arrow::array::StringArray::from(arr))
        }
    }
}

pub(super) fn literal_as_i32(lit: &GpuLiteral) -> Result<i32, Box<dyn Error>> {
    match lit {
        GpuLiteral::Int(i) => i32::try_from(*i).map_err(|_| "literal overflows i32".into()),
        GpuLiteral::Float(f) => Ok(*f as i32),
        GpuLiteral::Date(d) => Ok(*d),
        GpuLiteral::Str(_) => Err("cannot convert string to i32".into()),
    }
}

pub(super) fn literal_as_date32(lit: &GpuLiteral) -> Result<i32, Box<dyn Error>> {
    match lit {
        GpuLiteral::Date(d) => Ok(*d),
        GpuLiteral::Int(i) => i32::try_from(*i).map_err(|_| "literal overflows i32".into()),
        _ => Err("expected date or integer literal for date comparison".into()),
    }
}

pub(super) fn literal_as_f64(lit: &GpuLiteral) -> f64 {
    match lit {
        GpuLiteral::Int(i) => *i as f64,
        GpuLiteral::Float(f) => *f,
        GpuLiteral::Date(d) => *d as f64,
        GpuLiteral::Str(_) => 0.0,
    }
}

pub(super) fn literal_as_i64(lit: &GpuLiteral) -> i64 {
    match lit {
        GpuLiteral::Int(i) => *i,
        GpuLiteral::Float(f) => *f as i64,
        GpuLiteral::Date(d) => *d as i64,
        GpuLiteral::Str(_) => 0,
    }
}

pub(super) fn literal_as_f32(lit: &GpuLiteral) -> f32 {
    match lit {
        GpuLiteral::Int(i) => *i as f32,
        GpuLiteral::Float(f) => *f as f32,
        GpuLiteral::Date(d) => *d as f32,
        GpuLiteral::Str(_) => 0.0,
    }
}

/// Compute filtered row indices for a table given a set of filters.
/// Uses GPU-native filter → mask → indices pipeline when all filters are on
/// GPU-supported types. Falls back to CPU for f64/Utf8 filters.
pub(super) fn compute_filter_indices(
    table: &PinnedTable,
    filters: &[&GpuFilter],
    config: &BufferConfig,
) -> Result<Vec<usize>, Box<dyn Error>> {
    // Try GPU-native path: all filters must be on GPU-supported column types.
    if let Some(indices) = try_gpu_filter_indices(table, filters, config)? {
        return Ok(indices);
    }

    // CPU fallback: compute bool masks and convert to indices.
    let mut combined: Option<Vec<bool>> = None;
    for filter in filters {
        let col = table
            .column(&filter.column)
            .ok_or_else(|| format!("pinned table missing filter column {:?}", filter.column))?;
        let m = match apply_single_filter(col, filter, config)? {
            Some(m) => m,
            None => return Err(format!("unsupported filter on column {:?}", filter.column).into()),
        };
        combined = Some(match combined {
            None => m,
            Some(mut prev) => {
                for (a, b) in prev.iter_mut().zip(m.iter()) {
                    *a &= *b;
                }
                prev
            }
        });
    }
    Ok(match combined {
        Some(m) => (0..table.row_count).filter(|&i| m[i]).collect(),
        None => (0..table.row_count).collect(),
    })
}

/// GPU-native filter path: compare → mask → AND → filter_to_indices, all on GPU.
fn try_gpu_filter_indices(
    table: &PinnedTable,
    filters: &[&GpuFilter],
    config: &BufferConfig,
) -> Result<Option<Vec<usize>>, Box<dyn Error>> {
    use crate::metal::GpuBuffer;

    let mut gpu_mask: Option<GpuBuffer> = None;

    // Batch compare + and_masks into one command buffer when there are multiple filters.
    let batching = filters.len() > 1;
    if batching {
        metal::begin_batch(config)?;
    }

    for filter in filters {
        let col = match table.column(&filter.column) {
            Some(c) => c,
            None => {
                if batching {
                    metal::end_batch(config)?;
                }
                return Ok(None);
            }
        };

        let mask = match gpu_mask_for_filter(col, filter, config)? {
            Some(mask) => mask,
            None => {
                if batching {
                    metal::end_batch(config)?;
                }
                return Ok(None);
            }
        };

        gpu_mask = Some(match gpu_mask {
            None => mask,
            Some(prev) => metal::and_masks_gpu(&prev, &mask, config)?,
        });
    }

    if batching {
        metal::end_batch(config)?;
    }

    let gpu_mask = match gpu_mask {
        Some(m) => m,
        None => return Ok(None),
    };

    // Convert GPU mask to index array, then download to CPU.
    let gpu_indices = metal::filter_to_indices_gpu(&gpu_mask, config)?;
    let indices_u32: Vec<u32> = gpu_indices.download_as();
    Ok(Some(indices_u32.iter().map(|&i| i as usize).collect()))
}

pub(super) fn filter_runs_on_gpu(col: &PinnedColumn, filter: &GpuFilter) -> bool {
    match col {
        PinnedColumn::Int32(_)
        | PinnedColumn::Date32(_)
        | PinnedColumn::Int64(_)
        | PinnedColumn::Float32(_) => !matches!(filter.literal, GpuLiteral::Str(_)),
        PinnedColumn::Float64(_) => false,
        PinnedColumn::DictionaryI32 { .. } => {
            matches!(filter.op, ScalarCompareOp::Eq) && matches!(filter.literal, GpuLiteral::Str(_))
        }
    }
}

pub(super) fn gpu_mask_for_filter(
    col: &PinnedColumn,
    filter: &GpuFilter,
    config: &BufferConfig,
) -> Result<Option<metal::GpuBuffer>, Box<dyn Error>> {
    if !filter_runs_on_gpu(col, filter) {
        return Ok(None);
    }

    let mask = match col {
        PinnedColumn::Int32(values) => {
            let scalar = literal_as_i32(&filter.literal)?;
            metal::compare_i32_to_gpu(values, scalar, filter.op, config)?
        }
        PinnedColumn::Date32(values) => {
            let scalar = literal_as_date32(&filter.literal)?;
            metal::compare_date32_to_gpu(values, scalar, filter.op, config)?
        }
        PinnedColumn::Int64(values) => {
            let scalar = literal_as_i64(&filter.literal);
            metal::compare_i64_to_gpu(values, scalar, filter.op, config)?
        }
        PinnedColumn::Float32(values) => {
            let scalar = literal_as_f32(&filter.literal);
            metal::compare_f32_to_gpu(values, scalar, filter.op, config)?
        }
        PinnedColumn::DictionaryI32 { codes, .. } => {
            let scalar = dictionary_literal_code(col, &filter.literal)?;
            metal::compare_i32_to_gpu(codes, scalar, filter.op, config)?
        }
        PinnedColumn::Float64(_) => return Ok(None),
    };
    Ok(Some(mask))
}

pub(super) fn dictionary_literal_code(
    col: &PinnedColumn,
    lit: &GpuLiteral,
) -> Result<i32, Box<dyn Error>> {
    match lit {
        GpuLiteral::Str(value) => Ok(col.dictionary_code(value).unwrap_or(-1)),
        _ => Err("dictionary filter requires a string literal".into()),
    }
}

fn apply_single_filter(
    col: &PinnedColumn,
    filter: &GpuFilter,
    config: &BufferConfig,
) -> Result<Option<Vec<bool>>, Box<dyn Error>> {
    match col {
        PinnedColumn::Int32(values) => {
            let scalar = literal_as_i32(&filter.literal)?;
            Ok(Some(metal::compare_i32_values(
                values, scalar, filter.op, config,
            )?))
        }
        PinnedColumn::Date32(values) => {
            let scalar = literal_as_date32(&filter.literal)?;
            Ok(Some(metal::compare_date32_values(
                values, scalar, filter.op, config,
            )?))
        }
        PinnedColumn::Int64(values) => {
            let scalar = literal_as_i64(&filter.literal);
            Ok(Some(metal::compare_i64_values(
                values, scalar, filter.op, config,
            )?))
        }
        PinnedColumn::Float32(values) => {
            let scalar = literal_as_f32(&filter.literal);
            Ok(Some(metal::compare_f32_values(
                values, scalar, filter.op, config,
            )?))
        }
        PinnedColumn::Float64(values) => {
            let scalar = literal_as_f64(&filter.literal);
            Ok(Some(
                values
                    .iter()
                    .map(|&v| match filter.op {
                        ScalarCompareOp::Eq => v == scalar,
                        ScalarCompareOp::Lt => v < scalar,
                        ScalarCompareOp::Le => v <= scalar,
                        ScalarCompareOp::Gt => v > scalar,
                        ScalarCompareOp::Ge => v >= scalar,
                    })
                    .collect(),
            ))
        }
        PinnedColumn::DictionaryI32 { codes, .. } => {
            let target = dictionary_literal_code(col, &filter.literal)?;
            match filter.op {
                ScalarCompareOp::Eq => Ok(Some(codes.iter().map(|&code| code == target).collect())),
                _ => Ok(None),
            }
        }
    }
}

pub(super) fn sort_record_batch(
    batch: &RecordBatch,
    order_by: &[GpuOrderBy],
) -> Result<RecordBatch, Box<dyn Error>> {
    use std::cmp::Ordering;

    let num_rows = batch.num_rows();
    let mut indices: Vec<usize> = (0..num_rows).collect();

    // Resolve sort column indices once.
    let sort_col_indices: Vec<(usize, bool)> = order_by
        .iter()
        .filter_map(|ob| {
            let idx = batch.schema().index_of(&ob.column).ok()?;
            Some((idx, ob.descending))
        })
        .collect();

    if sort_col_indices.is_empty() {
        return Ok(batch.clone());
    }

    // Use typed comparison to avoid f64 precision loss on i64 values.
    indices.sort_by(|&a, &b| {
        for &(col_idx, desc) in &sort_col_indices {
            let col = batch.column(col_idx);
            let cmp = compare_array_values(col, a, b);
            let cmp = if desc { cmp.reverse() } else { cmp };
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    });

    let new_columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|col_idx| reorder_column(batch.column(col_idx), &indices))
        .collect();

    Ok(RecordBatch::try_new(batch.schema(), new_columns)?)
}

fn compare_array_values(col: &ArrayRef, a: usize, b: usize) -> std::cmp::Ordering {
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        return arr.value(a).cmp(&arr.value(b));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return arr.value(a).cmp(&arr.value(b));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return arr
            .value(a)
            .partial_cmp(&arr.value(b))
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return arr
            .value(a)
            .partial_cmp(&arr.value(b))
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Date32Array>() {
        return arr.value(a).cmp(&arr.value(b));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return arr.value(a).cmp(arr.value(b));
    }
    std::cmp::Ordering::Equal
}

fn reorder_column(col: &ArrayRef, indices: &[usize]) -> ArrayRef {
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        return Arc::new(Int32Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return Arc::new(Int64Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return Arc::new(Float32Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return Arc::new(Float64Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Date32Array>() {
        return Arc::new(Date32Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Arc::new(StringArray::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    col.clone()
}
