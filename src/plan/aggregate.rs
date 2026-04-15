//! Grouped and ungrouped aggregate execution paths.

use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use duckdb::arrow::array::{ArrayRef, Date32Array, Float64Array, Int32Array, StringArray};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;

use crate::config::BufferConfig;
use crate::metal::{self, GpuGroupedAggregateSpec, GpuGroupedAggregateStateKind};
use crate::pinned::PinnedColumn;
use crate::processing::MaterializedQueryResult;

use super::helpers::{
    ColumnLookup, column_arrow_type, evaluate_expr, gather_column, gather_group_key_as_i32_gv,
    resolve_column_gpu, sort_record_batch, upload_indices_as_gv,
};
use super::ir::{
    AggFunc, AggregateInputPlan, AggregateSlotPlan, AggregateStateKind, GpuAggregate, GpuExpr,
    GpuPlan, GroupedAggregateOutput, GroupedAggregatePlan,
};

pub(super) fn build_grouped_aggregate_plan(
    plan: &GpuPlan,
    agg: &GpuAggregate,
) -> Option<GroupedAggregatePlan> {
    let mut aggregate_slots = Vec::new();
    let mut outputs = Vec::with_capacity(plan.projections.len());

    for proj in &plan.projections {
        match &proj.expr {
            GpuExpr::Column(name) => {
                let group_idx = agg
                    .group_by
                    .iter()
                    .position(|group_col| group_col.eq_ignore_ascii_case(name))?;
                outputs.push(GroupedAggregateOutput::GroupKey {
                    group_idx,
                    alias: proj.alias.clone(),
                });
            }
            GpuExpr::AggCall { func, arg } => {
                let input = if *func == AggFunc::Count
                    && matches!(arg.as_ref(), GpuExpr::Column(c) if c == "*")
                {
                    AggregateInputPlan::CountStar
                } else {
                    AggregateInputPlan::Expr((**arg).clone())
                };
                let agg_idx = aggregate_slots.len();
                aggregate_slots.push(AggregateSlotPlan {
                    alias: proj.alias.clone(),
                    func: *func,
                    input,
                    state_kind: super::helpers::aggregate_state_kind(*func),
                });
                outputs.push(GroupedAggregateOutput::Aggregate { agg_idx });
            }
            _ => return None,
        }
    }

    Some(GroupedAggregatePlan {
        group_by: agg.group_by.clone(),
        aggregate_slots,
        outputs,
    })
}

pub(super) fn grouped_aggregate_output_type(
    output: &GroupedAggregateOutput,
    grouped_plan: &GroupedAggregatePlan,
    lookup: &dyn ColumnLookup,
) -> DataType {
    match output {
        GroupedAggregateOutput::GroupKey { group_idx, .. } => lookup
            .get_column(&grouped_plan.group_by[*group_idx])
            .map(|(col, _)| column_arrow_type(col))
            .unwrap_or(DataType::Utf8),
        // Aggregate outputs stay Float64 today because both CPU and GPU grouped
        // paths normalize numeric state into f64 result columns.
        GroupedAggregateOutput::Aggregate { .. } => DataType::Float64,
    }
}

pub(super) fn execute_aggregate_impl(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
    agg: &GpuAggregate,
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    if agg.group_by.is_empty() {
        return execute_ungrouped_aggregate(plan, lookup, config);
    }
    let Some(grouped_plan) = build_grouped_aggregate_plan(plan, agg) else {
        return Ok(None);
    };

    // GPU fast path: single integer group-by key with only SUM aggregates.
    if let Some(result) = try_gpu_grouped_aggregate(plan, lookup, &grouped_plan, config)? {
        return Ok(Some(result));
    }

    // CPU fallback: multi-key or non-SUM aggregates.
    let num_rows = lookup.row_count();
    if num_rows == 0 {
        let schema = build_grouped_aggregate_schema(&grouped_plan, lookup);
        return Ok(Some(MaterializedQueryResult::from_batches(
            Arc::new(schema),
            Vec::new(),
        )));
    }

    // Gather group-by columns with their per-table indices.
    let group_cols: Vec<(&PinnedColumn, &[usize])> = grouped_plan
        .group_by
        .iter()
        .map(|name| {
            lookup
                .get_column(name)
                .ok_or_else(|| format!("pinned table missing group-by column {name:?}").into())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    // Build composite key → group index map.
    let mut key_to_group: HashMap<Vec<GroupKey>, usize> = HashMap::new();
    let mut group_positions: Vec<Vec<usize>> = Vec::new();

    for pos in 0..num_rows {
        let key: Vec<GroupKey> = group_cols
            .iter()
            .map(|(col, indices)| GroupKey::from_column(col, indices[pos]))
            .collect();
        let group_idx = match key_to_group.get(&key) {
            Some(&idx) => idx,
            None => {
                let idx = group_positions.len();
                key_to_group.insert(key, idx);
                group_positions.push(Vec::new());
                idx
            }
        };
        group_positions[group_idx].push(pos);
    }

    let num_groups = group_positions.len();

    // Pre-evaluate all aggregate inner expressions ONCE over all selected rows.
    // This avoids per-group GPU dispatches — the key performance optimization.
    let mut precomputed: Vec<Option<Vec<f64>>> = Vec::new();
    for slot in &grouped_plan.aggregate_slots {
        match &slot.input {
            AggregateInputPlan::CountStar => precomputed.push(None),
            AggregateInputPlan::Expr(expr) => {
                let values = evaluate_expr(expr, lookup, config)?;
                precomputed.push(Some(values));
            }
        }
    }

    // Build output columns.
    let mut fields = Vec::new();
    let mut output_columns: Vec<ArrayRef> = Vec::new();

    for output in &grouped_plan.outputs {
        match output {
            GroupedAggregateOutput::GroupKey { group_idx, alias } => {
                let col_name = &grouped_plan.group_by[*group_idx];
                let (col, col_indices) = lookup
                    .get_column(col_name)
                    .ok_or_else(|| format!("pinned table missing column {col_name:?}"))?;
                let representative_indices: Vec<usize> =
                    group_positions.iter().map(|g| col_indices[g[0]]).collect();
                fields.push(Field::new(alias, column_arrow_type(col), false));
                output_columns.push(gather_column(col, &representative_indices));
            }
            GroupedAggregateOutput::Aggregate { agg_idx } => {
                let slot = &grouped_plan.aggregate_slots[*agg_idx];
                let vals = &precomputed[*agg_idx];
                let mut group_results = Vec::with_capacity(num_groups);
                for group_pos in &group_positions {
                    let result = match slot.state_kind {
                        AggregateStateKind::CountU64 => group_pos.len() as f64,
                        AggregateStateKind::SumF64 => {
                            let v = vals.as_ref().unwrap();
                            group_pos.iter().map(|&p| v[p]).sum()
                        }
                        AggregateStateKind::AvgSumCount => {
                            let v = vals.as_ref().unwrap();
                            let sum: f64 = group_pos.iter().map(|&p| v[p]).sum();
                            sum / group_pos.len() as f64
                        }
                        AggregateStateKind::MinF64 => {
                            let v = vals.as_ref().unwrap();
                            group_pos
                                .iter()
                                .map(|&p| v[p])
                                .fold(f64::INFINITY, f64::min)
                        }
                        AggregateStateKind::MaxF64 => {
                            let v = vals.as_ref().unwrap();
                            group_pos
                                .iter()
                                .map(|&p| v[p])
                                .fold(f64::NEG_INFINITY, f64::max)
                        }
                    };
                    group_results.push(result);
                }
                fields.push(Field::new(&slot.alias, DataType::Float64, false));
                output_columns.push(Arc::new(Float64Array::from(group_results)));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let mut batch = RecordBatch::try_new(schema.clone(), output_columns)?;

    if !plan.order_by.is_empty() {
        batch = sort_record_batch(&batch, &plan.order_by)?;
    }
    if let Some(limit) = plan.limit {
        if batch.num_rows() > limit {
            batch = batch.slice(0, limit);
        }
    }

    Ok(Some(MaterializedQueryResult::from_batches(
        batch.schema(),
        vec![batch],
    )))
}

fn build_group_key_array_from_i32_values(col: &PinnedColumn, keys: &[i32]) -> Option<ArrayRef> {
    match col {
        PinnedColumn::Int32(_) => Some(Arc::new(Int32Array::from(keys.to_vec()))),
        PinnedColumn::Date32(_) => Some(Arc::new(Date32Array::from(keys.to_vec()))),
        PinnedColumn::DictionaryI32 { values, .. } => {
            let decoded: Vec<String> = keys
                .iter()
                .map(|&code| {
                    usize::try_from(code)
                        .ok()
                        .and_then(|idx| values.get(idx))
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();
            Some(Arc::new(StringArray::from(decoded)))
        }
        PinnedColumn::Int64(_) | PinnedColumn::Float32(_) | PinnedColumn::Float64(_) => None,
    }
}

fn gpu_state_kind(kind: AggregateStateKind) -> GpuGroupedAggregateStateKind {
    match kind {
        AggregateStateKind::CountU64 => GpuGroupedAggregateStateKind::Count,
        AggregateStateKind::SumF64 => GpuGroupedAggregateStateKind::Sum,
        AggregateStateKind::MinF64 => GpuGroupedAggregateStateKind::Min,
        AggregateStateKind::MaxF64 => GpuGroupedAggregateStateKind::Max,
        AggregateStateKind::AvgSumCount => GpuGroupedAggregateStateKind::Avg,
    }
}

/// GPU fast path for grouped aggregates: exact i32-like grouping with multi-state aggregates.
fn try_gpu_grouped_aggregate(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
    grouped_plan: &GroupedAggregatePlan,
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let num_rows = lookup.row_count();
    if grouped_plan.group_by.is_empty()
        || grouped_plan.group_by.len() > 4
        || grouped_plan.aggregate_slots.is_empty()
    {
        return Ok(None);
    }

    if num_rows == 0 {
        let schema = build_grouped_aggregate_schema(grouped_plan, lookup);
        return Ok(Some(MaterializedQueryResult::from_batches(
            Arc::new(schema),
            Vec::new(),
        )));
    }

    let Some((table_refs, table_indices)) = lookup.gpu_relation_bindings() else {
        return Ok(None);
    };
    let gv_indices = upload_indices_as_gv(table_indices, config)?;

    let mut group_key_vectors = Vec::with_capacity(grouped_plan.group_by.len());
    let mut group_key_columns = Vec::with_capacity(grouped_plan.group_by.len());
    for group_name in &grouped_plan.group_by {
        let (table_idx, group_col) = match resolve_column_gpu(group_name, table_refs) {
            Some(pair) => pair,
            None => return Ok(None),
        };
        let Some(group_gv) = gather_group_key_as_i32_gv(group_col, &gv_indices[table_idx], config)?
        else {
            return Ok(None);
        };
        group_key_vectors.push(group_gv);
        group_key_columns.push(group_col);
    }

    let mut aggregate_input_vectors: Vec<Option<metal::GpuVector>> =
        Vec::with_capacity(grouped_plan.aggregate_slots.len());
    for slot in &grouped_plan.aggregate_slots {
        match &slot.input {
            AggregateInputPlan::CountStar => aggregate_input_vectors.push(None),
            AggregateInputPlan::Expr(inner_expr) => {
                let Some(values_gv) = super::helpers::evaluate_expr_gv(
                    inner_expr,
                    table_refs,
                    &gv_indices,
                    config,
                )?
                else {
                    return Ok(None);
                };
                aggregate_input_vectors.push(Some(values_gv));
            }
        }
    }

    let aggregate_specs: Vec<GpuGroupedAggregateSpec<'_>> = grouped_plan
        .aggregate_slots
        .iter()
        .zip(aggregate_input_vectors.iter())
        .map(|(slot, values)| GpuGroupedAggregateSpec {
            state_kind: gpu_state_kind(slot.state_kind),
            values: values.as_ref(),
        })
        .collect();
    let gpu_rows = metal::grouped_aggregate_exact_gv(&group_key_vectors, &aggregate_specs, config)?;

    // Build output columns.
    let mut fields = Vec::new();
    let mut output_columns: Vec<ArrayRef> = Vec::new();

    for output in &grouped_plan.outputs {
        match output {
            GroupedAggregateOutput::GroupKey { group_idx, alias } => {
                let col = group_key_columns[*group_idx];
                let Some(array) =
                    build_group_key_array_from_i32_values(col, &gpu_rows.key_lanes[*group_idx])
                else {
                    return Ok(None);
                };
                fields.push(Field::new(alias, column_arrow_type(col), false));
                output_columns.push(array);
            }
            GroupedAggregateOutput::Aggregate { agg_idx } => {
                let slot = &grouped_plan.aggregate_slots[*agg_idx];
                let gpu_res = &gpu_rows.slots[*agg_idx];
                let values_f64: Vec<f64> = match slot.state_kind {
                    AggregateStateKind::CountU64 => gpu_res
                        .counts
                        .as_ref()
                        .ok_or("gpu grouped aggregate count slot is missing counts")?
                        .iter()
                        .map(|&count| count as f64)
                        .collect(),
                    AggregateStateKind::SumF64
                    | AggregateStateKind::MinF64
                    | AggregateStateKind::MaxF64 => gpu_res
                        .values
                        .as_ref()
                        .ok_or("gpu grouped aggregate value slot is missing values")?
                        .iter()
                        .map(|&value| value as f64)
                        .collect(),
                    AggregateStateKind::AvgSumCount => {
                        let sums = gpu_res
                            .values
                            .as_ref()
                            .ok_or("gpu grouped aggregate avg slot is missing sums")?;
                        let counts = gpu_res
                            .counts
                            .as_ref()
                            .ok_or("gpu grouped aggregate avg slot is missing counts")?;
                        sums.iter()
                            .zip(counts.iter())
                            .map(|(&sum, &count)| {
                                if count == 0 {
                                    0.0
                                } else {
                                    f64::from(sum) / count as f64
                                }
                            })
                            .collect()
                    }
                };
                fields.push(Field::new(&slot.alias, DataType::Float64, false));
                output_columns.push(Arc::new(Float64Array::from(values_f64)));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let mut batch = RecordBatch::try_new(schema.clone(), output_columns)?;

    if !plan.order_by.is_empty() {
        batch = sort_record_batch(&batch, &plan.order_by)?;
    }
    if let Some(limit) = plan.limit {
        if batch.num_rows() > limit {
            batch = batch.slice(0, limit);
        }
    }

    Ok(Some(MaterializedQueryResult::from_batches(
        batch.schema(),
        vec![batch],
    )))
}

fn execute_ungrouped_aggregate(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    for proj in &plan.projections {
        match &proj.expr {
            GpuExpr::AggCall { func, arg } => {
                let value = compute_ungrouped_agg(*func, arg, lookup, config)?;
                match value {
                    Some(v) => {
                        fields.push(Field::new(&proj.alias, DataType::Float64, false));
                        columns.push(Arc::new(Float64Array::from(vec![v])));
                    }
                    None => return Ok(None),
                }
            }
            GpuExpr::Column(_) => return Ok(None),
            _ => return Ok(None),
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    Ok(Some(MaterializedQueryResult::from_batches(
        schema,
        vec![batch],
    )))
}

pub(super) fn build_grouped_aggregate_schema(
    grouped_plan: &GroupedAggregatePlan,
    lookup: &dyn ColumnLookup,
) -> Schema {
    let fields: Vec<Field> = grouped_plan
        .outputs
        .iter()
        .map(|output| match output {
            GroupedAggregateOutput::GroupKey { alias, .. } => Field::new(
                alias,
                grouped_aggregate_output_type(output, grouped_plan, lookup),
                false,
            ),
            GroupedAggregateOutput::Aggregate { agg_idx } => Field::new(
                &grouped_plan.aggregate_slots[*agg_idx].alias,
                grouped_aggregate_output_type(output, grouped_plan, lookup),
                false,
            ),
        })
        .collect();
    Schema::new(fields)
}

/// Hashable group key value for CPU-side grouped aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupKey {
    Int32(i32),
    Int64(i64),
    Date32(i32),
    Str(String),
    /// For f32/f64 we use the bit pattern for hashing.
    F64Bits(u64),
}

impl GroupKey {
    fn from_column(col: &PinnedColumn, idx: usize) -> Self {
        match col {
            PinnedColumn::Int32(v) => GroupKey::Int32(v[idx]),
            PinnedColumn::Int64(v) => GroupKey::Int64(v[idx]),
            PinnedColumn::Date32(v) => GroupKey::Date32(v[idx]),
            PinnedColumn::Float32(v) => GroupKey::F64Bits((v[idx] as f64).to_bits()),
            PinnedColumn::Float64(v) => GroupKey::F64Bits(v[idx].to_bits()),
            PinnedColumn::DictionaryI32 { codes, values, .. } => {
                let value = usize::try_from(codes[idx])
                    .ok()
                    .and_then(|code| values.get(code))
                    .cloned()
                    .unwrap_or_default();
                GroupKey::Str(value)
            }
        }
    }
}

fn compute_ungrouped_agg(
    func: AggFunc,
    arg: &GpuExpr,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Option<f64>, Box<dyn Error>> {
    let num_rows = lookup.row_count();
    if func == AggFunc::Count && matches!(arg, GpuExpr::Column(c) if c == "*") {
        return Ok(Some(num_rows as f64));
    }

    if num_rows == 0 {
        return Ok(Some(0.0));
    }

    let values = evaluate_expr(arg, lookup, config)?;

    let result = match func {
        AggFunc::Count => values.len() as f64,
        AggFunc::Sum => values.iter().sum(),
        AggFunc::Avg => {
            let sum: f64 = values.iter().sum();
            sum / values.len() as f64
        }
        AggFunc::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        AggFunc::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    };

    Ok(Some(result))
}
