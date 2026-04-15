//! GPU plan execution: IR → Metal kernels (or CPU-orchestrated join loop with
//! per-operation GPU calls). Includes join planning, liveness analysis,
//! GPU-native join+aggregate pipeline, and the `try_execute` entry point.

mod cache;
mod fast_path;
mod join_plan;
mod resolve;
mod topk;

use std::error::Error;
use std::sync::Arc;

use duckdb::arrow::array::{ArrayRef, Float64Array};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;

use crate::config::BufferConfig;
use crate::host_plan;
use crate::metal;
use crate::pinned::{self, PinnedColumn, PinnedTable};
use crate::processing::MaterializedQueryResult;
use crate::state;

use super::aggregate::{build_grouped_aggregate_plan, execute_aggregate_impl};
use super::helpers::{
    ColumnLookup, JoinResult, JoinResultLookup, column_arrow_type, compute_filter_indices,
    evaluate_expr, gather_column, sort_record_batch,
};
use super::ir::{GpuExpr, GpuPlan, SourceSelection};

pub(in crate::plan) use join_plan::{
    choose_gpu_join_strategy, join_execution_steps, live_sources_after_join_steps,
};

use join_plan::{build_source_selections, execution_source_order, NativeGpuShape};
use resolve::source_filters;
use topk::is_grouped_topk_pilot;

use fast_path::execute_gpu_native;

/// Try to execute a parsed GpuPlan against pinned tables. Returns None if the
/// table isn't pinned or a column type isn't GPU-executable.
pub fn try_execute(
    plan: &GpuPlan,
    sql: Option<&str>,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    // Every source table must be pinned.
    let pinned_tables: Vec<Arc<PinnedTable>> = plan
        .sources
        .iter()
        .map(|src| pinned::get(&src.name))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if pinned_tables.len() != plan.sources.len() {
        return Ok(None);
    }

    let config = state::current_config()?;
    let host_hints = if plan.sources.len() > 1 && !plan.joins.is_empty() {
        sql.and_then(host_plan::explain_hints)
    } else {
        None
    };
    let execution_order = execution_source_order(plan, &pinned_tables, host_hints.as_ref());

    match native_gpu_shape(plan, &pinned_tables, &execution_order) {
        Some(NativeGpuShape::SingleTableGroupedAggregate) => {
            if let Some(result) = execute_single_table_native(plan, &pinned_tables, &config)? {
                return Ok(Some(result));
            }
        }
        Some(NativeGpuShape::JoinedGroupedAggregate) => {
            if let Some(result) = execute_joined_grouped_aggregate_native(
                plan,
                &pinned_tables,
                &execution_order,
                &config,
            )? {
                return Ok(Some(result));
            }
        }
        Some(NativeGpuShape::JoinedGroupedTopKPilot) => {
            if let Some(result) =
                execute_gpu_native(plan, &pinned_tables, None, &execution_order, &config)?
            {
                return Ok(Some(result));
            }
        }
        None => {
            if plan.sources.len() == 1 && plan.joins.is_empty() {
                return execute_single_table_native(plan, &pinned_tables, &config);
            }
        }
    }

    // Fallback: CPU-orchestrated join loop with per-operation GPU calls.
    let source_selections = build_source_selections(plan, &pinned_tables, &config)?;
    execute_multi_table(
        plan,
        &pinned_tables,
        source_selections,
        &execution_order,
        &config,
    )
}

pub fn result_cache_key(plan: &GpuPlan) -> Option<String> {
    if plan.sources.is_empty() {
        return None;
    }
    Some(format!("query-result|plan={plan:?}"))
}

fn native_gpu_shape(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> Option<NativeGpuShape> {
    if plan.sources.len() == 1 && plan.joins.is_empty() {
        let agg = plan.aggregate.as_ref()?;
        let grouped_plan = build_grouped_aggregate_plan(plan, agg)?;
        if grouped_plan.group_by.is_empty() || grouped_plan.aggregate_slots.is_empty() {
            return None;
        }
        return Some(NativeGpuShape::SingleTableGroupedAggregate);
    }

    if is_grouped_topk_pilot(plan, pinned_tables, execution_order) {
        return Some(NativeGpuShape::JoinedGroupedTopKPilot);
    }

    if is_joined_grouped_aggregate_shape(plan, pinned_tables, execution_order) {
        return Some(NativeGpuShape::JoinedGroupedAggregate);
    }

    None
}

fn execute_single_table_native(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let table = &pinned_tables[0];
    let src_ref = plan.sources[0]
        .alias
        .as_deref()
        .unwrap_or(plan.sources[0].name.as_str());
    let all_filters = source_filters(plan, table, src_ref);
    let indices = if all_filters.is_empty() {
        (0..table.row_count).collect()
    } else {
        compute_filter_indices(table, &all_filters, config)?
    };
    let join_result = JoinResult {
        tables: vec![(src_ref.to_owned(), pinned_tables[0].clone())],
        table_indices: vec![indices],
    };
    let lookup = JoinResultLookup(&join_result);
    execute_from_lookup(plan, &lookup, config)
}

fn is_joined_grouped_aggregate_shape(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> bool {
    if plan.sources.len() < 2 || plan.joins.is_empty() {
        return false;
    }

    let Some(agg) = plan.aggregate.as_ref() else {
        return false;
    };
    let Some(grouped_plan) = build_grouped_aggregate_plan(plan, agg) else {
        return false;
    };
    if grouped_plan.group_by.is_empty() || grouped_plan.aggregate_slots.is_empty() {
        return false;
    }

    join_execution_steps(plan, pinned_tables, execution_order).is_some()
}

fn execute_joined_grouped_aggregate_native(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let source_selections = build_source_selections(plan, pinned_tables, config)?;
    execute_multi_table(
        plan,
        pinned_tables,
        source_selections,
        execution_order,
        config,
    )
}

/// Shared post-join execution: aggregate routing → projection → ORDER BY → LIMIT.
fn execute_from_lookup(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    if lookup.row_count() == 0 {
        let schema = build_output_schema(plan, lookup)?;
        return Ok(Some(MaterializedQueryResult::from_batches(
            Arc::new(schema),
            Vec::new(),
        )));
    }

    if let Some(ref agg) = plan.aggregate {
        return execute_aggregate_impl(plan, lookup, agg, config);
    }

    let schema = build_output_schema(plan, lookup)?;
    let mut output_columns: Vec<ArrayRef> = Vec::new();

    for proj in &plan.projections {
        match &proj.expr {
            GpuExpr::Column(col_name) => {
                let (col, idx) = lookup
                    .get_column(col_name)
                    .ok_or_else(|| format!("column {col_name:?} not found"))?;
                output_columns.push(gather_column(col, idx));
            }
            GpuExpr::BinaryOp { .. } | GpuExpr::Literal(_) => {
                let values = evaluate_expr(&proj.expr, lookup, config)?;
                output_columns.push(Arc::new(Float64Array::from(values)));
            }
            GpuExpr::AggCall { .. } => return Ok(None),
        }
    }

    let mut batch = RecordBatch::try_new(Arc::new(schema), output_columns)?;

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

fn build_join_result_from_gpu_indices(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    joined_sources: &[usize],
    gv_indices: &[metal::GpuVector],
) -> JoinResult {
    let tables = joined_sources
        .iter()
        .map(|&src_idx| {
            let source = &plan.sources[src_idx];
            (
                source.alias.as_deref().unwrap_or(&source.name).to_owned(),
                pinned_tables[src_idx].clone(),
            )
        })
        .collect();
    let table_indices = joined_sources
        .iter()
        .map(|&src_idx| {
            gv_indices[src_idx]
                .download_data_as::<u32>(gv_indices[src_idx].download_count())
                .into_iter()
                .map(|idx| idx as usize)
                .collect()
        })
        .collect();
    JoinResult {
        tables,
        table_indices,
    }
}


fn execute_multi_table(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    source_selections: Vec<SourceSelection>,
    execution_order: &[usize],
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let Some(join_steps) = join_execution_steps(plan, pinned_tables, execution_order) else {
        return Ok(None);
    };
    let live_after_steps =
        live_sources_after_join_steps(plan, pinned_tables, execution_order, &join_steps, &[]);

    // Late materialization: carry per-table index arrays through the join
    // loop instead of copying column data.
    let mut join_result = JoinResult {
        tables: Vec::new(),
        table_indices: Vec::new(),
    };
    let mut joined_source_ids = Vec::new();

    for (order_pos, &src_idx) in execution_order.iter().enumerate() {
        let src = &plan.sources[src_idx];
        let table = &pinned_tables[src_idx];
        let src_ref = src.alias.as_deref().unwrap_or(src.name.as_str());
        let indices = source_selections[src_idx].final_indices.clone();

        if order_pos == 0 {
            // First table: just store a reference and filtered indices.
            join_result
                .tables
                .push((src_ref.to_owned(), pinned_tables[src_idx].clone()));
            join_result.table_indices.push(indices);
            joined_source_ids.push(src_idx);
        } else {
            // Join with running result.
            let step = &join_steps[order_pos - 1];
            let build_col_name = &step.build_col_name;
            let probe_col_name = &step.probe_col_name;

            // Extract build-side join key from original pinned table via index array.
            let build_lookup = JoinResultLookup(&join_result);
            let (build_col, build_indices) = build_lookup
                .get_column(build_col_name.as_str())
                .ok_or_else(|| {
                    format!("join key {build_col_name:?} not found in running result")
                })?;
            let build_keys: Vec<i32> = match build_col {
                PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
                    build_indices.iter().map(|&i| v[i]).collect()
                }
                PinnedColumn::Int64(v) => build_indices.iter().map(|&i| v[i] as i32).collect(),
                _ => return Ok(None),
            };

            // Extract probe-side join key from this table's filtered rows.
            let probe_col = table
                .column(probe_col_name.as_str())
                .ok_or_else(|| format!("pinned table missing join column {probe_col_name:?}"))?;
            let probe_keys: Vec<i32> = match probe_col {
                PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
                    indices.iter().map(|&i| v[i]).collect()
                }
                PinnedColumn::Int64(v) => indices.iter().map(|&i| v[i] as i32).collect(),
                _ => return Ok(None),
            };

            // GPU hash join.
            let ht_capacity = (build_keys.len().saturating_mul(2).max(16)).next_power_of_two();
            let ht_capacity = u32::try_from(ht_capacity)
                .map_err(|_| "hash join capacity overflowed u32".to_owned())?;
            let matched =
                metal::hash_join_probe_i32(&build_keys, &probe_keys, ht_capacity, config)?;

            let mut probe_kept: Vec<usize> = Vec::new();
            let mut build_kept: Vec<usize> = Vec::new();
            for (probe_idx, &build_row_idx) in matched.iter().enumerate() {
                if build_row_idx >= 0 {
                    probe_kept.push(probe_idx);
                    build_kept.push(build_row_idx as usize);
                }
            }

            let live_after = &live_after_steps[order_pos - 1];
            let mut next_tables = Vec::with_capacity(join_result.tables.len() + 1);
            let mut next_table_indices = Vec::with_capacity(join_result.table_indices.len() + 1);
            let mut next_joined_source_ids = Vec::with_capacity(joined_source_ids.len() + 1);

            for (table_pos, &joined_source_idx) in joined_source_ids.iter().enumerate() {
                if live_after[joined_source_idx] {
                    next_tables.push(join_result.tables[table_pos].clone());
                    next_table_indices.push(
                        build_kept
                            .iter()
                            .map(|&bi| join_result.table_indices[table_pos][bi])
                            .collect(),
                    );
                    next_joined_source_ids.push(joined_source_idx);
                }
            }

            if live_after[src_idx] {
                let new_table_indices: Vec<usize> =
                    probe_kept.iter().map(|&pi| indices[pi]).collect();
                next_tables.push((src_ref.to_owned(), pinned_tables[src_idx].clone()));
                next_table_indices.push(new_table_indices);
                next_joined_source_ids.push(src_idx);
            }

            join_result.tables = next_tables;
            join_result.table_indices = next_table_indices;
            joined_source_ids = next_joined_source_ids;
        }

        if join_result.row_count() == 0 && order_pos > 0 {
            let lookup = JoinResultLookup(&join_result);
            let schema = build_output_schema(plan, &lookup)?;
            return Ok(Some(MaterializedQueryResult::from_batches(
                Arc::new(schema),
                Vec::new(),
            )));
        }
    }

    let lookup = JoinResultLookup(&join_result);
    execute_from_lookup(plan, &lookup, config)
}

fn build_output_schema(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
) -> Result<Schema, Box<dyn Error>> {
    let mut fields = Vec::new();
    for proj in &plan.projections {
        let data_type = match &proj.expr {
            GpuExpr::Column(col_name) => {
                let (col, _) = lookup
                    .get_column(col_name)
                    .ok_or_else(|| format!("pinned table missing column {col_name:?}"))?;
                column_arrow_type(col)
            }
            GpuExpr::BinaryOp { .. } | GpuExpr::Literal(_) => DataType::Float64,
            GpuExpr::AggCall { .. } => DataType::Float64,
        };
        fields.push(Field::new(&proj.alias, data_type, false));
    }
    Ok(Schema::new(fields))
}
