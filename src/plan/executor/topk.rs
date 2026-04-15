//! Grouped top-K pilot planning: recognizing the shape, building representative
//! and candidate plans, and materializing the final grouped top-K result.

use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;

use duckdb::arrow::array::{ArrayRef, Float64Array};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;

use crate::pinned::{PinnedColumn, PinnedTable};
use crate::processing::MaterializedQueryResult;

use super::super::aggregate::build_grouped_aggregate_plan;
use super::super::helpers::{
    column_arrow_type, column_name_only, filter_runs_on_gpu, gather_column, sort_record_batch,
};
use super::super::ir::{AggFunc, AggregateInputPlan, GpuAggregate, GpuExpr, GpuPlan};
use super::join_plan::{find_join_for_source, join_execution_steps};
use super::resolve::{
    find_equivalent_column_on_table, resolve_column_in_sources, resolve_column_name_in_sources,
};

#[derive(Debug, Clone, Copy)]
pub(in crate::plan) struct GroupedTopKPlan {
    pub(super) limit: usize,
    pub(super) secondary_order_group_idx: Option<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct CandidateGroupedTopKPlan {
    pub(super) aggregate_source_idx: usize,
    pub(super) representative_key_column: String,
    pub(super) aggregate_key_column: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExperimentalGroupedTopKMode {
    CandidateCompact,
    SlotValuesGpuTopK,
    SlotFusedGpuTopK,
    SlotFusedCpuTopK,
    GenericGroupedAggregate,
}

pub(super) struct RepresentativeMaterializePlan {
    pub(super) table_idx: usize,
    pub(super) group_by_anchor_columns: Vec<String>,
    pub(super) secondary_order_column: String,
}

pub(super) fn grouped_topk_plan(plan: &GpuPlan) -> Option<GroupedTopKPlan> {
    let agg = plan.aggregate.as_ref()?;
    let grouped_plan = build_grouped_aggregate_plan(plan, agg)?;

    let limit = plan.limit?;
    if !(1..=64).contains(&limit) {
        return None;
    }

    if grouped_plan.group_by.is_empty() || grouped_plan.aggregate_slots.len() != 1 {
        return None;
    }
    let agg_slot = &grouped_plan.aggregate_slots[0];
    if agg_slot.func != AggFunc::Sum || !matches!(agg_slot.input, AggregateInputPlan::Expr(_)) {
        return None;
    }

    if plan.order_by.is_empty() || plan.order_by.len() > 2 {
        return None;
    }

    let primary = &plan.order_by[0];
    if !primary.descending || !primary.column.eq_ignore_ascii_case(&agg_slot.alias) {
        return None;
    }

    let secondary_order_group_idx = match plan.order_by.get(1) {
        Some(ob) if !ob.descending => grouped_plan
            .group_by
            .iter()
            .position(|group_col| ob.column.eq_ignore_ascii_case(group_col)),
        Some(_) => return None,
        None => None,
    };
    if plan.order_by.get(1).is_some() && secondary_order_group_idx.is_none() {
        return None;
    }

    Some(GroupedTopKPlan {
        limit,
        secondary_order_group_idx,
    })
}

pub(super) fn experimental_grouped_topk_mode() -> Option<ExperimentalGroupedTopKMode> {
    match std::env::var("METTLE_EXPERIMENTAL_GROUPED_TOPK_MODE") {
        Ok(mode) => match mode.as_str() {
            "candidate" => Some(ExperimentalGroupedTopKMode::CandidateCompact),
            "slot" => Some(ExperimentalGroupedTopKMode::SlotValuesGpuTopK),
            "slot_fused" => Some(ExperimentalGroupedTopKMode::SlotFusedGpuTopK),
            "slot_cpu" => Some(ExperimentalGroupedTopKMode::SlotFusedCpuTopK),
            "generic" => Some(ExperimentalGroupedTopKMode::GenericGroupedAggregate),
            _ => None,
        },
        Err(_) if std::env::var_os("METTLE_EXPERIMENTAL_CANDIDATE_TOPK").is_some() => {
            Some(ExperimentalGroupedTopKMode::CandidateCompact)
        }
        Err(_) => None,
    }
}

pub(super) fn collect_expr_source_indices(
    expr: &GpuExpr,
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    sources: &mut HashSet<usize>,
) -> Option<()> {
    match expr {
        GpuExpr::Column(name) => {
            if name != "*" {
                let (source_idx, _) = resolve_column_name_in_sources(name, plan, pinned_tables)?;
                sources.insert(source_idx);
            }
        }
        GpuExpr::Literal(_) => {}
        GpuExpr::BinaryOp { left, right, .. } => {
            collect_expr_source_indices(left, plan, pinned_tables, sources)?;
            collect_expr_source_indices(right, plan, pinned_tables, sources)?;
        }
        GpuExpr::AggCall { arg, .. } => {
            collect_expr_source_indices(arg, plan, pinned_tables, sources)?;
        }
    }
    Some(())
}

pub(super) fn candidate_grouped_topk_plan(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
    grouped_plan: &super::super::ir::GroupedAggregatePlan,
    representative_plan: &RepresentativeMaterializePlan,
) -> Option<CandidateGroupedTopKPlan> {
    let agg_slot = grouped_plan.aggregate_slots.first()?;
    let AggregateInputPlan::Expr(sum_expr) = &agg_slot.input else {
        return None;
    };

    let mut expr_sources = HashSet::new();
    collect_expr_source_indices(sum_expr, plan, pinned_tables, &mut expr_sources)?;
    if expr_sources.len() != 1 {
        return None;
    }
    let aggregate_source_idx = *expr_sources.iter().next()?;
    if aggregate_source_idx == representative_plan.table_idx {
        return None;
    }

    let join_steps = join_execution_steps(plan, pinned_tables, execution_order)?;
    let final_step = join_steps.last()?;
    if final_step.probe_source_idx != aggregate_source_idx
        || final_step.build_source_idx != representative_plan.table_idx
    {
        return None;
    }

    let representative_key_column = final_step.build_col_name.clone();
    let aggregate_key_column = final_step.probe_col_name.clone();
    if !pinned_tables[representative_plan.table_idx].is_known_unique(&representative_key_column) {
        return None;
    }
    if !matches!(
        pinned_tables[representative_plan.table_idx].column(&representative_key_column),
        Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_))
    ) || !matches!(
        pinned_tables[aggregate_source_idx].column(&aggregate_key_column),
        Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_))
    ) {
        return None;
    }

    let join_key_in_group = grouped_plan.group_by.iter().any(|group_col| {
        let representative_match = find_equivalent_column_on_table(
            group_col,
            representative_plan.table_idx,
            plan,
            pinned_tables,
        )
        .is_some_and(|col| col.eq_ignore_ascii_case(&representative_key_column));
        let aggregate_match =
            find_equivalent_column_on_table(group_col, aggregate_source_idx, plan, pinned_tables)
                .is_some_and(|col| col.eq_ignore_ascii_case(&aggregate_key_column));
        representative_match && aggregate_match
    });
    if !join_key_in_group {
        return None;
    }

    Some(CandidateGroupedTopKPlan {
        aggregate_source_idx,
        representative_key_column,
        aggregate_key_column,
    })
}

pub(super) fn grouped_candidate_better(
    lhs_sum: f32,
    lhs_secondary: i32,
    lhs_row_id: u32,
    rhs_sum: f32,
    rhs_secondary: i32,
    rhs_row_id: u32,
    use_secondary: bool,
) -> bool {
    if lhs_sum > rhs_sum {
        return true;
    }
    if lhs_sum < rhs_sum {
        return false;
    }
    if use_secondary {
        if lhs_secondary < rhs_secondary {
            return true;
        }
        if lhs_secondary > rhs_secondary {
            return false;
        }
    }
    lhs_row_id < rhs_row_id
}

pub(super) fn build_representative_materialize_plan(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    agg: &GpuAggregate,
    topk: &GroupedTopKPlan,
) -> Option<RepresentativeMaterializePlan> {
    for table_idx in 0..plan.sources.len() {
        let mut group_by_anchor_columns = Vec::with_capacity(agg.group_by.len());
        for group_col in &agg.group_by {
            let Some(anchor_col) =
                find_equivalent_column_on_table(group_col, table_idx, plan, pinned_tables)
            else {
                group_by_anchor_columns.clear();
                break;
            };
            group_by_anchor_columns.push(anchor_col);
        }
        if group_by_anchor_columns.len() != agg.group_by.len() {
            continue;
        }

        let secondary_order_column = if let Some(group_idx) = topk.secondary_order_group_idx {
            let col_name = &group_by_anchor_columns[group_idx];
            match pinned_tables[table_idx].column(col_name.as_str()) {
                Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_)) => col_name.clone(),
                _ => continue,
            }
        } else if let Some(col_name) = group_by_anchor_columns.iter().find(|col_name| {
            matches!(
                pinned_tables[table_idx].column(col_name.as_str()),
                Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_))
            )
        }) {
            col_name.clone()
        } else {
            continue;
        };

        return Some(RepresentativeMaterializePlan {
            table_idx,
            group_by_anchor_columns,
            secondary_order_column,
        });
    }

    None
}

pub(super) fn materialize_grouped_topk_result(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    grouped_agg_plan: &super::super::ir::GroupedAggregatePlan,
    representative_plan: &RepresentativeMaterializePlan,
    representative_row_ids: &[u32],
    aggregate_vals: &[f32],
) -> Result<MaterializedQueryResult, Box<dyn Error>> {
    use super::super::ir::GroupedAggregateOutput;

    let gather_indices: Vec<usize> = representative_row_ids.iter().map(|&v| v as usize).collect();
    let representative_table = &pinned_tables[representative_plan.table_idx];

    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    for output in &grouped_agg_plan.outputs {
        match output {
            GroupedAggregateOutput::GroupKey { group_idx, alias } => {
                let anchor_col_name = &representative_plan.group_by_anchor_columns[*group_idx];
                let col = representative_table
                    .column(anchor_col_name)
                    .ok_or_else(|| {
                        format!("fast path representative column missing: {anchor_col_name}")
                    })?;
                let dt = column_arrow_type(col);
                fields.push(Field::new(alias, dt.clone(), false));
                columns.push(gather_column(col, &gather_indices));
            }
            GroupedAggregateOutput::Aggregate { agg_idx } => {
                let slot = &grouped_agg_plan.aggregate_slots[*agg_idx];
                fields.push(Field::new(&slot.alias, DataType::Float64, false));
                columns.push(Arc::new(Float64Array::from(
                    aggregate_vals
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect::<Vec<_>>(),
                )) as ArrayRef);
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let mut batch = RecordBatch::try_new(schema, columns)?;
    if !plan.order_by.is_empty() {
        batch = sort_record_batch(&batch, &plan.order_by)?;
    }
    if let Some(limit) = plan.limit {
        if batch.num_rows() > limit {
            batch = batch.slice(0, limit);
        }
    }

    Ok(MaterializedQueryResult::from_batches(
        batch.schema(),
        vec![batch],
    ))
}

pub(super) fn is_grouped_topk_pilot(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> bool {
    if plan.sources.len() < 2 || plan.joins.is_empty() {
        return false;
    }

    let Some(topk) = grouped_topk_plan(plan) else {
        return false;
    };

    if plan.filters.iter().any(|filter| {
        let Some((table_idx, col)) = resolve_column_in_sources(filter, plan, pinned_tables) else {
            return true;
        };
        !pinned_tables[table_idx].is_pilot_supported_column(&filter.column)
            || !filter_runs_on_gpu(col, filter)
    }) {
        return false;
    }

    let Some(agg) = &plan.aggregate else {
        return false;
    };

    if agg.group_by.is_empty() {
        return false;
    }

    if agg.group_by.iter().any(|name| {
        let Some((table_idx, col)) = resolve_column_name_in_sources(name, plan, pinned_tables)
        else {
            return true;
        };
        !pinned_tables[table_idx].is_pilot_supported_column(column_name_only(name))
            || matches!(
                col,
                PinnedColumn::DictionaryI32 { .. } | PinnedColumn::Float64(_)
            )
    }) {
        return false;
    }

    if plan.projections.iter().any(|proj| match &proj.expr {
        GpuExpr::Column(name) => !agg
            .group_by
            .iter()
            .any(|group_col| group_col.eq_ignore_ascii_case(name)),
        GpuExpr::AggCall {
            func: AggFunc::Sum, ..
        } => false,
        _ => true,
    }) {
        return false;
    }

    let mut joined_sources = vec![execution_order[0]];
    for &src_idx in execution_order.iter().skip(1) {
        let Some((build_table_idx, build_col_name, _probe_col_name)) =
            find_join_for_source(plan, pinned_tables, src_idx, &joined_sources)
        else {
            return false;
        };
        if !pinned_tables[build_table_idx].is_known_unique(build_col_name.as_str()) {
            return false;
        }
        joined_sources.push(src_idx);
    }

    build_representative_materialize_plan(plan, pinned_tables, agg, &topk).is_some()
}
