//! Join strategy selection, execution-order planning, liveness analysis, and
//! per-source selection construction.

use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;

use crate::config::BufferConfig;
use crate::host_plan::HostPlanHints;
use crate::pinned::PinnedTable;
use crate::source_cache;

use super::super::helpers::{compute_filter_indices, refs_eq};
use super::super::ir::{
    GpuExpr, GpuJoinStrategy, GpuPlan, JoinExecutionStep, SourceSelection,
};
use super::cache::source_filter_cache_key;
use super::resolve::{
    apply_predicate_transfer, resolve_column_name_in_sources, resolve_join_endpoint, source_filters,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeGpuShape {
    SingleTableGroupedAggregate,
    JoinedGroupedAggregate,
    JoinedGroupedTopKPilot,
}

pub(in crate::plan) fn choose_gpu_join_strategy(
    build_table: &PinnedTable,
    build_col_name: &str,
    build_len: usize,
) -> Result<GpuJoinStrategy, Box<dyn Error>> {
    if build_table.is_known_unique(build_col_name) && build_len >= 1_024 {
        if let Some((min_key, max_key)) = build_table.integer_range_i32(build_col_name) {
            let span = i64::from(max_key) - i64::from(min_key) + 1;
            if span > 0 {
                let slot_count =
                    usize::try_from(span).map_err(|_| "dense join span overflowed usize")?;
                if slot_count <= (1 << 24) && slot_count <= build_len.saturating_mul(16).max(4_096)
                {
                    let slot_count = u32::try_from(slot_count)
                        .map_err(|_| "dense join span overflowed u32".to_owned())?;
                    return Ok(GpuJoinStrategy::DenseLookup {
                        min_key,
                        slot_count,
                    });
                }
            }
        }
    }

    Ok(GpuJoinStrategy::Hash {
        ht_capacity: hash_join_capacity(build_len)?,
    })
}

pub(super) fn execution_source_order(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    host_hints: Option<&HostPlanHints>,
) -> Vec<usize> {
    let mut candidate = Vec::with_capacity(plan.sources.len());
    let mut used = HashSet::with_capacity(plan.sources.len());

    if let Some(hints) = host_hints {
        for hinted_source in &hints.preferred_sources {
            if let Some((idx, _)) = plan.sources.iter().enumerate().find(|(_, source)| {
                refs_eq(source.name.as_str(), hinted_source)
                    || source
                        .alias
                        .as_deref()
                        .is_some_and(|alias| refs_eq(alias, hinted_source))
            }) {
                if used.insert(idx) {
                    candidate.push(idx);
                }
            }
        }
    }

    for idx in 0..plan.sources.len() {
        if used.insert(idx) {
            candidate.push(idx);
        }
    }

    if order_supports_unique_builds(plan, pinned_tables, &candidate) {
        candidate
    } else {
        (0..plan.sources.len()).collect()
    }
}

pub(in crate::plan) fn join_execution_steps(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> Option<Vec<JoinExecutionStep>> {
    if execution_order.is_empty() {
        return Some(Vec::new());
    }

    let mut steps = Vec::with_capacity(execution_order.len().saturating_sub(1));
    let mut joined_sources = vec![execution_order[0]];
    for &probe_source_idx in execution_order.iter().skip(1) {
        let (build_source_idx, build_col_name, probe_col_name) =
            find_join_for_source(plan, pinned_tables, probe_source_idx, &joined_sources)?;
        steps.push(JoinExecutionStep {
            build_source_idx,
            probe_source_idx,
            build_col_name,
            probe_col_name,
        });
        joined_sources.push(probe_source_idx);
    }
    Some(steps)
}

pub(super) fn find_join_for_source(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    probe_source_idx: usize,
    joined_sources: &[usize],
) -> Option<(usize, String, String)> {
    for join in &plan.joins {
        let (left_idx, left_col) =
            resolve_join_endpoint(&join.left_table, &join.left_col, plan, pinned_tables)?;
        let (right_idx, right_col) =
            resolve_join_endpoint(&join.right_table, &join.right_col, plan, pinned_tables)?;

        if left_idx == probe_source_idx && joined_sources.contains(&right_idx) {
            return Some((right_idx, right_col, left_col));
        }
        if right_idx == probe_source_idx && joined_sources.contains(&left_idx) {
            return Some((left_idx, left_col, right_col));
        }
    }
    None
}

fn mark_column_source(
    name: &str,
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    live_sources: &mut [bool],
) {
    if name == "*" {
        return;
    }
    if let Some((table_idx, _)) = resolve_column_name_in_sources(name, plan, pinned_tables) {
        live_sources[table_idx] = true;
    }
}

fn mark_expr_sources(
    expr: &GpuExpr,
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    live_sources: &mut [bool],
) {
    match expr {
        GpuExpr::Column(name) => mark_column_source(name, plan, pinned_tables, live_sources),
        GpuExpr::Literal(_) => {}
        GpuExpr::BinaryOp { left, right, .. } => {
            mark_expr_sources(left, plan, pinned_tables, live_sources);
            mark_expr_sources(right, plan, pinned_tables, live_sources);
        }
        GpuExpr::AggCall { arg, .. } => mark_expr_sources(arg, plan, pinned_tables, live_sources),
    }
}

fn output_live_sources(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    extra_live_sources: &[usize],
) -> Vec<bool> {
    let mut live_sources = vec![false; plan.sources.len()];

    for projection in &plan.projections {
        mark_expr_sources(&projection.expr, plan, pinned_tables, &mut live_sources);
    }

    if let Some(agg) = &plan.aggregate {
        for group_col in &agg.group_by {
            mark_column_source(group_col, plan, pinned_tables, &mut live_sources);
        }
    }

    for order_by in &plan.order_by {
        mark_column_source(&order_by.column, plan, pinned_tables, &mut live_sources);
    }

    for &source_idx in extra_live_sources {
        if source_idx < live_sources.len() {
            live_sources[source_idx] = true;
        }
    }

    live_sources
}

pub(in crate::plan) fn live_sources_after_join_steps(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
    join_steps: &[JoinExecutionStep],
    extra_live_sources: &[usize],
) -> Vec<Vec<bool>> {
    let mut needed_after = output_live_sources(plan, pinned_tables, extra_live_sources);
    let mut live_after_steps = vec![vec![false; plan.sources.len()]; join_steps.len()];

    for step_idx in (0..join_steps.len()).rev() {
        let mut live = needed_after.clone();
        let joined_after_step = &execution_order[..=step_idx + 1];
        if !joined_after_step.iter().any(|&source_idx| live[source_idx]) {
            live[join_steps[step_idx].probe_source_idx] = true;
        }
        live_after_steps[step_idx] = live;
        needed_after[join_steps[step_idx].build_source_idx] = true;
    }

    live_after_steps
}

fn order_supports_unique_builds(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> bool {
    let Some(join_steps) = join_execution_steps(plan, pinned_tables, execution_order) else {
        return false;
    };
    if execution_order.is_empty() {
        return false;
    }
    for step in &join_steps {
        let build_table_idx = step.build_source_idx;
        let build_col_name = &step.build_col_name;
        if !pinned_tables[build_table_idx].is_known_unique(build_col_name.as_str()) {
            return false;
        }
    }
    true
}

pub(super) fn build_source_selections(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    config: &BufferConfig,
) -> Result<Vec<SourceSelection>, Box<dyn Error>> {
    let mut local_indices: Vec<Vec<usize>> = Vec::with_capacity(plan.sources.len());
    for (src_idx, src) in plan.sources.iter().enumerate() {
        let table = &pinned_tables[src_idx];
        let src_ref = src.alias.as_deref().unwrap_or(src.name.as_str());
        let src_filters = source_filters(plan, table, src_ref);
        let indices = if src_filters.is_empty() {
            (0..table.row_count).collect()
        } else if let Some(cache_key) = source_filter_cache_key(table, &src_filters) {
            if let Some(cached) = source_cache::get_local_indices(&cache_key) {
                cached.iter().map(|&value| value as usize).collect()
            } else {
                let indices = compute_filter_indices(table, &src_filters, config)?;
                let cached_indices: Option<Vec<u32>> = indices
                    .iter()
                    .map(|&value| u32::try_from(value).ok())
                    .collect();
                if let Some(cached_indices) = cached_indices {
                    source_cache::insert_local_indices(cache_key, &cached_indices);
                }
                indices
            }
        } else {
            compute_filter_indices(table, &src_filters, config)?
        };
        local_indices.push(indices);
    }

    let mut final_indices = local_indices.clone();
    apply_predicate_transfer(plan, pinned_tables, &mut final_indices);

    Ok(local_indices
        .into_iter()
        .zip(final_indices)
        .map(|(local_indices, final_indices)| SourceSelection {
            local_indices,
            final_indices,
        })
        .collect())
}

pub(super) fn hash_join_capacity(build_len: usize) -> Result<u32, Box<dyn Error>> {
    u32::try_from((build_len.saturating_mul(2).max(16)).next_power_of_two())
        .map_err(|_| "hash join capacity overflowed u32".to_owned().into())
}

pub(super) fn membership_filter_bit_count(build_len: usize) -> u32 {
    let target_bits = build_len.saturating_mul(8).max(4_096);
    let rounded = target_bits.next_power_of_two().min(1 << 24);
    u32::try_from(rounded).unwrap_or(1 << 24)
}

pub(super) fn should_apply_resident_membership_filter(build_len: usize, probe_len: usize) -> bool {
    std::env::var_os("METTLE_EXPERIMENTAL_JOIN_PREFILTER").is_some()
        && build_len >= 1_024
        && probe_len >= 65_536
        && probe_len >= build_len.saturating_mul(4)
}
