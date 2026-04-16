//! Fast-path setup: cache probe + derived plan + input validation.

use std::error::Error;
use std::time::Instant;

use crate::pinned::PinnedColumn;
use crate::source_cache;
use crate::state;

use super::super::super::aggregate::build_grouped_aggregate_plan;
use super::super::super::ir::{GroupedAggregatePlan, JoinExecutionStep};
use super::super::cache::{
    grouped_topk_result_cache_key, join_output_cache_key, profile_log, query_result_cache_enabled,
};
use super::super::join_plan::{join_execution_steps, live_sources_after_join_steps};
use super::super::topk::{
    build_representative_materialize_plan, candidate_grouped_topk_plan, is_grouped_topk_pilot,
    materialize_grouped_topk_result, CandidateGroupedTopKPlan, GroupedTopKPlan,
    RepresentativeMaterializePlan,
};
use super::{FastPathCtx, PhaseFlow};

/// All plan-derived artifacts needed by later phases.
pub(super) struct FastPathPlan {
    pub grouped_topk: GroupedTopKPlan,
    pub grouped_agg_plan: GroupedAggregatePlan,
    pub representative_plan: RepresentativeMaterializePlan,
    pub candidate_topk: Option<CandidateGroupedTopKPlan>,
    pub join_steps: Vec<JoinExecutionStep>,
    pub active_join_step_count: usize,
    pub active_execution_order: Vec<usize>,
    pub live_after_steps: Vec<Vec<bool>>,
    pub join_output_cache_key: Option<String>,
    pub grouped_topk_result_cache_key: String,
    pub use_candidate_topk: bool,
}

impl FastPathPlan {
    pub fn active_join_steps(&self) -> &[JoinExecutionStep] {
        &self.join_steps[..self.active_join_step_count]
    }
}

pub(super) fn derive_plan(
    ctx: &FastPathCtx<'_>,
    execution_order: &[usize],
) -> Result<PhaseFlow<FastPathPlan>, Box<dyn Error>> {
    let plan = ctx.plan;
    let pinned_tables = ctx.pinned_tables;

    if plan.joins.is_empty() || plan.sources.len() < 2 {
        return Ok(PhaseFlow::Bail);
    }
    let Some(grouped_topk) = super::super::topk::grouped_topk_plan(plan) else {
        return Ok(PhaseFlow::Bail);
    };
    if !is_grouped_topk_pilot(plan, pinned_tables, execution_order) {
        return Ok(PhaseFlow::Bail);
    }
    let agg = plan
        .aggregate
        .as_ref()
        .expect("pilot path requires aggregate");
    let Some(grouped_agg_plan) = build_grouped_aggregate_plan(plan, agg) else {
        return Ok(PhaseFlow::Bail);
    };
    let Some(representative_plan) =
        build_representative_materialize_plan(plan, pinned_tables, agg, &grouped_topk)
    else {
        return Ok(PhaseFlow::Bail);
    };
    let grouped_topk_result_cache_key =
        grouped_topk_result_cache_key(plan, execution_order, representative_plan.table_idx);
    if query_result_cache_enabled() {
        if let Some(cached) = source_cache::get_grouped_topk_result(&grouped_topk_result_cache_key)
        {
            let materialize_start = Instant::now();
            state::set_last_execution_mode("metal_hot_result_cache");
            let result = materialize_grouped_topk_result(
                plan,
                pinned_tables,
                &grouped_agg_plan,
                &representative_plan,
                cached.representative_row_ids.as_slice(),
                cached.aggregate_values.as_slice(),
            )?;
            profile_log(ctx.profile, "download+materialize", materialize_start);
            return Ok(PhaseFlow::Fallback(result));
        }
    }
    let Some(join_steps) = join_execution_steps(plan, pinned_tables, execution_order) else {
        return Ok(PhaseFlow::Bail);
    };
    let candidate_topk = candidate_grouped_topk_plan(
        plan,
        pinned_tables,
        execution_order,
        &grouped_agg_plan,
        &representative_plan,
    );
    let use_candidate_topk = candidate_topk.is_some();
    let join_output_cache_key = if use_candidate_topk {
        candidate_topk.as_ref().map(|candidate_topk| {
            join_output_cache_key(
                plan,
                execution_order,
                representative_plan.table_idx,
                candidate_topk.aggregate_source_idx,
            )
        })
    } else {
        None
    };
    let active_join_step_count = if use_candidate_topk {
        join_steps.len().saturating_sub(1)
    } else {
        join_steps.len()
    };
    let active_execution_order = execution_order[..active_join_step_count + 1].to_vec();
    let live_after_steps = live_sources_after_join_steps(
        plan,
        pinned_tables,
        &active_execution_order,
        &join_steps[..active_join_step_count],
        &[representative_plan.table_idx],
    );
    let representative_table = &pinned_tables[representative_plan.table_idx];
    if representative_table
        .column(&representative_plan.secondary_order_column)
        .is_none()
    {
        return Err("fast path representative secondary-order column missing".into());
    }

    // Validate: all join key columns must be integer types.
    for join in &plan.joins {
        for col_name in [&join.left_col, &join.right_col] {
            let ok = pinned_tables.iter().any(|t| {
                matches!(
                    t.column(col_name),
                    Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_) | PinnedColumn::Int64(_))
                )
            });
            if !ok {
                return Ok(PhaseFlow::Bail);
            }
        }
    }

    Ok(PhaseFlow::Continue(FastPathPlan {
        grouped_topk,
        grouped_agg_plan,
        representative_plan,
        candidate_topk,
        join_steps,
        active_join_step_count,
        active_execution_order,
        live_after_steps,
        join_output_cache_key,
        grouped_topk_result_cache_key,
        use_candidate_topk,
    }))
}

