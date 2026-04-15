//! Fast-path Phase 2 + 3a: join loop + finalize (empty-result / fallback handling).

use std::error::Error;
use std::sync::Arc;

use crate::metal;
use crate::pinned::PinnedTable;

use super::super::super::helpers::{JoinResult, JoinResultLookup};
use super::super::super::ir::GpuJoinStrategy;
use super::super::join_plan::{
    membership_filter_bit_count, should_apply_resident_membership_filter,
};
use super::super::{
    build_join_result_from_gpu_indices, choose_gpu_join_strategy, execute_from_lookup,
};
use super::setup::FastPathPlan;
use super::super::topk::ExperimentalGroupedTopKMode;
use super::{FastPathCtx, PhaseFlow};

/// Output of Phase 2: which source indices are still live, and whether we
/// replayed cached join rows.
pub(super) struct JoinLoopOutput {
    pub joined_sources: Vec<usize>,
    pub used_cached_join_rows: bool,
    pub base_source_idx: usize,
}

pub(super) fn run_join_loop(
    ctx: &FastPathCtx<'_>,
    derived: &FastPathPlan,
    gv_indices: &mut Vec<metal::GpuVector>,
) -> Result<PhaseFlow<JoinLoopOutput>, Box<dyn Error>> {
    let pinned_tables = ctx.pinned_tables;
    let config = ctx.config;

    let base_source_idx = derived.active_execution_order[0];
    let mut joined_sources = vec![base_source_idx];
    let mut cached_join_rows = if !derived.use_candidate_topk {
        derived
            .join_output_cache_key
            .as_deref()
            .and_then(|cache_key| metal::get_join_result_cache_gv(cache_key, config))
    } else {
        None
    };
    let used_cached_join_rows = if let Some((representative_rows, aggregate_rows)) =
        cached_join_rows.take()
    {
        gv_indices[derived.representative_plan.table_idx] = representative_rows;
        if let Some(candidate_topk) = &derived.candidate_topk {
            gv_indices[candidate_topk.aggregate_source_idx] = aggregate_rows;
            joined_sources = vec![
                derived.representative_plan.table_idx,
                candidate_topk.aggregate_source_idx,
            ];
        } else {
            joined_sources = vec![derived.representative_plan.table_idx];
        }
        true
    } else {
        for (step_idx, step) in derived.active_join_steps().iter().enumerate() {
            let src_idx = step.probe_source_idx;
            let build_table_idx = step.build_source_idx;
            let build_col_name = &step.build_col_name;
            let probe_col_name = &step.probe_col_name;

            let build_col = match pinned_tables[build_table_idx].column(build_col_name.as_str()) {
                Some(c) => c,
                None => {
                    metal::end_batch(config)?;
                    return Ok(PhaseFlow::Bail);
                }
            };
            let probe_col = match pinned_tables[src_idx].column(probe_col_name.as_str()) {
                Some(c) => c,
                None => {
                    metal::end_batch(config)?;
                    return Ok(PhaseFlow::Bail);
                }
            };

            let build_col_gpu = metal::column_to_gpu_buffer(build_col, config)?;
            let probe_col_gpu = metal::column_to_gpu_buffer(probe_col, config)?;

            let join_strategy = choose_gpu_join_strategy(
                &pinned_tables[build_table_idx],
                build_col_name,
                gv_indices[build_table_idx].max_len,
            )?;
            let use_prefilter = should_apply_resident_membership_filter(
                gv_indices[build_table_idx].max_len,
                gv_indices[src_idx].max_len,
            );

            let mut build_keys = None;
            let mut probe_keys = None;
            if use_prefilter || matches!(join_strategy, GpuJoinStrategy::Hash { .. }) {
                build_keys = Some(metal::gather_i32_gv(
                    &build_col_gpu,
                    &gv_indices[build_table_idx],
                    config,
                )?);
                probe_keys = Some(metal::gather_i32_gv(
                    &probe_col_gpu,
                    &gv_indices[src_idx],
                    config,
                )?);
            }

            if use_prefilter {
                let build_keys_ref = build_keys
                    .as_ref()
                    .ok_or_else(|| "resident join prefilter requires build keys".to_owned())?;
                let probe_keys_ref = probe_keys
                    .as_mut()
                    .ok_or_else(|| "resident join prefilter requires probe keys".to_owned())?;
                let bit_count = membership_filter_bit_count(build_keys_ref.max_len);
                let membership =
                    metal::build_membership_filter_gv(build_keys_ref, bit_count, config)?;
                let mask = metal::membership_filter_mask_gv(
                    probe_keys_ref,
                    &membership,
                    bit_count,
                    config,
                )?;
                let selected_probe = metal::filter_to_indices_gv(&mask, config)?;
                gv_indices[src_idx] =
                    metal::reindex_gv(&gv_indices[src_idx], &selected_probe, config)?;
                *probe_keys_ref = metal::reindex_gv(probe_keys_ref, &selected_probe, config)?;
            }

            let matched = match join_strategy {
                GpuJoinStrategy::Hash { ht_capacity } => {
                    let build_keys = build_keys
                        .as_ref()
                        .ok_or_else(|| "hash join requires gathered build keys".to_owned())?;
                    let probe_keys = probe_keys
                        .as_ref()
                        .ok_or_else(|| "hash join requires gathered probe keys".to_owned())?;
                    let (ht_k, ht_i) = metal::hash_join_build_gv(build_keys, ht_capacity, config)?;
                    metal::hash_join_probe_gv(probe_keys, &ht_k, &ht_i, ht_capacity, config)?
                }
                GpuJoinStrategy::DenseLookup {
                    min_key,
                    slot_count,
                } => {
                    if let Some(build_keys) = build_keys.as_ref() {
                        let probe_keys = probe_keys
                            .as_ref()
                            .ok_or_else(|| "dense join requires gathered probe keys".to_owned())?;
                        let lookup_rows = metal::dense_lookup_build_i32_gv(
                            build_keys, min_key, slot_count, config,
                        )?;
                        metal::dense_lookup_probe_i32_gv(
                            probe_keys,
                            &lookup_rows,
                            min_key,
                            slot_count,
                            config,
                        )?
                    } else {
                        let lookup_rows = metal::dense_lookup_build_from_indices_i32_gv(
                            &build_col_gpu,
                            &gv_indices[build_table_idx],
                            min_key,
                            slot_count,
                            config,
                        )?;
                        metal::dense_lookup_probe_from_indices_i32_gv(
                            &probe_col_gpu,
                            &gv_indices[src_idx],
                            &lookup_rows,
                            min_key,
                            slot_count,
                            config,
                        )?
                    }
                }
            };

            let (probe_kept, build_kept) = metal::compact_matches_gv(&matched, config)?;
            let live_after = &derived.live_after_steps[step_idx];
            let mut next_joined_sources = Vec::with_capacity(joined_sources.len() + 1);
            for &joined_idx in &joined_sources {
                if live_after[joined_idx] {
                    gv_indices[joined_idx] =
                        metal::reindex_gv(&gv_indices[joined_idx], &build_kept, config)?;
                    next_joined_sources.push(joined_idx);
                }
            }
            if live_after[src_idx] {
                gv_indices[src_idx] = metal::reindex_gv(&gv_indices[src_idx], &probe_kept, config)?;
                next_joined_sources.push(src_idx);
            }
            joined_sources = next_joined_sources;
        }
        false
    };

    Ok(PhaseFlow::Continue(JoinLoopOutput {
        joined_sources,
        used_cached_join_rows,
        base_source_idx,
    }))
}

/// State needed by Phase 3b/3c produced by Phase 3a.
pub(super) struct FinalizedJoin {
    #[allow(dead_code)]
    pub joined_sources: Vec<usize>,
    #[allow(dead_code)]
    pub final_row_source_idx: usize,
    pub candidate_representative_row_ids: Option<metal::GpuVector>,
    pub candidate_mode: Option<ExperimentalGroupedTopKMode>,
    pub fused_revenue_columns: Option<(String, String)>,
    pub needs_materialized_sum: bool,
    pub secondary_order_gpu: metal::GpuBuffer,
}

pub(super) fn finalize_join(
    ctx: &FastPathCtx<'_>,
    derived: &FastPathPlan,
    gv_indices: &mut [metal::GpuVector],
    out: JoinLoopOutput,
) -> Result<PhaseFlow<FinalizedJoin>, Box<dyn Error>> {
    let plan = ctx.plan;
    let pinned_tables = ctx.pinned_tables;
    let config = ctx.config;

    let JoinLoopOutput {
        joined_sources,
        used_cached_join_rows,
        base_source_idx,
    } = out;

    let final_row_source_idx = if used_cached_join_rows || derived.use_candidate_topk {
        derived.representative_plan.table_idx
    } else {
        joined_sources.first().copied().unwrap_or(base_source_idx)
    };
    let final_join_count = gv_indices[final_row_source_idx].download_count();
    if final_join_count == 0 {
        let tables: Vec<(String, Arc<PinnedTable>)> = plan
            .sources
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (
                    s.alias.as_deref().unwrap_or(&s.name).to_owned(),
                    pinned_tables[i].clone(),
                )
            })
            .collect();
        let join_result = JoinResult {
            tables,
            table_indices: vec![Vec::new(); plan.sources.len()],
        };
        let materialized = execute_from_lookup(plan, &JoinResultLookup(&join_result), config)?
            .ok_or_else(|| {
                "fast-path fallback execute_from_lookup returned None for empty join".to_owned()
            })?;
        return Ok(PhaseFlow::Fallback(materialized));
    }
    for &joined_idx in &joined_sources {
        gv_indices[joined_idx].max_len = final_join_count;
    }
    if !used_cached_join_rows {
        if let (Some(cache_key), Some(candidate_topk)) = (
            derived.join_output_cache_key.as_deref(),
            derived.candidate_topk.as_ref(),
        ) {
            metal::put_join_result_cache_gv(
                cache_key.to_owned(),
                &gv_indices[derived.representative_plan.table_idx],
                &gv_indices[candidate_topk.aggregate_source_idx],
                config,
            );
        }
    }
    let candidate_representative_row_ids = derived
        .candidate_topk
        .as_ref()
        .map(|_| gv_indices[derived.representative_plan.table_idx].clone());

    // Validate the sum slot exists with Expr input (not e.g. COUNT(*)). Uses
    // first aggregate slot; bails otherwise. This matches the original guard.
    let agg_slots = &derived.grouped_agg_plan.aggregate_slots;
    let sum_expr_ok = matches!(
        agg_slots.first(),
        Some(super::super::super::ir::AggregateSlotPlan {
            func: super::super::super::ir::AggFunc::Sum,
            input: super::super::super::ir::AggregateInputPlan::Expr(_),
            ..
        })
    );
    if !sum_expr_ok {
        metal::end_batch(config)?;
        return Ok(PhaseFlow::Bail);
    }
    let sum_expr = match &agg_slots[0].input {
        super::super::super::ir::AggregateInputPlan::Expr(expr) => expr,
        _ => unreachable!(),
    };

    let fused_revenue_columns = super::super::resolve::fused_mul_one_minus_columns(sum_expr)
        .map(|(a, b)| (a.to_owned(), b.to_owned()));
    let candidate_mode = derived.requested_candidate_mode.or_else(|| {
        if derived.candidate_topk.is_some() {
            Some(if fused_revenue_columns.is_some() {
                ExperimentalGroupedTopKMode::SlotFusedGpuTopK
            } else {
                ExperimentalGroupedTopKMode::CandidateCompact
            })
        } else {
            None
        }
    });
    if candidate_mode == Some(ExperimentalGroupedTopKMode::GenericGroupedAggregate) {
        let join_result =
            build_join_result_from_gpu_indices(plan, pinned_tables, &joined_sources, gv_indices);
        let materialized = execute_from_lookup(plan, &JoinResultLookup(&join_result), config)?
            .ok_or_else(|| {
                "fast-path fallback execute_from_lookup returned None for generic mode".to_owned()
            })?;
        return Ok(PhaseFlow::Fallback(materialized));
    }

    let needs_materialized_sum = !matches!(
        candidate_mode,
        Some(ExperimentalGroupedTopKMode::SlotFusedGpuTopK)
            | Some(ExperimentalGroupedTopKMode::SlotFusedCpuTopK)
    );

    // ── Phase 3: Project -> Aggregate -> Top-K, still on device ───────
    let representative_table = &pinned_tables[derived.representative_plan.table_idx];
    let secondary_order_col = representative_table
        .column(&derived.representative_plan.secondary_order_column)
        .ok_or_else(|| "fast path representative secondary-order column missing".to_owned())?;
    let secondary_order_gpu = metal::column_to_gpu_buffer(secondary_order_col, config)?;

    Ok(PhaseFlow::Continue(FinalizedJoin {
        joined_sources,
        final_row_source_idx,
        candidate_representative_row_ids,
        candidate_mode,
        fused_revenue_columns,
        needs_materialized_sum,
        secondary_order_gpu,
    }))
}
