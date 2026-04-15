//! Fast-path Phase 3b (sum projection) + Phase 3c (top-K).

use std::error::Error;
use std::time::Instant;

use crate::metal::{self, GpuVector};

use super::super::super::helpers::{evaluate_expr_gv, resolve_column_gpu};
use super::super::super::ir::{AggFunc, AggregateInputPlan, AggregateSlotPlan, GpuJoinStrategy};
use super::super::cache::{profile_enabled, profile_log};
use super::super::choose_gpu_join_strategy;
use super::super::topk::{grouped_candidate_better, ExperimentalGroupedTopKMode};
use super::join_loop::FinalizedJoin;
use super::setup::FastPathPlan;
use super::{FastPathCtx, FastPathTopKOutput, PhaseFlow};

pub(super) fn compute_sum_gv(
    ctx: &FastPathCtx<'_>,
    derived: &FastPathPlan,
    finalized: &FinalizedJoin,
    gv_indices: &[metal::GpuVector],
) -> Result<PhaseFlow<Option<GpuVector>>, Box<dyn Error>> {
    let config = ctx.config;
    if !finalized.needs_materialized_sum {
        return Ok(PhaseFlow::Continue(None));
    }

    let sum_expr = match derived.grouped_agg_plan.aggregate_slots.first() {
        Some(AggregateSlotPlan {
            func: AggFunc::Sum,
            input: AggregateInputPlan::Expr(expr),
            ..
        }) => expr,
        _ => {
            metal::end_batch(config)?;
            return Ok(PhaseFlow::Bail);
        }
    };

    let sum_gv = if let Some((price_name, discount_name)) = &finalized.fused_revenue_columns {
        let (price_tidx, price_col) = match resolve_column_gpu(price_name, &ctx.table_refs) {
            Some(r) => r,
            None => {
                metal::end_batch(config)?;
                return Ok(PhaseFlow::Bail);
            }
        };
        let (discount_tidx, discount_col) =
            match resolve_column_gpu(discount_name, &ctx.table_refs) {
                Some(r) => r,
                None => {
                    metal::end_batch(config)?;
                    return Ok(PhaseFlow::Bail);
                }
            };
        if price_tidx != discount_tidx {
            metal::end_batch(config)?;
            return Ok(PhaseFlow::Bail);
        }
        let price_gpu = metal::column_to_gpu_buffer(price_col, config)?;
        let discount_gpu = metal::column_to_gpu_buffer(discount_col, config)?;
        metal::project_revenue_from_indices_gv(
            &price_gpu,
            &discount_gpu,
            &gv_indices[price_tidx],
            config,
        )?
    } else {
        let Some(sum_gv) = evaluate_expr_gv(sum_expr, &ctx.table_refs, gv_indices, config)? else {
            metal::end_batch(config)?;
            return Ok(PhaseFlow::Bail);
        };
        sum_gv
    };

    Ok(PhaseFlow::Continue(Some(sum_gv)))
}

pub(super) fn run_topk(
    ctx: &FastPathCtx<'_>,
    derived: &FastPathPlan,
    finalized: &FinalizedJoin,
    gv_indices: &[metal::GpuVector],
    sum_gv: Option<&GpuVector>,
    encode_agg_start: Instant,
) -> Result<PhaseFlow<FastPathTopKOutput>, Box<dyn Error>> {
    let pinned_tables = ctx.pinned_tables;
    let config = ctx.config;
    let profile = ctx.profile;
    let grouped_topk = &derived.grouped_topk;
    let representative_plan = &derived.representative_plan;
    let representative_table = &pinned_tables[representative_plan.table_idx];
    let secondary_order_gpu = &finalized.secondary_order_gpu;

    let output = if let Some(candidate_topk) = &derived.candidate_topk {
        let candidate_row_ids = finalized
            .candidate_representative_row_ids
            .as_ref()
            .expect("candidate grouped top-k requires representative candidates");
        let candidate_secondary_order_values =
            metal::gather_i32_gv(secondary_order_gpu, candidate_row_ids, config)?;

        let representative_key_col = representative_table
            .column(&candidate_topk.representative_key_column)
            .ok_or_else(|| "candidate top-k representative key column missing".to_owned())?;
        let aggregate_key_col = pinned_tables[candidate_topk.aggregate_source_idx]
            .column(&candidate_topk.aggregate_key_column)
            .ok_or_else(|| "candidate top-k aggregate key column missing".to_owned())?;
        let representative_key_gpu = metal::column_to_gpu_buffer(representative_key_col, config)?;
        let aggregate_key_gpu = metal::column_to_gpu_buffer(aggregate_key_col, config)?;
        let join_strategy = choose_gpu_join_strategy(
            representative_table,
            &candidate_topk.representative_key_column,
            candidate_row_ids.max_len,
        )?;

        let matches = match join_strategy {
            GpuJoinStrategy::Hash { ht_capacity } => {
                let build_keys =
                    metal::gather_i32_gv(&representative_key_gpu, candidate_row_ids, config)?;
                let probe_keys = metal::gather_i32_gv(
                    &aggregate_key_gpu,
                    &gv_indices[candidate_topk.aggregate_source_idx],
                    config,
                )?;
                let (ht_k, ht_i) = metal::hash_join_build_gv(&build_keys, ht_capacity, config)?;
                metal::hash_join_probe_gv(&probe_keys, &ht_k, &ht_i, ht_capacity, config)?
            }
            GpuJoinStrategy::DenseLookup {
                min_key,
                slot_count,
            } => {
                let lookup_rows = metal::dense_lookup_build_from_indices_i32_gv(
                    &representative_key_gpu,
                    candidate_row_ids,
                    min_key,
                    slot_count,
                    config,
                )?;
                metal::dense_lookup_probe_from_indices_i32_gv(
                    &aggregate_key_gpu,
                    &gv_indices[candidate_topk.aggregate_source_idx],
                    &lookup_rows,
                    min_key,
                    slot_count,
                    config,
                )?
            }
        };
        match finalized.candidate_mode {
            Some(ExperimentalGroupedTopKMode::SlotValuesGpuTopK) => {
                let dense_sums = metal::accumulate_dense_slots_gv(
                    &matches,
                    sum_gv.expect("slot values path requires materialized aggregate input"),
                    candidate_row_ids,
                    config,
                )?;
                let dense_rows = metal::GpuDenseGroupedRows {
                    representative_row_ids: candidate_row_ids.clone(),
                    secondary_order_values: candidate_secondary_order_values,
                    aggregate_values: dense_sums,
                };
                let dense_topk = metal::topk_dense_grouped_rows(
                    &dense_rows,
                    grouped_topk.limit,
                    grouped_topk.secondary_order_group_idx.is_some(),
                    config,
                )?;
                FastPathTopKOutput::Gpu {
                    row_ids: dense_topk.representative_row_ids,
                    aggregate_values: dense_topk.aggregate_values,
                }
            }
            Some(ExperimentalGroupedTopKMode::SlotFusedGpuTopK)
            | Some(ExperimentalGroupedTopKMode::SlotFusedCpuTopK) => {
                let Some((price_name, discount_name)) = &finalized.fused_revenue_columns else {
                    metal::end_batch(config)?;
                    return Ok(PhaseFlow::Bail);
                };
                let (price_tidx, price_col) = match resolve_column_gpu(price_name, &ctx.table_refs)
                {
                    Some(r) => r,
                    None => {
                        metal::end_batch(config)?;
                        return Ok(PhaseFlow::Bail);
                    }
                };
                let (discount_tidx, discount_col) =
                    match resolve_column_gpu(discount_name, &ctx.table_refs) {
                        Some(r) => r,
                        None => {
                            metal::end_batch(config)?;
                            return Ok(PhaseFlow::Bail);
                        }
                    };
                if price_tidx != discount_tidx
                    || price_tidx != candidate_topk.aggregate_source_idx
                {
                    metal::end_batch(config)?;
                    return Ok(PhaseFlow::Bail);
                }
                let price_gpu = metal::column_to_gpu_buffer(price_col, config)?;
                let discount_gpu = metal::column_to_gpu_buffer(discount_col, config)?;
                let dense_sums = metal::accumulate_revenue_dense_slots_from_indices_gv(
                    &matches,
                    &price_gpu,
                    &discount_gpu,
                    &gv_indices[price_tidx],
                    candidate_row_ids,
                    config,
                )?;
                let use_secondary = grouped_topk.secondary_order_group_idx.is_some();
                if finalized.candidate_mode == Some(ExperimentalGroupedTopKMode::SlotFusedCpuTopK) {
                    profile_log(profile, "encode aggregate+topk", encode_agg_start);
                    let agg_wait_start = Instant::now();
                    metal::end_batch(config)?;
                    profile_log(profile, "execute aggregate+topk", agg_wait_start);

                    let dense_count = dense_sums.download_count();
                    let representative_row_ids: Vec<u32> =
                        candidate_row_ids.download_data_as(dense_count);
                    let secondary_values: Vec<i32> =
                        candidate_secondary_order_values.download_data_as(dense_count);
                    let aggregate_values: Vec<f32> = dense_sums.download_data_as(dense_count);

                    let mut ranked: Vec<(u32, f32, i32)> = representative_row_ids
                        .into_iter()
                        .zip(aggregate_values)
                        .zip(secondary_values)
                        .map(|((row_id, sum), secondary)| (row_id, sum, secondary))
                        .filter(|(_, sum, _)| *sum != 0.0)
                        .collect();
                    ranked.sort_by(|lhs, rhs| {
                        let lhs_better = grouped_candidate_better(
                            lhs.1,
                            lhs.2,
                            lhs.0,
                            rhs.1,
                            rhs.2,
                            rhs.0,
                            use_secondary,
                        );
                        let rhs_better = grouped_candidate_better(
                            rhs.1,
                            rhs.2,
                            rhs.0,
                            lhs.1,
                            lhs.2,
                            lhs.0,
                            use_secondary,
                        );
                        match (lhs_better, rhs_better) {
                            (true, false) => std::cmp::Ordering::Less,
                            (false, true) => std::cmp::Ordering::Greater,
                            _ => std::cmp::Ordering::Equal,
                        }
                    });
                    ranked.truncate(grouped_topk.limit);
                    FastPathTopKOutput::Cpu {
                        row_ids: ranked.iter().map(|(row_id, _, _)| *row_id).collect(),
                        aggregate_values: ranked.iter().map(|(_, sum, _)| *sum).collect(),
                    }
                } else {
                    let dense_rows = metal::GpuDenseGroupedRows {
                        representative_row_ids: candidate_row_ids.clone(),
                        secondary_order_values: candidate_secondary_order_values,
                        aggregate_values: dense_sums,
                    };
                    let dense_topk = metal::topk_dense_grouped_rows(
                        &dense_rows,
                        grouped_topk.limit,
                        use_secondary,
                        config,
                    )?;
                    FastPathTopKOutput::Gpu {
                        row_ids: dense_topk.representative_row_ids,
                        aggregate_values: dense_topk.aggregate_values,
                    }
                }
            }
            Some(ExperimentalGroupedTopKMode::CandidateCompact) | None => {
                let diag = profile_enabled();
                // [DIAG] flush candidate hash_join phase
                if diag {
                    let t = Instant::now();
                    metal::end_batch(config)?;
                    eprintln!("[DIAG B: candidate gather+hash_join]  {:?}", t.elapsed());
                    metal::begin_batch(config)?;
                }
                let (probe_kept, build_kept) = metal::compact_matches_gv(&matches, config)?;
                let candidate_sums = metal::reindex_gv(
                    sum_gv.expect("candidate compact path requires materialized aggregate input"),
                    &probe_kept,
                    config,
                )?;
                // [DIAG] flush compact+reindex
                if diag {
                    let t = Instant::now();
                    metal::end_batch(config)?;
                    eprintln!("[DIAG C: compact_matches+reindex]  {:?}", t.elapsed());
                    eprintln!(
                        "[DIAG   build_kept.max_len={} probe_kept.max_len={} matches.max_len={}",
                        build_kept.max_len, probe_kept.max_len, matches.max_len
                    );
                    metal::begin_batch(config)?;
                }
                let dense_topk = metal::aggregate_topk_grouped_rows_gv(
                    &build_kept,
                    &candidate_secondary_order_values.data,
                    &candidate_sums,
                    grouped_topk.limit,
                    grouped_topk.secondary_order_group_idx.is_some(),
                    config,
                )?;
                // [DIAG] flush agg_topk core
                if diag {
                    let t = Instant::now();
                    metal::end_batch(config)?;
                    eprintln!("[DIAG D: aggregate_topk_grouped_rows_gv]  {:?}", t.elapsed());
                    metal::begin_batch(config)?;
                }
                let actual_row_ids = metal::reindex_gv(
                    candidate_row_ids,
                    &dense_topk.representative_row_ids,
                    config,
                )?;
                FastPathTopKOutput::Gpu {
                    row_ids: actual_row_ids,
                    aggregate_values: dense_topk.aggregate_values,
                }
            }
            Some(ExperimentalGroupedTopKMode::GenericGroupedAggregate) => unreachable!(),
        }
    } else {
        let topk_rows = metal::aggregate_topk_grouped_rows_gv(
            &gv_indices[representative_plan.table_idx],
            secondary_order_gpu,
            sum_gv.expect("default grouped top-k path requires materialized aggregate input"),
            grouped_topk.limit,
            grouped_topk.secondary_order_group_idx.is_some(),
            config,
        )?;
        FastPathTopKOutput::Gpu {
            row_ids: topk_rows.representative_row_ids,
            aggregate_values: topk_rows.aggregate_values,
        }
    };

    Ok(PhaseFlow::Continue(output))
}
