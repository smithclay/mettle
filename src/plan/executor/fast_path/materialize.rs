//! Fast-path Phase 4: download results + cache + materialize output.

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use crate::metal;
use crate::pinned::PinnedTable;
use crate::processing::MaterializedQueryResult;
use crate::source_cache;

use super::super::super::helpers::{JoinResult, JoinResultLookup};
use super::super::cache::{profile_log, query_result_cache_enabled};
use super::super::execute_from_lookup;
use super::super::topk::materialize_grouped_topk_result;
use super::setup::FastPathPlan;
use super::{FastPathCtx, FastPathTopKOutput};

pub(super) fn finalize(
    ctx: &FastPathCtx<'_>,
    derived: &FastPathPlan,
    topk_output: FastPathTopKOutput,
    encode_agg_start: Instant,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let plan = ctx.plan;
    let pinned_tables = ctx.pinned_tables;
    let config = ctx.config;
    let profile = ctx.profile;

    let (representative_row_ids, aggregate_vals) = match topk_output {
        FastPathTopKOutput::Gpu {
            row_ids,
            aggregate_values,
        } => {
            profile_log(profile, "encode aggregate+topk", encode_agg_start);
            let agg_wait_start = Instant::now();
            metal::end_batch(config)?;
            profile_log(profile, "execute aggregate+topk", agg_wait_start);
            let final_count = aggregate_values.download_count();
            (
                row_ids.download_data_as(final_count),
                aggregate_values.download_data_as(final_count),
            )
        }
        FastPathTopKOutput::Cpu {
            row_ids,
            aggregate_values,
        } => (row_ids, aggregate_values),
    };

    let materialize_start = Instant::now();
    let final_count = aggregate_vals.len();
    if final_count == 0 {
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
        return execute_from_lookup(plan, &JoinResultLookup(&join_result), config);
    }
    if query_result_cache_enabled() {
        source_cache::insert_grouped_topk_result(
            derived.grouped_topk_result_cache_key.clone(),
            &representative_row_ids,
            &aggregate_vals,
        );
    }
    let result = materialize_grouped_topk_result(
        plan,
        pinned_tables,
        &derived.grouped_agg_plan,
        &derived.representative_plan,
        &representative_row_ids,
        &aggregate_vals,
    )?;
    profile_log(profile, "download+materialize", materialize_start);

    Ok(Some(result))
}
