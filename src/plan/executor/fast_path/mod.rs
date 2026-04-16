//! GPU-native fast path for grouped top-K queries (TPC-H Q3 shape).
//!
//! Decomposes the single-command-buffer filter→join→aggregate→top-K pipeline
//! into phase-specific helpers. Control flow bifurcates into three kinds of
//! exits at various points:
//!   * `PhaseFlow::Continue(T)` — produced a value, proceed.
//!   * `PhaseFlow::Bail` — this fast path cannot handle the query; the
//!     orchestrator returns `Ok(None)`. When bailing mid-pipeline, the
//!     helper is responsible for calling `metal::end_batch` if a batch is
//!     in flight.
//!   * `PhaseFlow::Fallback(result)` — a materialized result is ready via
//!     the generic `execute_from_lookup` path.

mod aggregate;
mod filter;
mod join_loop;
mod materialize;
mod setup;

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use crate::config::BufferConfig;
use crate::metal::{self, GpuVector};
use crate::pinned::PinnedTable;
use crate::processing::MaterializedQueryResult;

use super::super::ir::{GpuPlan, SourceSelection};
use super::cache::{profile_enabled, profile_log};

/// Shared read-only inputs threaded through every fast-path phase.
pub(super) struct FastPathCtx<'a> {
    pub plan: &'a GpuPlan,
    pub pinned_tables: &'a [Arc<PinnedTable>],
    pub config: &'a BufferConfig,
    pub profile: bool,
    pub table_refs: Vec<(String, Arc<PinnedTable>)>,
}

/// Three-way phase outcome; see module docs.
pub(super) enum PhaseFlow<T> {
    Continue(T),
    Bail,
    Fallback(MaterializedQueryResult),
}

/// Result of the top-K phase: on-device row IDs and aggregate values.
pub(super) struct FastPathTopKOutput {
    pub row_ids: GpuVector,
    pub aggregate_values: GpuVector,
}

/// GPU-native pipeline: encode the entire filter-join chain into one Metal
/// command buffer (one waitUntilCompleted). Falls back to None for unsupported
/// shapes. After the GPU phase, downloads post-join index arrays and delegates
/// to the existing CPU pipeline for expression eval / aggregate / sort / limit.
pub(super) fn execute_gpu_native(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    source_selections: Option<&[SourceSelection]>,
    execution_order: &[usize],
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let profile = profile_enabled();
    let table_refs: Vec<(String, Arc<PinnedTable>)> = plan
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

    let ctx = FastPathCtx {
        plan,
        pinned_tables,
        config,
        profile,
        table_refs,
    };

    // ── Setup: derive fast-path plan, check result cache, validate ───────
    let derived = match setup::derive_plan(&ctx, execution_order)? {
        PhaseFlow::Continue(v) => v,
        PhaseFlow::Bail => return Ok(None),
        PhaseFlow::Fallback(r) => return Ok(Some(r)),
    };

    let encode_join_start = Instant::now();
    metal::begin_batch(config)?;

    // ── Phase 1: per-table filter → GpuVector indices ────────────────────
    let mut gv_indices = match filter::build_filter_gv_indices(&ctx, source_selections)? {
        PhaseFlow::Continue(v) => v,
        PhaseFlow::Bail => return Ok(None),
        PhaseFlow::Fallback(r) => return Ok(Some(r)),
    };

    // ── Phase 2: join loop ───────────────────────────────────────────────
    let join_out = match join_loop::run_join_loop(&ctx, &derived, &mut gv_indices)? {
        PhaseFlow::Continue(v) => v,
        PhaseFlow::Bail => return Ok(None),
        PhaseFlow::Fallback(r) => return Ok(Some(r)),
    };
    profile_log(profile, "encode filter+join", encode_join_start);

    let join_wait_start = Instant::now();
    metal::end_batch(config)?;
    profile_log(profile, "execute filter+join", join_wait_start);

    // ── Phase 3a: finalize join, handle empty/fallback ───────────────────
    let finalized = match join_loop::finalize_join(&ctx, &derived, &mut gv_indices, join_out)? {
        PhaseFlow::Continue(v) => v,
        PhaseFlow::Bail => return Ok(None),
        PhaseFlow::Fallback(r) => return Ok(Some(r)),
    };

    // ── Phase 3b: project sum expression ─────────────────────────────────
    let encode_agg_start = Instant::now();
    metal::begin_batch(config)?;
    let sum_gv = match aggregate::compute_sum_gv(&ctx, &derived, &finalized, &gv_indices)? {
        PhaseFlow::Continue(v) => v,
        PhaseFlow::Bail => return Ok(None),
        PhaseFlow::Fallback(r) => return Ok(Some(r)),
    };

    // ── Phase 3c: top-K ──────────────────────────────────────────────────
    let topk_output =
        match aggregate::run_topk(&ctx, &derived, &finalized, &gv_indices, sum_gv.as_ref())? {
            PhaseFlow::Continue(v) => v,
            PhaseFlow::Bail => return Ok(None),
            PhaseFlow::Fallback(r) => return Ok(Some(r)),
        };

    // ── Phase 4: download + materialize ──────────────────────────────────
    materialize::finalize(&ctx, &derived, topk_output, encode_agg_start)
}
