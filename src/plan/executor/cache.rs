//! Profiling hooks and cache-key construction for source/join/top-K caches.

use std::sync::Arc;
use std::time::Instant;

use crate::pinned::PinnedTable;

use super::super::ir::{GpuFilter, GpuLiteral, GpuPlan};

pub(super) fn profile_enabled() -> bool {
    std::env::var_os("METTLE_PROFILE").is_some()
}

pub(super) fn profile_log(enabled: bool, label: &str, start: Instant) {
    if enabled {
        eprintln!(
            "mettle profile: {label:<28} {:>9.3} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

pub(super) fn query_result_cache_enabled() -> bool {
    std::env::var_os("METTLE_DISABLE_QUERY_CACHE").is_none()
}

pub(super) fn source_filter_cache_key(
    table: &Arc<PinnedTable>,
    filters: &[&GpuFilter],
) -> Option<String> {
    if filters.is_empty() {
        return None;
    }

    let mut key = format!("table={:p};rows={};", Arc::as_ptr(table), table.row_count);
    for filter in filters {
        let literal = match &filter.literal {
            GpuLiteral::Int(value) => format!("i:{value}"),
            GpuLiteral::Float(value) => format!("f:{value:.6}"),
            GpuLiteral::Str(value) => format!("s:{value}"),
            GpuLiteral::Date(value) => format!("d:{value}"),
        };
        key.push_str(&format!(
            "{}:{:?}:{};",
            filter.column.to_lowercase(),
            filter.op,
            literal
        ));
    }
    Some(key)
}

pub(super) fn join_output_cache_key(
    plan: &GpuPlan,
    execution_order: &[usize],
    representative_table_idx: usize,
    aggregate_source_idx: usize,
) -> String {
    format!(
        "join-output|rep={representative_table_idx}|agg={aggregate_source_idx}|order={execution_order:?}|plan={plan:?}"
    )
}

pub(super) fn grouped_topk_result_cache_key(
    plan: &GpuPlan,
    execution_order: &[usize],
    representative_table_idx: usize,
) -> String {
    format!(
        "grouped-topk-result|rep={representative_table_idx}|order={execution_order:?}|plan={plan:?}"
    )
}
