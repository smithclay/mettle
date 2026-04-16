//! Fast-path Phase 1: per-source filter → `GpuVector` indices.

use std::error::Error;

use crate::metal;

use super::super::super::helpers::{compute_filter_indices, filter_runs_on_gpu, gpu_mask_for_filter};
use super::super::super::ir::SourceSelection;
use super::super::cache::source_filter_cache_key;
use super::super::resolve::source_filters;
use super::{FastPathCtx, PhaseFlow};

pub(super) fn build_filter_gv_indices(
    ctx: &FastPathCtx<'_>,
    source_selections: Option<&[SourceSelection]>,
) -> Result<PhaseFlow<Vec<metal::GpuVector>>, Box<dyn Error>> {
    let plan = ctx.plan;
    let pinned_tables = ctx.pinned_tables;
    let config = ctx.config;
    let mut gv_indices: Vec<metal::GpuVector> = Vec::new();

    for (src_idx, src) in plan.sources.iter().enumerate() {
        let table = &pinned_tables[src_idx];
        let src_ref = src.alias.as_deref().unwrap_or(src.name.as_str());
        let src_filters = source_filters(plan, ctx.pinned_tables, table, src_ref);
        let selection = source_selections.map(|selections| &selections[src_idx]);

        let has_cpu_filter = src_filters.iter().any(|f| {
            let Some(col) = table.column(&f.column) else {
                return true;
            };
            !filter_runs_on_gpu(col, f)
        });

        if selection.is_none_or(|selection| selection.final_indices == selection.local_indices)
            && !has_cpu_filter
            && !src_filters.is_empty()
        {
            let cache_key = source_filter_cache_key(table, &src_filters);
            if let Some(cache_key) = cache_key.as_deref() {
                if let Some(cached) = metal::get_selection_cache_gv(cache_key, config) {
                    gv_indices.push(cached);
                    continue;
                }
            }

            // Pure GPU filter path
            let mut gpu_mask: Option<metal::GpuBuffer> = None;
            for filter in &src_filters {
                let col = match table.column(&filter.column) {
                    Some(c) => c,
                    None => {
                        metal::end_batch(config)?;
                        return Ok(PhaseFlow::Bail);
                    }
                };
                let mask = gpu_mask_for_filter(col, filter, config)?
                    .ok_or_else(|| "unsupported GPU filter in execute_gpu_native".to_owned())?;
                gpu_mask = Some(match gpu_mask {
                    None => mask,
                    Some(prev) => metal::and_masks_gpu(&prev, &mask, config)?,
                });
            }
            let indices = metal::filter_to_indices_gv(&gpu_mask.unwrap(), config)?;
            if let Some(cache_key) = cache_key {
                metal::put_selection_cache_gv(cache_key, &indices, config);
            }
            gv_indices.push(indices);
        } else {
            // CPU filter and/or predicate transfer result.
            let indices_u32: Vec<u32> = if let Some(selection) = selection {
                selection
                    .final_indices
                    .iter()
                    .map(|&i| u32::try_from(i).map_err(|_| "row index overflowed u32".to_owned()))
                    .collect::<Result<_, _>>()?
            } else if src_filters.is_empty() {
                (0..table.row_count as u32).collect()
            } else {
                compute_filter_indices(table, &src_filters, config)?
                    .iter()
                    .map(|&i| u32::try_from(i).map_err(|_| "row index overflowed u32".to_owned()))
                    .collect::<Result<_, _>>()?
            };
            gv_indices.push(metal::upload_gv(&indices_u32, config)?);
        }
    }

    Ok(PhaseFlow::Continue(gv_indices))
}
