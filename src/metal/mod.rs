mod gpu_buffer;
mod runtime;
mod shaders;

pub use gpu_buffer::{GpuBuffer, GpuVector};
use runtime::MetalRuntime;

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::BufferConfig;
use crate::error::{MettleError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub registry_id: u64,
    pub unified_memory: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuerySubmission {
    pub device: MetalDeviceInfo,
    pub query_bytes: u64,
    pub fingerprint_hex: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetalCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bypasses: u64,
    pub entries: u64,
    pub cached_bytes: u64,
    pub input_entries: u64,
    pub input_cached_bytes: u64,
    pub selection_entries: u64,
    pub selection_cached_bytes: u64,
    pub join_result_entries: u64,
    pub join_result_cached_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarProjectOp {
    Add,
    Multiply,
    Subtract,
    SubtractFrom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarCompareOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

pub struct GpuGroupedTopKRows {
    pub representative_row_ids: GpuVector,
    pub aggregate_values: GpuVector,
}

pub struct GpuDenseGroupedRows {
    pub representative_row_ids: GpuVector,
    pub secondary_order_values: GpuVector,
    pub aggregate_values: GpuVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuGroupedAggregateStateKind {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

pub struct GpuGroupedAggregateSpec<'a> {
    pub state_kind: GpuGroupedAggregateStateKind,
    pub values: Option<&'a GpuVector>,
}

pub struct GpuGroupedAggregateSlotResult {
    pub values: Option<Vec<f32>>,
    pub counts: Option<Vec<u32>>,
}

pub struct GpuGroupedAggregateRows {
    pub key_lanes: Vec<Vec<i32>>,
    pub slots: Vec<GpuGroupedAggregateSlotResult>,
}

pub fn validate_runtime() -> Result<MetalDeviceInfo> {
    MetalRuntime::new(0).map(|runtime| runtime.info)
}

pub fn stage_query(query: &str, config: &BufferConfig) -> Result<QuerySubmission> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.stage_query(query, config)
    })
}

pub fn reset_runtime_cache() {
    CACHE_HITS.store(0, Ordering::Relaxed);
    CACHE_MISSES.store(0, Ordering::Relaxed);
    CACHE_EVICTIONS.store(0, Ordering::Relaxed);
    CACHE_BYPASSES.store(0, Ordering::Relaxed);
    RUNTIME_GENERATION.fetch_add(1, Ordering::Relaxed);

    #[cfg(target_os = "macos")]
    SHARED_RUNTIME.with(|runtime| {
        *runtime.borrow_mut() = None;
    });
}

pub fn cache_stats() -> MetalCacheStats {
    let mut stats = MetalCacheStats {
        hits: CACHE_HITS.load(Ordering::Relaxed),
        misses: CACHE_MISSES.load(Ordering::Relaxed),
        evictions: CACHE_EVICTIONS.load(Ordering::Relaxed),
        bypasses: CACHE_BYPASSES.load(Ordering::Relaxed),
        ..MetalCacheStats::default()
    };

    #[cfg(target_os = "macos")]
    SHARED_RUNTIME.with(|runtime| {
        if let Some(runtime) = runtime.borrow().as_ref() {
            let detail = runtime.cache_stats();
            stats.entries = detail.entries;
            stats.cached_bytes = detail.cached_bytes;
            stats.input_entries = detail.input_entries;
            stats.input_cached_bytes = detail.input_cached_bytes;
            stats.selection_entries = detail.selection_entries;
            stats.selection_cached_bytes = detail.selection_cached_bytes;
            stats.join_result_entries = detail.join_result_entries;
            stats.join_result_cached_bytes = detail.join_result_cached_bytes;
        }
    });

    stats
}

pub fn project_f32_values(
    values: &[f32],
    scalar: f32,
    op: ScalarProjectOp,
    config: &BufferConfig,
) -> Result<Vec<f32>> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.project_f32_values(values, scalar, op)
    })
}

pub fn compare_i32_values(
    values: &[i32],
    scalar: i32,
    op: ScalarCompareOp,
    config: &BufferConfig,
) -> Result<Vec<bool>> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.compare_i32_values(values, scalar, op)
    })
}

pub fn compare_i64_values(
    values: &[i64],
    scalar: i64,
    op: ScalarCompareOp,
    config: &BufferConfig,
) -> Result<Vec<bool>> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.compare_i64_values(values, scalar, op)
    })
}

pub fn compare_f32_values(
    values: &[f32],
    scalar: f32,
    op: ScalarCompareOp,
    config: &BufferConfig,
) -> Result<Vec<bool>> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.compare_f32_values(values, scalar, op)
    })
}

pub fn compare_date32_values(
    values: &[i32],
    scalar: i32,
    op: ScalarCompareOp,
    config: &BufferConfig,
) -> Result<Vec<bool>> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.compare_date32_values(values, scalar, op)
    })
}

/// GPU-resident filter: compare column values and return mask as a GpuBuffer (u8).
/// Stays on GPU — no copy back to CPU.
pub fn compare_i32_to_gpu(
    values: &[i32],
    scalar: i32,
    op: ScalarCompareOp,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.compare_i32_to_gpu(values, scalar, op)
    })
}

pub fn compare_i64_to_gpu(
    values: &[i64],
    scalar: i64,
    op: ScalarCompareOp,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.compare_i64_to_gpu(values, scalar, op)
    })
}

pub fn compare_f32_to_gpu(
    values: &[f32],
    scalar: f32,
    op: ScalarCompareOp,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.compare_f32_to_gpu(values, scalar, op)
    })
}

pub fn compare_date32_to_gpu(
    values: &[i32],
    scalar: i32,
    op: ScalarCompareOp,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    // Date32 uses same kernel as i32.
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.compare_i32_to_gpu(values, scalar, op)
    })
}

/// AND two GPU mask buffers element-wise. Both must be u8 masks of the same length.
pub fn and_masks_gpu(a: &GpuBuffer, b: &GpuBuffer, config: &BufferConfig) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.and_masks_gpu(a, b)
    })
}

/// Convert a GPU u8 mask to a GPU u32 index array (compacted indices where mask==1).
/// Output order is non-deterministic.
pub fn filter_to_indices_gpu(mask: &GpuBuffer, config: &BufferConfig) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.filter_to_indices_gpu(mask)
    })
}

/// Begin command buffer batching. GPU operations that return GpuBuffer will
/// encode into a shared command buffer without committing, reducing
/// waitUntilCompleted stalls. Call `end_batch` to commit and wait.
pub fn begin_batch(config: &BufferConfig) -> Result<()> {
    with_runtime(config.cache_region_bytes, |runtime| runtime.begin_batch())
}

/// Commit and wait on the batched command buffer.
pub fn end_batch(config: &BufferConfig) -> Result<()> {
    with_runtime(config.cache_region_bytes, |runtime| runtime.end_batch())
}

pub fn grouped_aggregate_exact_gv(
    key_lanes: &[GpuVector],
    specs: &[GpuGroupedAggregateSpec<'_>],
    config: &BufferConfig,
) -> Result<GpuGroupedAggregateRows> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.grouped_aggregate_exact_gv(key_lanes, specs)
    })
}

/// Element-wise f64 binary operation on two vectors via GPU.
pub fn elementwise_f64(
    a: &[f64],
    b: &[f64],
    op: ScalarProjectOp,
    config: &BufferConfig,
) -> Result<Vec<f64>> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.elementwise_f64(a, b, op)
    })
}

/// GPU hash join: build a hash table from `build_keys` (PK side), then
/// probe with `probe_keys` (FK side). Returns `Vec<i32>` parallel to
/// `probe_keys`, with each entry being the build-side row index that
/// matched, or -1 for no match.
pub fn hash_join_probe_i32(
    build_keys: &[i32],
    probe_keys: &[i32],
    ht_capacity: u32,
    config: &BufferConfig,
) -> Result<Vec<i32>> {
    with_runtime(config.cache_region_bytes, |runtime| {
        runtime.hash_join_probe_i32(build_keys, probe_keys, ht_capacity)
    })
}

// ── GpuVector (count-passing) public API ────────────────────────────

pub fn filter_to_indices_gv(mask: &GpuBuffer, config: &BufferConfig) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.filter_to_indices_gv(mask)
    })
}

pub fn gather_i32_gv(
    data: &GpuBuffer,
    indices: &GpuVector,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.gather_i32_gv(data, indices)
    })
}

pub fn gather_i32_as_f32_gv(
    data: &GpuBuffer,
    indices: &GpuVector,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.gather_i32_as_f32_gv(data, indices)
    })
}

pub fn gather_f32_gv(
    data: &GpuBuffer,
    indices: &GpuVector,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.gather_f32_gv(data, indices)
    })
}

pub fn reindex_gv(
    base: &GpuVector,
    selection: &GpuVector,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.reindex_gv(base, selection)
    })
}

pub fn compact_matches_gv(
    matches: &GpuVector,
    config: &BufferConfig,
) -> Result<(GpuVector, GpuVector)> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.compact_matches_gv(matches)
    })
}

pub fn hash_join_build_gv(
    keys: &GpuVector,
    ht_capacity: u32,
    config: &BufferConfig,
) -> Result<(GpuBuffer, GpuBuffer)> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.hash_join_build_gv(keys, ht_capacity)
    })
}

pub fn hash_join_probe_gv(
    probe_keys: &GpuVector,
    ht_keys: &GpuBuffer,
    ht_idx: &GpuBuffer,
    ht_capacity: u32,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.hash_join_probe_gv(probe_keys, ht_keys, ht_idx, ht_capacity)
    })
}

pub fn dense_lookup_build_i32_gv(
    keys: &GpuVector,
    min_key: i32,
    slot_count: u32,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.dense_lookup_build_i32_gv(keys, min_key, slot_count)
    })
}

pub fn dense_lookup_probe_i32_gv(
    probe_keys: &GpuVector,
    lookup_rows: &GpuBuffer,
    min_key: i32,
    slot_count: u32,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.dense_lookup_probe_i32_gv(probe_keys, lookup_rows, min_key, slot_count)
    })
}

pub fn dense_lookup_build_from_indices_i32_gv(
    column: &GpuBuffer,
    row_ids: &GpuVector,
    min_key: i32,
    slot_count: u32,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.dense_lookup_build_from_indices_i32_gv(column, row_ids, min_key, slot_count)
    })
}

pub fn dense_lookup_probe_from_indices_i32_gv(
    column: &GpuBuffer,
    row_ids: &GpuVector,
    lookup_rows: &GpuBuffer,
    min_key: i32,
    slot_count: u32,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.dense_lookup_probe_from_indices_i32_gv(column, row_ids, lookup_rows, min_key, slot_count)
    })
}

pub fn build_membership_filter_gv(
    keys: &GpuVector,
    bit_count: u32,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.build_membership_filter_gv(keys, bit_count)
    })
}

pub fn membership_filter_mask_gv(
    probe_keys: &GpuVector,
    membership: &GpuBuffer,
    bit_count: u32,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.membership_filter_mask_gv(probe_keys, membership, bit_count)
    })
}

pub fn project_f32_scalar_gv(
    input: &GpuVector,
    scalar: f32,
    op: ScalarProjectOp,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.project_f32_scalar_gv(input, scalar, op)
    })
}

pub fn elementwise_f32_gv(
    a: &GpuVector,
    b: &GpuVector,
    op: ScalarProjectOp,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.elementwise_f32_gv(a, b, op)
    })
}

pub fn column_to_gpu_buffer(
    col: &crate::pinned::PinnedColumn,
    config: &BufferConfig,
) -> Result<GpuBuffer> {
    with_runtime(config.cache_region_bytes, |rt| rt.column_to_gpu_buffer(col))
}

pub fn upload_gv(data: &[u32], config: &BufferConfig) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        GpuVector::from_upload(&rt.device, data)
    })
}

pub fn get_selection_cache_gv(key: &str, config: &BufferConfig) -> Option<GpuVector> {
    match with_runtime(config.cache_region_bytes, |rt| {
        Ok(rt.get_selection_cache_gv(key))
    }) {
        Ok(v) => v,
        Err(err) => {
            log_cache_error("get_selection_cache_gv", &err);
            None
        }
    }
}

pub fn put_selection_cache_gv(key: String, vector: &GpuVector, config: &BufferConfig) {
    if let Err(err) = with_runtime(config.cache_region_bytes, |rt| {
        rt.put_selection_cache_gv(key, vector);
        Ok(())
    }) {
        log_cache_error("put_selection_cache_gv", &err);
    }
}

pub fn get_join_result_cache_gv(
    key: &str,
    config: &BufferConfig,
) -> Option<(GpuVector, GpuVector)> {
    match with_runtime(config.cache_region_bytes, |rt| {
        Ok(rt.get_join_result_cache_gv(key))
    }) {
        Ok(v) => v,
        Err(err) => {
            log_cache_error("get_join_result_cache_gv", &err);
            None
        }
    }
}

pub fn put_join_result_cache_gv(
    key: String,
    representative_row_ids: &GpuVector,
    aggregate_row_ids: &GpuVector,
    config: &BufferConfig,
) {
    if let Err(err) = with_runtime(config.cache_region_bytes, |rt| {
        rt.put_join_result_cache_gv(key, representative_row_ids, aggregate_row_ids);
        Ok(())
    }) {
        log_cache_error("put_join_result_cache_gv", &err);
    }
}

fn log_cache_error(op: &str, err: &MettleError) {
    if std::env::var_os("METTLE_DEBUG").is_some() || std::env::var_os("METTLE_PROFILE").is_some() {
        eprintln!("mettle: Metal cache op {op} failed: {err}");
    }
}

pub fn project_revenue_from_indices_gv(
    extendedprice: &GpuBuffer,
    discount: &GpuBuffer,
    indices: &GpuVector,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.project_revenue_from_indices_gv(extendedprice, discount, indices)
    })
}

pub fn accumulate_dense_slots_gv(
    slot_ids: &GpuVector,
    values: &GpuVector,
    dense_rows: &GpuVector,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.accumulate_dense_slots_gv(slot_ids, values, dense_rows)
    })
}

pub fn accumulate_revenue_dense_slots_from_indices_gv(
    slot_ids: &GpuVector,
    extendedprice: &GpuBuffer,
    discount: &GpuBuffer,
    row_ids: &GpuVector,
    dense_rows: &GpuVector,
    config: &BufferConfig,
) -> Result<GpuVector> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.accumulate_revenue_dense_slots_from_indices_gv(
            slot_ids,
            extendedprice,
            discount,
            row_ids,
            dense_rows,
        )
    })
}

pub fn aggregate_topk_grouped_rows_gv(
    representative_row_ids: &GpuVector,
    secondary_order_values: &GpuBuffer,
    values: &GpuVector,
    limit: usize,
    use_secondary_order: bool,
    config: &BufferConfig,
) -> Result<GpuGroupedTopKRows> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.aggregate_topk_grouped_rows_gv(
            representative_row_ids,
            secondary_order_values,
            values,
            limit,
            use_secondary_order,
        )
    })
}

pub fn topk_dense_grouped_rows(
    rows: &GpuDenseGroupedRows,
    limit: usize,
    use_secondary_order: bool,
    config: &BufferConfig,
) -> Result<GpuGroupedTopKRows> {
    with_runtime(config.cache_region_bytes, |rt| {
        rt.topk_dense_grouped_rows(rows, limit, use_secondary_order)
    })
}

#[cfg(target_os = "macos")]
thread_local! {
    static SHARED_RUNTIME: RefCell<Option<MetalRuntime>> = const { RefCell::new(None) };
}

static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static CACHE_EVICTIONS: AtomicU64 = AtomicU64::new(0);
static CACHE_BYPASSES: AtomicU64 = AtomicU64::new(0);
static RUNTIME_GENERATION: AtomicU64 = AtomicU64::new(1);

#[cfg(target_os = "macos")]
fn with_runtime<T>(
    cache_region_bytes: u64,
    f: impl FnOnce(&mut MetalRuntime) -> Result<T>,
) -> Result<T> {
    SHARED_RUNTIME.with(|runtime| {
        let mut runtime = runtime.borrow_mut();
        let generation = RUNTIME_GENERATION.load(Ordering::Relaxed);
        let should_reinitialize = runtime
            .as_ref()
            .map(|runtime| {
                runtime.cache_region_bytes != cache_region_bytes || runtime.generation != generation
            })
            .unwrap_or(true);
        if should_reinitialize {
            *runtime = Some(MetalRuntime::new(cache_region_bytes)?);
        }
        f(runtime
            .as_mut()
            .expect("shared Metal runtime was initialized"))
    })
}

#[cfg(not(target_os = "macos"))]
fn with_runtime<T>(
    _cache_region_bytes: u64,
    _f: impl FnOnce(&mut MetalRuntime) -> Result<T>,
) -> Result<T> {
    Err(MettleError::MetalUnavailable(
        "Metal is only available on Apple platforms".to_owned(),
    ))
}
