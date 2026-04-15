# Current State

## Snapshot
- `mettle` is a DuckDB Rust extension with Sirius-style entrypoints: `gpu_buffer_init(...)` and `gpu_processing(...)`.
- A generic SQL → GPU plan builder (`GpuPlan` IR via `sqlparser`) handles arbitrary single-table and multi-table join queries, not just a hardcoded Q3 subset.
- The GPU-native pipeline keeps data on Metal from filter through expression eval. CPU only does the final HashMap aggregate + sort.

## Architecture

### Module Layout
```
src/
  lib.rs              — DuckDB extension entry (VTab registrations)
  table_functions.rs   — gpu_buffer_init VTab
  processing.rs        — gpu_processing VTab (try plan → try execute → fallback)
  plan.rs              — GpuPlan IR, SQL parser, GPU-native executor (~2300 lines)
  metal/
    mod.rs             — public API, types, with_runtime (~400 lines)
    gpu_buffer.rs      — GpuBuffer, GpuVector types (~115 lines)
    shaders.rs         — MSL kernel source + compile/encode helpers (~650 lines)
    runtime.rs         — MetalRuntime struct + all GPU methods (~1950 lines)
  pinned.rs            — GPU-resident table cache
  config.rs            — BufferConfig
  error.rs             — MettleError
  state.rs             — global runtime state
```

### Key Abstractions

**GpuBuffer** — type-erased Metal buffer handle (`Retained<ProtocolObject<dyn MTLBuffer>>` + len + element_size).

**GpuVector** — GPU-resident variable-length array. `data: GpuBuffer` + `count: GpuBuffer` (single u32 on GPU) + `max_len: usize` (dispatch grid size). The count stays on GPU between operations — CPU only reads it after `end_batch()`. Downstream kernels read `count[0]` and early-exit threads beyond it.

**GpuPlan** — flat pipeline IR: sources → filters → joins → projections → aggregate → order_by → limit. Parsed from SQL via `sqlparser` (DuckDB dialect).

**ColumnLookup** trait — returns `(&PinnedColumn, &[usize])` for late materialization. `JoinResultLookup` implements it over `JoinResult`.

### GPU-Native Pipeline (`execute_gpu_native`)

The primary execution path for multi-table join queries:

```
begin_batch()
  GPU: compare filters → mask → filter_to_indices → GpuVector indices (per table)
  CPU: string filters → upload indices as GpuVector
  GPU: gather join keys → hash_join_build → hash_join_probe → compact_matches → reindex
end_batch()                          ← WAIT (read post-join count)
trim gv_indices max_len to count     ← metadata only, no data copy
begin_batch()
  GPU: gather output columns → expression eval (project_f32_scalar, elementwise_f32)
end_batch()                          ← WAIT
CPU: download typed columns (~100K × 4B each)
CPU: HashMap aggregate + sort + limit
```

Two GPU waits total. No intermediate data re-upload. The trim between batches is a `usize` assignment — the GPU buffers already contain valid data.

### What Runs On Metal Today
- Scalar compare kernels: `compare_{i32,i64,f32}_scalar` with Eq/Lt/Le/Gt/Ge
- Scalar projection kernels: `project_{i32,i64,f32}_scalar` with Add/Multiply/Subtract/SubtractFrom
- Element-wise f32 arithmetic: `elementwise_f32`
- GPU hash join: `hash_join_build_i32` + `hash_join_probe_i32` (open-addressing, atomic CAS)
- GPU hash aggregate (single i32 key): `hash_agg_sum_f32` (atomic CAS float accumulation)
- GPU composite-key grouped aggregate: `grouped_aggregate_exact_gv` supports up to 4 i32 key lanes with SUM/MIN/MAX/AVG/COUNT (via `grouped_aggregate_build_keys_v`)
- GPU grouped top-K pilot path: `hash_agg_grouped_sum_v` + `compact_grouped_rows` + `topk_dense_grouped_rows_parallel_v` (128-thread parallel top-K for LIMIT ≤ 20; falls back to single-thread `topk_dense_grouped_rows` for larger LIMIT) — fuses grouped aggregate with ORDER BY/LIMIT. This is the path Q3 takes.
- Filter compaction: `and_masks`, `filter_to_indices` (atomic counter)
- Index operations: `gather_i32`, `gather_f32`, `reindex`, `compact_matches`
- Count-passing variants (`_v` suffix): 10 kernels that read length from a GPU buffer instead of a CPU constant — enables single-command-buffer pipelines
- `gather_i32_as_f32_v`: cast-during-gather for integer columns in expressions
- Command buffer batching: `begin_batch`/`end_batch` with per-operation `commit_if_standalone`
- Pointer-keyed column GPU cache (no per-query hashing)
- Query result cache (`METTLE_DISABLE_QUERY_CACHE=1` to bypass) — hot repeats of the same plan return cached materialized results in ~0.1 ms
- DECIMAL columns pinned as Float32 (not Float64) — enables GPU-resident expression eval

### Pinning
- `CALL gpu_register_table('tablename')` pins all columns via `PinnedColumn` enum (Int32, Int64, Float32, Float64, Date32, Utf8).
- DECIMAL/NUMERIC columns are cast to FLOAT at pin time (`build_cast_select`), stored as `PinnedColumn::Float32`.
- Float64 columns (native DOUBLE) are converted to f32 during GPU upload via `column_to_gpu_buffer`.

### Fallback Paths
- Single-table queries: `try_execute` wraps as trivial `JoinResult`, calls `execute_from_lookup`.
- `execute_gpu_native` returns `Ok(None)` for unsupported shapes (non-integer join keys, etc.) → falls through to `execute_multi_table` (CPU-orchestrated joins with per-operation GPU calls).
- `execute_multi_table` returns `Ok(None)` → falls through to DuckDB CPU execution via the catalog connection.

## Supported Surface Area
- Arbitrary single-table queries with filters, projections, expressions, aggregates, ORDER BY, LIMIT.
- Multi-table equi-joins (comma-FROM and explicit `JOIN ... ON`) with integer/date join keys.
- Compound expressions: `price * (1 - discount)`, nested `BinaryOp` trees.
- Grouped aggregates: SUM, COUNT, MIN, MAX, AVG with single or composite group-by keys.
- String equality filters (`c_mktsegment = 'BUILDING'`) via CPU fallback.
- TPC-H Q3 end-to-end on Metal with correct results verified against DuckDB.

## Verification
- `cargo test` — 37 tests (21 unit + 16 integration).
- `cargo run --release --example bench_q3 -- <sf>` — TPC-H Q3 benchmark with DuckDB comparison.
- DuckDB-style SQL fixtures under `test/sql/*.test`.

## Benchmark Numbers (release build, M-series laptop)

Warm median over 8 runs. "pinned" = after `gpu_register_table` for all three TPC-H tables.

| SF   | lineitem rows | DuckDB  | Mettle pinned | vs DuckDB | Journey |
|------|---------------|---------|---------------|-----------|---------|
| 0.01 | 60,175        | 1.1 ms  | **0.80 ms**   | **1.35x** | was 25ms → 1.7ms → 0.92ms → 0.80ms |
| 0.1  | 600,572       | 2.2 ms  | **1.49 ms**   | **1.50x** | was 3.7ms → 2.1ms → 1.49ms |
| 1.0  | 6,001,215     | 6.7 ms  | **5.98 ms**   | **1.12x** | was 37ms → 24.5ms → 16.3ms → 5.98ms |

**Mettle now beats DuckDB at every scale.** SF=1 went from 2.4x slower to 1.12x faster after replacing the single-threaded `topk_dense_grouped_rows` kernel with a parallel version (`topk_dense_grouped_rows_parallel_v`, 128 threads, strided scan + threadgroup-memory merge). That change alone took Q3 SF=1 from 16.3 ms to 5.98 ms — the top-K kernel had been consuming 9.3 ms (93%) of the aggregate+topk phase before.

### SF=1 Q3 Per-Phase Breakdown (warm, `METTLE_PROFILE=1`, cache disabled) — after parallel top-K

The aggregate+topk phase dropped from ~9.5 ms to ~0.5 ms. Filter+join is now the larger remaining cost; nothing else is dominant.

## Remaining Gaps
- No multi-core CPU parallelism (DuckDB uses all cores).
- Composite-key grouped aggregate supports only i32 key lanes (up to 4). Int64/Float/String group keys fall back to CPU HashMap in `execute_aggregate_impl` (not the Q3 path). Q3's `l_orderkey` (BIGINT) is handled by the separate grouped-topk-pilot path which uses representative row ids, not by the general composite-key aggregate.
- String filters are CPU-only.
- No GPU-native table scan/decode path (relies on pinning).
- Over-dispatch within join batches: max_len = pre-filter table size even though actual count is much smaller after filters. Indirect dispatch would fix this but adds Metal API complexity.
- `topk_dense_grouped_rows_parallel_v` caps at LIMIT ≤ 20 (threadgroup-memory budget at 128 threads × 20 × 12 B). Larger limits fall back to the slow single-thread kernel. Most analytic queries use LIMIT ≤ 10, but this is a latent cliff.

## Next Useful Milestones
- Trim filter+join over-dispatch: aggregate phase is already trim-sized, but the filter+join batch still dispatches at pre-filter table sizes (~6M threads for lineitem at SF=1). Could shave another 1–2 ms at SF=1.
- Expand `grouped_aggregate_exact_gv` to accept Int64 keys via 2×i32 lane splitting (benefits non-Q3 shapes that currently fall back to CPU HashMap).
- Extend parallel top-K to larger limits via two-kernel split (per-threadgroup partial → global merge), removing the LIMIT ≤ 20 cliff.
- Expand query coverage beyond Q3-shaped queries.
- Consider indirect dispatch for truly single-command-buffer pipelines at scale.
