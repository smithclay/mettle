# mettle

`mettle` is a DuckDB Rust extension template aimed at Sirius-style GPU control flow on Apple platforms:

- `CALL gpu_buffer_init(...)` configures a Metal-backed buffer manager.
- `CALL gpu_processing(...)` follows the Sirius README pattern: initialize once, then submit ordinary SQL through a single entrypoint that either offloads a supported subset to Metal or falls back to DuckDB CPU execution.

This repo no longer carries the previous Substrait executor or the DuckDB `substrait`/`arrow` extension workflow. The current implementation targets a Sirius-compatible subset: it validates Metal availability, stores GPU memory budgets, stages submitted SQL through a Metal buffer, rewrites supported expressions onto Metal-backed scalar UDFs, and then returns the real DuckDB result set through the same `CALL gpu_processing(...)` interface.

## Build

```sh
make configure
make debug
make test_debug
make test_release
```

To package a `.duckdb_extension`, install `cargo-duckdb-ext-tools` and run:

```sh
cargo install cargo-duckdb-ext-tools
make package
```

## Usage

Load the extension into DuckDB, then initialize GPU budgets:

```sql
LOAD './target/release/mettle.duckdb_extension';
CALL gpu_buffer_init('1 GB', '2 GB');
CALL gpu_buffer_init('1 GB', '2 GB', pinned_memory_size = '4 GB');
```

Submit SQL text into the Metal path and return actual rows:

```sql
CALL gpu_processing('SELECT 42 AS answer');
```

For the supported Metal subset, the user-facing query stays in the Sirius shape:

```sql
CALL gpu_processing('SELECT value, value * 10 AS scaled FROM numbers ORDER BY value');
CALL gpu_processing('SELECT value, value + 5 AS shifted FROM numbers ORDER BY value');
CALL gpu_processing('SELECT value, value * 10 AS scaled FROM numbers WHERE value > 1 ORDER BY value');
```

Current scope is a Sirius-style subset:

- General row-producing `SELECT`/CTE queries execute through DuckDB after Metal staging and return real rows.
- The configured `cache_region` now backs a real Metal input-buffer cache. Repeated warm runs over the same column values reuse GPU-resident buffers instead of re-uploading every batch.
- A broader but still narrow query shape is actually offloaded to Metal today for single-table queries: optional `WHERE col {=,<,<=,>,>=} scalar`, optional numeric `col * scalar` or `col + scalar` projection, and optional `ORDER BY col`.
- The current real Metal-backed type coverage is `INTEGER`, `BIGINT`, `FLOAT`, plus `DATE` and `TIMESTAMP` filters.
- More complex queries currently fall back to DuckDB CPU execution after Metal staging, matching the graceful-fallback pattern described in the Sirius README.
- Internally, `gpu_processing` is registered as a DuckDB macro over `query(...)`, while `gpu_rewrite_query`, `gpu_add_scalar`, `gpu_mul_scalar`, and the `gpu_{eq,lt,le,gt,ge}_scalar` predicates provide the Metal-aware planning and execution hooks.
- DuckDB-style SQL tests live in `test/sql/*.test`, including a Sirius-shaped `tpch-sirius.test`, and can be run with `make test_debug` or `make test_release`, following the extension-template-rs pattern.
