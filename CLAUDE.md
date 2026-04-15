# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Mettle is a DuckDB Rust extension that offloads a subset of SQL execution to Apple Metal GPUs. It follows a "Sirius-style" pattern: initialize a Metal buffer manager with `gpu_buffer_init(...)`, then submit SQL through `gpu_processing(...)` which either runs supported operations on Metal or falls back to DuckDB CPU execution.

## Build & Test Commands

```sh
make configure          # create .venv with correct duckdb version
make debug              # cargo build --features loadable-extension
make release            # cargo build --release --features loadable-extension
cargo test              # Rust integration tests (tests/extension.rs)
cargo test <test_name>  # run a single Rust test
make test_debug         # package debug extension + run SQL fixture tests
make test_release       # package release extension + run SQL fixture tests
```

Packaging a `.duckdb_extension` requires `cargo-duckdb-ext-tools` (`cargo install cargo-duckdb-ext-tools`).

## Architecture

### Extension Entry Point

`src/lib.rs` — registers two table functions with DuckDB: `gpu_buffer_init` and `gpu_processing`. Uses the `duckdb` crate's `VTab` trait for both. The `loadable-extension` feature gate controls the C API entrypoint for loading as a `.duckdb_extension`.

### Core Modules

- **`src/table_functions.rs`** — `GpuBufferInitTable` VTab. Parses human-readable size strings, validates Metal availability, initializes the Metal runtime with cache/processing/pinned memory budgets.
- **`src/processing.rs`** — `GpuProcessingTable` VTab. This is the query execution engine: parses submitted SQL, decides whether to run on Metal or fall back to CPU, materializes results as Arrow `RecordBatch`es, and streams them back through DuckDB's table function protocol. Contains the SQL pattern matching logic that recognizes supported query shapes (simple scalar projections, TPC-H Q3).
- **`src/metal.rs`** — Metal GPU runtime. Manages the device, command queue, compute pipelines, shader compilation, input-buffer cache (LRU with cache stats), and kernel dispatch. All `objc2-metal` usage is behind `#[cfg(target_os = "macos")]`. Exposes `ScalarProjectOp` (Add/Multiply) and `ScalarCompareOp` (Eq/Lt/Le/Gt/Ge) for the operations that have Metal kernels.
- **`src/state.rs`** — Global mutable state: buffer config, catalog connection, and last execution mode. Uses `OnceLock`/`RwLock`/`Mutex`. The Metal runtime itself is held in a `thread_local!` inside `src/metal/mod.rs` (each DuckDB worker thread lazily constructs its own `MetalRuntime`); cache stats and `reset_runtime_cache()` therefore observe only the current thread's runtime.
- **`src/config.rs`** — `BufferConfig` struct for cache/processing/pinned memory region sizes with human-readable size parsing.
- **`src/error.rs`** — `MettleError` enum and `Result` type alias.

### Execution Flow

`gpu_processing(sql)` → parse SQL → pattern-match against supported shapes → if supported, run Metal kernels (filter/project/pipeline) → otherwise fall back to `query(sql)` on the catalog connection → return Arrow batches. The execution mode is recorded in global state (`last_execution_mode()`).

### Test Infrastructure

- **`tests/extension.rs`** — Rust integration tests that use `register_extension()` directly (no loadable extension packaging). Tests are serialized with a global `Mutex` and call `reset_runtime_state_for_tests()` before each test because of shared global Metal state.
- **`test/sql/*.test`** — DuckDB-style SQL fixture files, run by `test/run_sql_tests.py` which loads the packaged `.duckdb_extension` into a Python duckdb connection. Format supports `statement ok`, `statement error <substring>`, and `query <types> [rowsort]` directives.

### Platform Constraints

Metal code is macOS-only (`#[cfg(target_os = "macos")]`). On non-macOS, `gpu_buffer_init` returns an error. The `dev-dependencies` section uses `duckdb` with `bundled` feature for tests, while the main dependency links against an external DuckDB for the loadable extension build.
