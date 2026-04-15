EXTENSION_NAME := mettle
DUCKDB_VERSION := 1.5.1
DUCKDB_PLATFORM := osx_arm64
PYTHON := .venv/bin/python
EXTENSION_VERSION := v0.1.0
DEBUG_LIBRARY := target/debug/lib$(EXTENSION_NAME).dylib
DEBUG_EXTENSION := target/debug/$(EXTENSION_NAME).duckdb_extension
RELEASE_LIBRARY := target/release/lib$(EXTENSION_NAME).dylib
RELEASE_EXTENSION := target/release/$(EXTENSION_NAME).duckdb_extension

SF ?= 0.1

.PHONY: configure debug release test test_rust test_debug test_release package package_debug package_release clean bench_data bench

configure:
	uv venv .venv
	uv pip install --python .venv duckdb==$(DUCKDB_VERSION)

debug:
	cargo build --features loadable-extension

release:
	cargo build --release --features loadable-extension

test: test_rust test_debug

test_rust:
	cargo test

package_debug: debug
	cargo duckdb-ext package -i $(DEBUG_LIBRARY) -o $(DEBUG_EXTENSION) -v $(EXTENSION_VERSION) -p $(DUCKDB_PLATFORM) -d v$(DUCKDB_VERSION)

package_release: release
	cargo duckdb-ext package -i $(RELEASE_LIBRARY) -o $(RELEASE_EXTENSION) -v $(EXTENSION_VERSION) -p $(DUCKDB_PLATFORM) -d v$(DUCKDB_VERSION)

test_debug: package_debug
	$(PYTHON) test/run_sql_tests.py --extension $(DEBUG_EXTENSION) test/sql

test_release: package_release
	$(PYTHON) test/run_sql_tests.py --extension $(RELEASE_EXTENSION) test/sql

package: package_release

clean:
	cargo clean

bench_data:
	uv run python test/generate_tpch.py $(SF)

bench: bench_data
	cargo run --release --example bench_q3 -- $(SF)
