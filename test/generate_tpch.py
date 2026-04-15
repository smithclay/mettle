#!/usr/bin/env python3
"""Generate TPC-H tables as Parquet files for the mettle benchmark harness.

Usage:
    uv run python test/generate_tpch.py 0.01
    uv run python test/generate_tpch.py 0.1
    uv run python test/generate_tpch.py 1

Writes ``test/data/tpch-sf{SF}/{customer,orders,lineitem}.parquet``. Idempotent:
re-running for a scale factor that already exists is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb


TABLES = ("customer", "orders", "lineitem")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: generate_tpch.py <scale_factor>", file=sys.stderr)
        return 2

    scale_factor_arg = sys.argv[1]
    try:
        scale_factor = float(scale_factor_arg)
    except ValueError:
        print(f"scale factor must be numeric, got {scale_factor_arg!r}", file=sys.stderr)
        return 2

    out_dir = Path(__file__).parent / "data" / f"tpch-sf{scale_factor_arg}"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [table for table in TABLES if not (out_dir / f"{table}.parquet").exists()]
    if not missing:
        print(f"already generated: {out_dir}")
        return 0

    print(f"generating TPC-H sf={scale_factor} into {out_dir}")
    connection = duckdb.connect(":memory:")
    connection.execute("INSTALL tpch")
    connection.execute("LOAD tpch")
    connection.execute(f"CALL dbgen(sf={scale_factor})")

    for table in missing:
        target = out_dir / f"{table}.parquet"
        connection.execute(
            f"COPY {table} TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        rows = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  wrote {target.name} ({rows} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
