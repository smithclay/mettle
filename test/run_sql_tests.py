#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import duckdb


@dataclass
class StatementOk:
    line: int
    sql: str


@dataclass
class StatementError:
    line: int
    expected_substring: str
    sql: str


@dataclass
class Query:
    line: int
    column_types: str
    sort_mode: str
    sql: str
    expected_rows: list[str]


Record = StatementOk | StatementError | Query


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True, help="Path to the packaged .duckdb_extension file")
    parser.add_argument(
        "--run-file",
        help="Internal mode: run a single .test file in a fresh Python process",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["test/sql"],
        help="Test files or directories containing DuckDB-style .test files",
    )
    args = parser.parse_args()

    extension_path = Path(args.extension).resolve()
    if args.run_file:
        run_test_file(Path(args.run_file).resolve(), extension_path)
        return 0

    test_files = discover_test_files(args.paths)
    if not test_files:
        raise SystemExit("no DuckDB-style SQL test files were found")

    for test_file in test_files:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--extension",
                str(extension_path),
                "--run-file",
                str(test_file),
            ],
            check=True,
        )

    print(f"duckdb-style SQL tests passed: {len(test_files)} file(s)")
    return 0


def discover_test_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            files.append(path.resolve())
            continue
        if path.is_dir():
            files.extend(sorted(candidate.resolve() for candidate in path.rglob("*.test")))
            continue
        raise SystemExit(f"test path does not exist: {path}")
    return files


def run_test_file(test_file: Path, extension_path: Path) -> None:
    connection = duckdb.connect(
        database=":memory:",
        config={"allow_unsigned_extensions": "true"},
    )
    connection.execute(f"LOAD {sql_quote(str(extension_path))}")

    records = parse_fixture(test_file.read_text())
    for record in records:
        if isinstance(record, StatementOk):
            execute_statement_ok(connection, record)
            continue
        if isinstance(record, StatementError):
            execute_statement_error(connection, record)
            continue
        execute_query(connection, record)

    print(f"ok: {test_file}")


def parse_fixture(fixture: str) -> list[Record]:
    lines = fixture.splitlines()
    cursor = 0
    records: list[Record] = []

    while cursor < len(lines):
        header = lines[cursor].strip()
        if not header or header.startswith("#"):
            cursor += 1
            continue

        line_number = cursor + 1
        if header == "statement ok":
            cursor += 1
            records.append(StatementOk(line_number, collect_statement_sql(lines, cursor, line_number)))
            cursor = advance_statement(lines, cursor)
            continue

        if header.startswith("statement error "):
            cursor += 1
            records.append(
                StatementError(
                    line_number,
                    header.removeprefix("statement error ").strip(),
                    collect_statement_sql(lines, cursor, line_number),
                )
            )
            cursor = advance_statement(lines, cursor)
            continue

        if header.startswith("query "):
            header_parts = header.removeprefix("query ").split()
            if not header_parts:
                raise AssertionError(f"fixture line {line_number}: query header is missing column types")
            column_types = header_parts[0]
            sort_mode = header_parts[1] if len(header_parts) > 1 else "nosort"
            if sort_mode not in {"nosort", "rowsort"}:
                raise AssertionError(f"fixture line {line_number}: unsupported sort mode {sort_mode!r}")

            cursor += 1
            sql_lines: list[str] = []
            while cursor < len(lines) and lines[cursor].strip() != "----":
                sql_lines.append(lines[cursor])
                cursor += 1
            if cursor >= len(lines):
                raise AssertionError(f"fixture line {line_number}: query is missing ---- separator")
            if not sql_lines:
                raise AssertionError(f"fixture line {line_number}: query is missing SQL")
            cursor += 1

            expected_rows: list[str] = []
            while cursor < len(lines) and lines[cursor].strip():
                expected_rows.append(lines[cursor].strip())
                cursor += 1
            if cursor < len(lines) and not lines[cursor].strip():
                cursor += 1

            records.append(Query(line_number, column_types, sort_mode, "\n".join(sql_lines), expected_rows))
            continue

        raise AssertionError(f"fixture line {line_number}: unsupported record header {header!r}")

    return records


def collect_statement_sql(lines: list[str], cursor: int, line_number: int) -> str:
    sql_lines: list[str] = []
    while cursor < len(lines) and lines[cursor].strip():
        sql_lines.append(lines[cursor])
        cursor += 1
    if not sql_lines:
        raise AssertionError(f"fixture line {line_number}: statement is missing SQL")
    return "\n".join(sql_lines)


def advance_statement(lines: list[str], cursor: int) -> int:
    while cursor < len(lines) and lines[cursor].strip():
        cursor += 1
    if cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    return cursor


def execute_statement_ok(connection: duckdb.DuckDBPyConnection, record: StatementOk) -> None:
    try:
        connection.execute(record.sql).fetchall()
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"fixture line {record.line}: statement failed: {exc}\n{record.sql}"
        ) from exc


def execute_statement_error(connection: duckdb.DuckDBPyConnection, record: StatementError) -> None:
    try:
        connection.execute(record.sql).fetchall()
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if record.expected_substring not in message:
            raise AssertionError(
                f"fixture line {record.line}: expected error containing "
                f"{record.expected_substring!r}, got {message!r}\n{record.sql}"
            ) from exc
        return

    raise AssertionError(f"fixture line {record.line}: statement unexpectedly succeeded\n{record.sql}")


def execute_query(connection: duckdb.DuckDBPyConnection, record: Query) -> None:
    try:
        cursor = connection.execute(record.sql)
        rows = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"fixture line {record.line}: query failed: {exc}\n{record.sql}") from exc

    column_count = len(cursor.description or [])
    if column_count != len(record.column_types):
        raise AssertionError(
            f"fixture line {record.line}: query declared {record.column_types!r} "
            f"but returned {column_count} columns\n{record.sql}"
        )

    actual_rows = [" ".join(render_value(value) for value in row) for row in rows]
    expected_rows = list(record.expected_rows)
    if record.sort_mode == "rowsort":
        actual_rows.sort()
        expected_rows.sort()

    if actual_rows != expected_rows:
        raise AssertionError(
            f"fixture line {record.line}: result mismatch\n"
            f"expected={expected_rows!r}\nactual={actual_rows!r}\n{record.sql}"
        )


def render_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_real(value)
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    return str(value)


def format_real(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0")
    if rendered.endswith("."):
        rendered += "0"
    return rendered or "0.0"


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
