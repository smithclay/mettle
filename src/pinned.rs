//! GPU-resident table cache.
//!
//! Stores pre-scanned column data for any DuckDB table. Populated once
//! via `CALL gpu_register_table('...')`, read on every subsequent
//! `gpu_processing` call. Eliminates per-query DuckDB scans.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::{Arc, Mutex, OnceLock};

use duckdb::arrow::array::{
    Array, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use duckdb::arrow::datatypes::DataType;
use duckdb::Connection;

/// A single column stored in the pinned cache.
#[derive(Debug, Clone)]
pub enum PinnedColumn {
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Date32(Vec<i32>),
    DictionaryI32 {
        codes: Vec<i32>,
        values: Vec<String>,
        reverse: HashMap<String, i32>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnStats {
    Integer {
        min: i64,
        max: i64,
        distinct_count: u64,
    },
    Float {
        min: f64,
        max: f64,
        distinct_count: u64,
    },
    Dictionary {
        distinct_count: u32,
    },
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PilotColumnKind {
    Int32,
    Int64,
    Float32,
    Date32,
    DictionaryI32,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnMeta {
    pub stats: ColumnStats,
    pub pilot_kind: PilotColumnKind,
}

impl ColumnMeta {
    pub fn is_pilot_supported(&self) -> bool {
        self.pilot_kind != PilotColumnKind::Unsupported
    }

    pub fn distinct_count(&self) -> Option<u64> {
        match self.stats {
            ColumnStats::Integer { distinct_count, .. } => Some(distinct_count),
            ColumnStats::Float { distinct_count, .. } => Some(distinct_count),
            ColumnStats::Dictionary { distinct_count } => Some(u64::from(distinct_count)),
            ColumnStats::Unknown => None,
        }
    }
}

impl PinnedColumn {
    #[allow(dead_code)] // exercised by unit tests; kept for public API completeness
    pub fn as_i32(&self) -> Option<&[i32]> {
        match self {
            Self::Int32(v) | Self::Date32(v) => Some(v),
            Self::DictionaryI32 { codes, .. } => Some(codes),
            _ => None,
        }
    }

    pub fn dictionary_code(&self, value: &str) -> Option<i32> {
        match self {
            Self::DictionaryI32 { reverse, .. } => reverse.get(value).copied(),
            _ => None,
        }
    }

    #[allow(dead_code)] // exercised by unit tests; kept for public API completeness
    pub fn dictionary_value(&self, code: i32) -> Option<&str> {
        match self {
            Self::DictionaryI32 { values, .. } => usize::try_from(code)
                .ok()
                .and_then(|idx| values.get(idx))
                .map(String::as_str),
            _ => None,
        }
    }

    pub fn byte_count(&self) -> usize {
        match self {
            Self::Int32(v) | Self::Date32(v) => v.len() * 4,
            Self::Int64(v) => v.len() * 8,
            Self::Float32(v) => v.len() * 4,
            Self::Float64(v) => v.len() * 8,
            Self::DictionaryI32 { codes, values, .. } => {
                codes.len() * 4 + values.iter().map(|s| s.len()).sum::<usize>()
            }
        }
    }
}

/// A fully pinned table with named columns.
#[derive(Debug)]
pub struct PinnedTable {
    pub columns: HashMap<String, PinnedColumn>,
    pub column_meta: HashMap<String, ColumnMeta>,
    pub row_count: usize,
}

impl PinnedTable {
    pub fn column(&self, name: &str) -> Option<&PinnedColumn> {
        self.columns.get(name)
    }

    pub fn column_meta(&self, name: &str) -> Option<&ColumnMeta> {
        self.column_meta.get(name)
    }

    pub fn is_known_unique(&self, name: &str) -> bool {
        self.column_meta(name)
            .and_then(ColumnMeta::distinct_count)
            .is_some_and(|distinct_count| distinct_count as usize == self.row_count)
    }

    pub fn is_pilot_supported_column(&self, name: &str) -> bool {
        self.column_meta(name)
            .is_some_and(ColumnMeta::is_pilot_supported)
    }

    pub fn integer_range_i32(&self, name: &str) -> Option<(i32, i32)> {
        let meta = self.column_meta(name)?;
        let ColumnStats::Integer { min, max, .. } = meta.stats else {
            return None;
        };
        Some((i32::try_from(min).ok()?, i32::try_from(max).ok()?))
    }
}

/// Summary returned to the SQL caller.
#[derive(Debug, Clone)]
pub struct PinSummary {
    pub table_name: String,
    pub row_count: usize,
    pub byte_count: usize,
    /// Columns that could not be pinned (unsupported type, nulls, etc.).
    /// These columns will cause queries that reference them to fall back to CPU.
    pub skipped_columns: Vec<String>,
}

static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<PinnedTable>>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<String, Arc<PinnedTable>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Look up a pinned table by name (case-insensitive).
pub fn get(name: &str) -> Option<Arc<PinnedTable>> {
    let key = name.to_lowercase();
    registry()
        .lock()
        .expect("pinned registry poisoned")
        .get(&key)
        .cloned()
}

/// Register (pin) all columns of a DuckDB table by name.
pub fn register_table(
    connection: &Connection,
    table_name: &str,
) -> Result<PinSummary, Box<dyn Error>> {
    // Cast all DECIMAL columns to DOUBLE so Arrow gives us Float64 instead of
    // Decimal128 (which PinnedColumn doesn't support). We build the SELECT list
    // by introspecting the table schema.
    let cast_sql = build_cast_select(connection, table_name)?;
    let mut statement = connection.prepare(cast_sql.as_str())?;
    let batches = {
        let arrow = statement.query_arrow([])?;
        arrow.collect::<Vec<_>>()
    };
    let schema = statement.schema();

    let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
    let mut columns: HashMap<String, PinnedColumn> = HashMap::new();
    let mut column_meta: HashMap<String, ColumnMeta> = HashMap::new();
    let mut skipped_columns: Vec<String> = Vec::new();

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col_name = field.name().to_lowercase();
        match collect_column(&batches, col_idx, field.data_type(), row_count) {
            Ok((column, meta)) => {
                columns.insert(col_name.clone(), column);
                column_meta.insert(col_name, meta);
            }
            Err(err) => {
                // Queries that reference this column will fall back to CPU.
                // Surface it in PinSummary and log once so users can see why.
                eprintln!(
                    "mettle: gpu_register_table('{}') skipped column '{}': {err}",
                    table_name, col_name
                );
                skipped_columns.push(col_name);
            }
        }
    }

    let byte_count: usize = columns.values().map(PinnedColumn::byte_count).sum();
    let table = Arc::new(PinnedTable {
        columns,
        column_meta,
        row_count,
    });

    registry()
        .lock()
        .expect("pinned registry poisoned")
        .insert(table_name.to_lowercase(), table);

    Ok(PinSummary {
        table_name: table_name.to_owned(),
        row_count,
        byte_count,
        skipped_columns,
    })
}

/// Build a SELECT that casts DECIMAL columns to FLOAT so Arrow produces
/// Float32 instead of Decimal128. Float32 has 7 significant digits —
/// sufficient for analytics and enables GPU-resident expression eval
/// (Metal doesn't support f64 in device buffers).
fn build_cast_select(connection: &Connection, table_name: &str) -> Result<String, Box<dyn Error>> {
    let quoted_table = quote_identifier(table_name);
    let mut stmt = connection.prepare(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
    )?;
    let mut rows = stmt.query([table_name])?;
    let mut columns = Vec::new();
    while let Some(row) = rows.next()? {
        let col_name: String = row.get(0)?;
        let data_type: String = row.get(1)?;
        let quoted_col = quote_identifier(&col_name);
        if data_type.starts_with("DECIMAL") || data_type.starts_with("NUMERIC") {
            columns.push(format!("CAST({quoted_col} AS FLOAT) AS {quoted_col}"));
        } else {
            columns.push(quoted_col);
        }
    }
    if columns.is_empty() {
        // Fallback: table might not be in information_schema (e.g., views).
        Ok(format!("SELECT * FROM {quoted_table}"))
    } else {
        Ok(format!("SELECT {} FROM {quoted_table}", columns.join(", ")))
    }
}

/// Quote a SQL identifier using DuckDB's double-quote rules (internal `"` doubled).
fn quote_identifier(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

fn collect_column(
    batches: &[duckdb::arrow::record_batch::RecordBatch],
    col_idx: usize,
    data_type: &DataType,
    row_count: usize,
) -> Result<(PinnedColumn, ColumnMeta), Box<dyn Error>> {
    match data_type {
        DataType::Int32 => {
            let mut values = Vec::with_capacity(row_count);
            for batch in batches {
                let arr = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or("expected Int32Array")?;
                if arr.null_count() > 0 {
                    return Err(
                        "Int32 columns with nulls are not supported in pinned tables".into(),
                    );
                }
                values.extend_from_slice(arr.values());
            }
            Ok((
                PinnedColumn::Int32(values.clone()),
                ColumnMeta {
                    stats: integer_stats(values.iter().map(|&v| i64::from(v))),
                    pilot_kind: PilotColumnKind::Int32,
                },
            ))
        }
        DataType::Int64 => {
            let mut values = Vec::with_capacity(row_count);
            for batch in batches {
                let arr = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or("expected Int64Array")?;
                if arr.null_count() > 0 {
                    return Err(
                        "Int64 columns with nulls are not supported in pinned tables".into(),
                    );
                }
                values.extend_from_slice(arr.values());
            }
            Ok((
                PinnedColumn::Int64(values.clone()),
                ColumnMeta {
                    stats: integer_stats(values.iter().copied()),
                    pilot_kind: PilotColumnKind::Int64,
                },
            ))
        }
        DataType::Float32 => {
            let mut values = Vec::with_capacity(row_count);
            for batch in batches {
                let arr = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or("expected Float32Array")?;
                if arr.null_count() > 0 {
                    return Err(
                        "Float32 columns with nulls are not supported in pinned tables".into(),
                    );
                }
                values.extend_from_slice(arr.values());
            }
            Ok((
                PinnedColumn::Float32(values.clone()),
                ColumnMeta {
                    stats: float32_stats(&values),
                    pilot_kind: PilotColumnKind::Float32,
                },
            ))
        }
        DataType::Float64 => {
            let mut values = Vec::with_capacity(row_count);
            for batch in batches {
                let arr = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or("expected Float64Array")?;
                if arr.null_count() > 0 {
                    return Err(
                        "Float64 columns with nulls are not supported in pinned tables".into(),
                    );
                }
                values.extend_from_slice(arr.values());
            }
            Ok((
                PinnedColumn::Float64(values.clone()),
                ColumnMeta {
                    stats: float64_stats(&values),
                    pilot_kind: PilotColumnKind::Unsupported,
                },
            ))
        }
        DataType::Date32 => {
            let mut values = Vec::with_capacity(row_count);
            for batch in batches {
                let arr = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .ok_or("expected Date32Array")?;
                if arr.null_count() > 0 {
                    return Err(
                        "Date32 columns with nulls are not supported in pinned tables".into(),
                    );
                }
                values.extend_from_slice(arr.values());
            }
            Ok((
                PinnedColumn::Date32(values.clone()),
                ColumnMeta {
                    stats: integer_stats(values.iter().map(|&v| i64::from(v))),
                    pilot_kind: PilotColumnKind::Date32,
                },
            ))
        }
        DataType::Utf8 => {
            let mut reverse = HashMap::new();
            let mut values = Vec::new();
            let mut codes = Vec::with_capacity(row_count);
            for batch in batches {
                let arr = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or("expected StringArray")?;
                if arr.null_count() > 0 {
                    return Err("Utf8 columns with nulls are not supported in pinned tables".into());
                }
                for i in 0..arr.len() {
                    let value = arr.value(i);
                    let code = match reverse.get(value) {
                        Some(&code) => code,
                        None => {
                            let code = i32::try_from(values.len())
                                .map_err(|_| "dictionary code overflowed i32".to_owned())?;
                            let owned = value.to_owned();
                            values.push(owned.clone());
                            reverse.insert(owned, code);
                            code
                        }
                    };
                    codes.push(code);
                }
            }
            let distinct_count = u32::try_from(reverse.len())
                .map_err(|_| "dictionary distinct count overflowed u32".to_owned())?;
            Ok((
                PinnedColumn::DictionaryI32 {
                    codes,
                    values,
                    reverse,
                },
                ColumnMeta {
                    stats: ColumnStats::Dictionary { distinct_count },
                    pilot_kind: PilotColumnKind::DictionaryI32,
                },
            ))
        }
        // Decimal types from TPC-H are exposed as Float64 after DuckDB
        // Arrow conversion, so they should hit the Float64 branch above.
        // For any unsupported type, store as Utf8 (CPU-only fallback).
        other => Err(format!(
            "gpu_register_table: column index {col_idx} has unsupported Arrow type {other:?}"
        )
        .into()),
    }
}

fn integer_stats(values: impl Iterator<Item = i64>) -> ColumnStats {
    let mut iter = values.peekable();
    let Some(first) = iter.peek().copied() else {
        return ColumnStats::Unknown;
    };
    let mut min = first;
    let mut max = first;
    let mut distinct = HashSet::new();

    for value in iter {
        min = min.min(value);
        max = max.max(value);
        distinct.insert(value);
    }

    ColumnStats::Integer {
        min,
        max,
        distinct_count: distinct.len() as u64,
    }
}

fn float32_stats(values: &[f32]) -> ColumnStats {
    let Some(&first) = values.first() else {
        return ColumnStats::Unknown;
    };
    let mut min = first;
    let mut max = first;
    let mut distinct = HashSet::new();

    for &value in values {
        min = min.min(value);
        max = max.max(value);
        distinct.insert(value.to_bits());
    }

    ColumnStats::Float {
        min: min as f64,
        max: max as f64,
        distinct_count: distinct.len() as u64,
    }
}

fn float64_stats(values: &[f64]) -> ColumnStats {
    let Some(&first) = values.first() else {
        return ColumnStats::Unknown;
    };
    let mut min = first;
    let mut max = first;
    let mut distinct = HashSet::new();

    for &value in values {
        min = min.min(value);
        max = max.max(value);
        distinct.insert(value.to_bits());
    }

    ColumnStats::Float {
        min,
        max,
        distinct_count: distinct.len() as u64,
    }
}

/// Drop all pinned tables.
pub fn reset_for_tests() {
    if let Some(lock) = REGISTRY.get() {
        lock.lock().expect("pinned registry poisoned").clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_column_encodes_and_decodes_values() {
        let mut reverse = HashMap::new();
        reverse.insert("A".to_owned(), 0);
        reverse.insert("B".to_owned(), 1);
        let column = PinnedColumn::DictionaryI32 {
            codes: vec![0, 1, 0],
            values: vec!["A".to_owned(), "B".to_owned()],
            reverse,
        };

        assert_eq!(column.dictionary_code("A"), Some(0));
        assert_eq!(column.dictionary_code("B"), Some(1));
        assert_eq!(column.dictionary_code("C"), None);
        assert_eq!(column.dictionary_value(0), Some("A"));
        assert_eq!(column.dictionary_value(1), Some("B"));
        assert_eq!(column.dictionary_value(2), None);
        assert_eq!(column.as_i32(), Some(&[0, 1, 0][..]));
    }

    #[test]
    fn known_unique_uses_distinct_count() {
        let table = PinnedTable {
            columns: HashMap::new(),
            column_meta: HashMap::from([(
                "id".to_owned(),
                ColumnMeta {
                    stats: ColumnStats::Integer {
                        min: 1,
                        max: 3,
                        distinct_count: 3,
                    },
                    pilot_kind: PilotColumnKind::Int32,
                },
            )]),
            row_count: 3,
        };

        assert!(table.is_known_unique("id"));
        assert!(table.is_pilot_supported_column("id"));
        assert!(!table.is_known_unique("missing"));
    }

    #[test]
    fn integer_range_i32_reads_integer_stats() {
        let table = PinnedTable {
            columns: HashMap::new(),
            column_meta: HashMap::from([(
                "id".to_owned(),
                ColumnMeta {
                    stats: ColumnStats::Integer {
                        min: -7,
                        max: 42,
                        distinct_count: 50,
                    },
                    pilot_kind: PilotColumnKind::Int32,
                },
            )]),
            row_count: 50,
        };

        assert_eq!(table.integer_range_i32("id"), Some((-7, 42)));
        assert_eq!(table.integer_range_i32("missing"), None);
    }
}
