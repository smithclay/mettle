use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use duckdb::arrow::array::{Array, StringArray};
use serde_json::Value;

use crate::state;

#[derive(Debug, Clone, Default)]
pub struct HostPlanHints {
    pub preferred_sources: Vec<String>,
    pub estimated_rows: HashMap<String, usize>,
}

pub fn explain_hints(query: &str) -> Option<HostPlanHints> {
    if let Some(cached) = hint_cache()
        .read()
        .expect("host plan cache lock poisoned")
        .get(query)
        .cloned()
    {
        return cached;
    }

    let hints = match state::with_catalog_connection(|connection| {
        let explain_sql = format!("EXPLAIN (FORMAT JSON) {query}");
        let mut statement = connection.prepare(&explain_sql)?;
        let batches = {
            let arrow = statement.query_arrow([])?;
            arrow.collect::<Vec<_>>()
        };
        Ok(extract_json_text(&batches).and_then(|json| parse_hints_from_json(&json)))
    }) {
        Ok(h) => h,
        Err(err) => {
            if debug_logging_enabled() {
                eprintln!("mettle: failed to derive host plan hints: {err}");
            }
            None
        }
    };

    hint_cache()
        .write()
        .expect("host plan cache lock poisoned")
        .insert(query.to_owned(), hints.clone());
    hints
}

fn debug_logging_enabled() -> bool {
    std::env::var_os("METTLE_DEBUG").is_some() || std::env::var_os("METTLE_PROFILE").is_some()
}

pub fn reset_cache() {
    hint_cache()
        .write()
        .expect("host plan cache lock poisoned")
        .clear();
}

fn hint_cache() -> &'static RwLock<HashMap<String, Option<HostPlanHints>>> {
    static CACHE: OnceLock<RwLock<HashMap<String, Option<HostPlanHints>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn extract_json_text(batches: &[duckdb::arrow::record_batch::RecordBatch]) -> Option<String> {
    let mut text = String::new();
    for batch in batches {
        for column in batch.columns() {
            let arr = column.as_any().downcast_ref::<StringArray>()?;
            for row_idx in 0..arr.len() {
                if arr.is_valid(row_idx) {
                    text.push_str(arr.value(row_idx));
                    text.push('\n');
                }
            }
        }
    }
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    Some(text[start..=end].to_owned())
}

fn parse_hints_from_json(json: &str) -> Option<HostPlanHints> {
    let value: Value = serde_json::from_str(json).ok()?;
    let root = value.as_array()?.first()?;
    let mut hints = HostPlanHints::default();
    collect_source_order(root, &mut hints.preferred_sources);
    dedup_source_order(&mut hints.preferred_sources);
    collect_estimated_rows(root, &mut hints.estimated_rows);
    if hints.preferred_sources.is_empty() && hints.estimated_rows.is_empty() {
        None
    } else {
        Some(hints)
    }
}

fn collect_source_order(node: &Value, output: &mut Vec<String>) {
    if let Some(table_name) = scan_table_name(node) {
        output.push(table_name);
    }
    for child in children(node) {
        collect_source_order(child, output);
    }
}

fn collect_estimated_rows(node: &Value, output: &mut HashMap<String, usize>) {
    if let (Some(table_name), Some(estimated_rows)) =
        (scan_table_name(node), estimated_cardinality(node))
    {
        output.insert(table_name, estimated_rows);
    }
    for child in children(node) {
        collect_estimated_rows(child, output);
    }
}

fn dedup_source_order(preferred_sources: &mut Vec<String>) {
    let mut deduped = Vec::with_capacity(preferred_sources.len());
    for source in preferred_sources.drain(..) {
        if !deduped
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&source))
        {
            deduped.push(source);
        }
    }
    *preferred_sources = deduped;
}

fn scan_table_name(node: &Value) -> Option<String> {
    let name = node.get("name")?.as_str()?;
    if !name.contains("SCAN") {
        return None;
    }
    let table = node.get("extra_info")?.get("Table")?.as_str()?;
    Some(normalize_table_name(table))
}

fn normalize_table_name(name: &str) -> String {
    name.split('.').next_back().unwrap_or(name).to_owned()
}

fn estimated_cardinality(node: &Value) -> Option<usize> {
    let value = node.get("extra_info")?.get("Estimated Cardinality")?;
    match value {
        Value::Number(number) => number.as_u64().and_then(|v| usize::try_from(v).ok()),
        Value::String(text) => text.parse::<usize>().ok(),
        _ => None,
    }
}

fn children(node: &Value) -> &[Value] {
    node.get("children")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::{parse_hints_from_json, HostPlanHints};

    fn preferred(hints: &HostPlanHints) -> Vec<String> {
        hints.preferred_sources.clone()
    }

    #[test]
    fn parses_preferred_sources_from_physical_plan_json() {
        let json = r#"
        [
          {
            "name": "HASH_JOIN",
            "children": [
              {
                "name": "SEQ_SCAN",
                "children": [],
                "extra_info": {
                  "Table": "memory.main.orders",
                  "Estimated Cardinality": "10"
                }
              },
              {
                "name": "HASH_JOIN",
                "children": [
                  {
                    "name": "SEQ_SCAN",
                    "children": [],
                    "extra_info": {
                      "Table": "memory.main.customer",
                      "Estimated Cardinality": "3"
                    }
                  },
                  {
                    "name": "SEQ_SCAN",
                    "children": [],
                    "extra_info": {
                      "Table": "memory.main.lineitem",
                      "Estimated Cardinality": "12"
                    }
                  }
                ],
                "extra_info": {}
              }
            ],
            "extra_info": {}
          }
        ]
        "#;
        let hints = parse_hints_from_json(json).unwrap();
        assert_eq!(
            preferred(&hints),
            vec![
                "orders".to_owned(),
                "customer".to_owned(),
                "lineitem".to_owned()
            ]
        );
        assert_eq!(hints.estimated_rows.get("customer").copied(), Some(3));
        assert_eq!(hints.estimated_rows.get("lineitem").copied(), Some(12));
    }
}

