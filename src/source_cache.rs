use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use crate::processing::MaterializedQueryResult;

#[derive(Default)]
struct ColdSourceSelectionCache {
    entries: HashMap<String, Arc<Vec<u32>>>,
    lru: VecDeque<String>,
    max_entries: usize,
}

#[derive(Clone)]
pub struct CachedGroupedTopKResult {
    pub representative_row_ids: Arc<Vec<u32>>,
    pub aggregate_values: Arc<Vec<f32>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceCacheStats {
    pub source_selection_entries: u64,
    pub source_selection_bytes: u64,
    pub grouped_topk_entries: u64,
    pub grouped_topk_bytes: u64,
    pub query_result_entries: u64,
    pub query_result_bytes: u64,
}

#[derive(Default)]
struct GroupedTopKResultCache {
    entries: HashMap<String, CachedGroupedTopKResult>,
    lru: VecDeque<String>,
    max_entries: usize,
}

#[derive(Default)]
struct QueryResultCache {
    entries: HashMap<String, Arc<MaterializedQueryResult>>,
    lru: VecDeque<String>,
    max_entries: usize,
}

impl ColdSourceSelectionCache {
    fn with_limit(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            max_entries,
        }
    }

    fn get(&mut self, key: &str) -> Option<Arc<Vec<u32>>> {
        let value = self.entries.get(key).cloned()?;
        self.touch(key);
        Some(value)
    }

    fn insert(&mut self, key: String, values: &[u32]) {
        let entry = Arc::new(values.to_vec());
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), entry);
            self.touch(&key);
            return;
        }

        while self.entries.len() >= self.max_entries {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }

        self.entries.insert(key.clone(), entry);
        self.lru.push_back(key);
    }

    fn touch(&mut self, key: &str) {
        if let Some(position) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(position);
        }
        self.lru.push_back(key.to_owned());
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    fn stats(&self) -> (u64, u64) {
        let entries = self.entries.len() as u64;
        let bytes = self
            .entries
            .values()
            .map(|values| (values.len() * std::mem::size_of::<u32>()) as u64)
            .sum();
        (entries, bytes)
    }
}

impl GroupedTopKResultCache {
    fn with_limit(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            max_entries,
        }
    }

    fn get(&mut self, key: &str) -> Option<CachedGroupedTopKResult> {
        let value = self.entries.get(key).cloned()?;
        self.touch(key);
        Some(value)
    }

    fn insert(&mut self, key: String, representative_row_ids: &[u32], aggregate_values: &[f32]) {
        let entry = CachedGroupedTopKResult {
            representative_row_ids: Arc::new(representative_row_ids.to_vec()),
            aggregate_values: Arc::new(aggregate_values.to_vec()),
        };
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), entry);
            self.touch(&key);
            return;
        }

        while self.entries.len() >= self.max_entries {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }

        self.entries.insert(key.clone(), entry);
        self.lru.push_back(key);
    }

    fn touch(&mut self, key: &str) {
        if let Some(position) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(position);
        }
        self.lru.push_back(key.to_owned());
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    fn stats(&self) -> (u64, u64) {
        let entries = self.entries.len() as u64;
        let bytes = self
            .entries
            .values()
            .map(|entry| {
                (entry.representative_row_ids.len() * std::mem::size_of::<u32>()
                    + entry.aggregate_values.len() * std::mem::size_of::<f32>())
                    as u64
            })
            .sum();
        (entries, bytes)
    }
}

impl QueryResultCache {
    fn with_limit(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            max_entries,
        }
    }

    fn get(&mut self, key: &str) -> Option<Arc<MaterializedQueryResult>> {
        let value = self.entries.get(key).cloned()?;
        self.touch(key);
        Some(value)
    }

    fn insert(&mut self, key: String, result: Arc<MaterializedQueryResult>) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), result);
            self.touch(&key);
            return;
        }

        while self.entries.len() >= self.max_entries {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }

        self.entries.insert(key.clone(), result);
        self.lru.push_back(key);
    }

    fn touch(&mut self, key: &str) {
        if let Some(position) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(position);
        }
        self.lru.push_back(key.to_owned());
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    fn stats(&self) -> (u64, u64) {
        let entries = self.entries.len() as u64;
        let bytes = self
            .entries
            .values()
            .map(|result| result.approx_byte_size())
            .sum();
        (entries, bytes)
    }
}

fn cold_cache() -> &'static Mutex<ColdSourceSelectionCache> {
    static CACHE: OnceLock<Mutex<ColdSourceSelectionCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(ColdSourceSelectionCache::with_limit(128)))
}

fn grouped_topk_cache() -> &'static Mutex<GroupedTopKResultCache> {
    static CACHE: OnceLock<Mutex<GroupedTopKResultCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(GroupedTopKResultCache::with_limit(64)))
}

fn query_result_cache() -> &'static Mutex<QueryResultCache> {
    static CACHE: OnceLock<Mutex<QueryResultCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(QueryResultCache::with_limit(32)))
}

pub fn get_local_indices(key: &str) -> Option<Arc<Vec<u32>>> {
    cold_cache()
        .lock()
        .expect("cold source selection cache lock poisoned")
        .get(key)
}

pub fn insert_local_indices(key: String, values: &[u32]) {
    cold_cache()
        .lock()
        .expect("cold source selection cache lock poisoned")
        .insert(key, values);
}

pub fn get_grouped_topk_result(key: &str) -> Option<CachedGroupedTopKResult> {
    grouped_topk_cache()
        .lock()
        .expect("grouped top-k result cache lock poisoned")
        .get(key)
}

pub fn insert_grouped_topk_result(
    key: String,
    representative_row_ids: &[u32],
    aggregate_values: &[f32],
) {
    grouped_topk_cache()
        .lock()
        .expect("grouped top-k result cache lock poisoned")
        .insert(key, representative_row_ids, aggregate_values);
}

pub fn get_materialized_query_result(key: &str) -> Option<Arc<MaterializedQueryResult>> {
    query_result_cache()
        .lock()
        .expect("query result cache lock poisoned")
        .get(key)
}

pub fn insert_materialized_query_result(key: String, result: Arc<MaterializedQueryResult>) {
    query_result_cache()
        .lock()
        .expect("query result cache lock poisoned")
        .insert(key, result);
}

pub fn stats() -> SourceCacheStats {
    let (source_selection_entries, source_selection_bytes) = cold_cache()
        .lock()
        .expect("cold source selection cache lock poisoned")
        .stats();
    let (grouped_topk_entries, grouped_topk_bytes) = grouped_topk_cache()
        .lock()
        .expect("grouped top-k result cache lock poisoned")
        .stats();
    let (query_result_entries, query_result_bytes) = query_result_cache()
        .lock()
        .expect("query result cache lock poisoned")
        .stats();
    SourceCacheStats {
        source_selection_entries,
        source_selection_bytes,
        grouped_topk_entries,
        grouped_topk_bytes,
        query_result_entries,
        query_result_bytes,
    }
}

pub fn reset() {
    cold_cache()
        .lock()
        .expect("cold source selection cache lock poisoned")
        .reset();
    grouped_topk_cache()
        .lock()
        .expect("grouped top-k result cache lock poisoned")
        .reset();
    query_result_cache()
        .lock()
        .expect("query result cache lock poisoned")
        .reset();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use duckdb::arrow::array::Int32Array;
    use duckdb::arrow::datatypes::{DataType, Field, Schema};
    use duckdb::arrow::record_batch::RecordBatch;

    use super::{
        get_grouped_topk_result, get_materialized_query_result, insert_grouped_topk_result,
        insert_local_indices, insert_materialized_query_result, reset,
    };
    use crate::processing::MaterializedQueryResult;

    #[test]
    fn grouped_topk_result_cache_round_trips_and_resets() {
        reset();
        insert_grouped_topk_result("q3".to_owned(), &[7, 11], &[42.5, 19.25]);
        let cached = get_grouped_topk_result("q3").expect("cached grouped top-k result");
        assert_eq!(cached.representative_row_ids.as_slice(), &[7, 11]);
        assert_eq!(cached.aggregate_values.as_slice(), &[42.5, 19.25]);

        reset();
        assert!(get_grouped_topk_result("q3").is_none());
    }

    #[test]
    fn local_indices_cache_still_round_trips() {
        reset();
        insert_local_indices("customer-filter".to_owned(), &[1, 4, 9]);
        let cached = super::get_local_indices("customer-filter").expect("cached local indices");
        assert_eq!(cached.as_slice(), &[1, 4, 9]);
        reset();
    }

    #[test]
    fn materialized_query_result_cache_round_trips_and_resets() {
        reset();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .expect("record batch");
        let result = Arc::new(MaterializedQueryResult::from_batches(schema, vec![batch]));
        insert_materialized_query_result("query".to_owned(), result);

        let cached = get_materialized_query_result("query").expect("cached query result");
        assert_eq!(cached.row_count(), 3);

        reset();
        assert!(get_materialized_query_result("query").is_none());
    }
}
