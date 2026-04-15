use std::error::Error;
use std::sync::Mutex;

use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::vtab::arrow::{record_batch_to_duckdb_data_chunk, to_duckdb_logical_type};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::Connection;

use crate::plan;
use crate::state;

pub fn register_gpu_processing(connection: &Connection) -> Result<(), Box<dyn Error>> {
    connection.register_table_function::<GpuProcessingTable>("gpu_processing")?;
    Ok(())
}

pub struct GpuProcessingTable;

#[derive(Debug, Clone)]
pub struct MaterializedQueryResult {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    row_count: usize,
}

#[derive(Debug)]
pub struct GpuProcessingInitData {
    next_batch_index: Mutex<usize>,
}

impl MaterializedQueryResult {
    pub(crate) fn from_batches(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        let row_count = batches.iter().map(RecordBatch::num_rows).sum();
        Self {
            schema,
            batches,
            row_count,
        }
    }

    pub(crate) fn approx_byte_size(&self) -> u64 {
        self.batches
            .iter()
            .map(|batch| batch.get_array_memory_size() as u64)
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn row_count(&self) -> usize {
        self.row_count
    }
}

impl VTab for GpuProcessingTable {
    type InitData = GpuProcessingInitData;
    type BindData = MaterializedQueryResult;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let query = bind.get_parameter(0).to_string();
        let materialized = materialize_query(query.as_str())?;

        for field in materialized.schema.fields() {
            bind.add_result_column(field.name(), to_duckdb_logical_type(field.data_type())?);
        }
        bind.set_cardinality(materialized.row_count as u64, true);
        Ok(materialized)
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        init.set_max_threads(1);
        Ok(GpuProcessingInitData {
            next_batch_index: Mutex::new(0),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut duckdb::core::DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let bind_data = func.get_bind_data();
        let init_data = func.get_init_data();
        let mut next_batch_index = init_data
            .next_batch_index
            .lock()
            .expect("gpu_processing batch index lock poisoned");

        if *next_batch_index >= bind_data.batches.len() {
            output.set_len(0);
            return Ok(());
        }

        record_batch_to_duckdb_data_chunk(&bind_data.batches[*next_batch_index], output)?;
        *next_batch_index += 1;
        Ok(())
    }

    fn parameters() -> Option<Vec<duckdb::core::LogicalTypeHandle>> {
        Some(vec![duckdb::core::LogicalTypeId::Varchar.into()])
    }
}

fn materialize_query(query: &str) -> Result<MaterializedQueryResult, Box<dyn Error>> {
    if query.trim().is_empty() {
        return Err("gpu_processing requires a non-empty SQL string".into());
    }

    state::clear_last_execution_mode();
    let query_cache_enabled = std::env::var_os("METTLE_DISABLE_QUERY_CACHE").is_none();
    if let Some(gpu_plan) = plan::try_build_gpu_plan(query) {
        if query_cache_enabled {
            if let Some(cache_key) = plan::result_cache_key(&gpu_plan) {
                if let Some(cached) = crate::source_cache::get_materialized_query_result(&cache_key)
                {
                    state::set_last_execution_mode("metal_hot_query_cache");
                    return Ok(cached.as_ref().clone());
                }
            }
        }

        state::stage_query(query)?;
        if let Some(result) = plan::try_execute(&gpu_plan, Some(query))? {
            if state::last_execution_mode().is_none() {
                state::set_last_execution_mode("metal_generic_plan");
            }
            if query_cache_enabled {
                if let Some(cache_key) = plan::result_cache_key(&gpu_plan) {
                    crate::source_cache::insert_materialized_query_result(
                        cache_key,
                        std::sync::Arc::new(result.clone()),
                    );
                }
            }
            return Ok(result);
        }
    } else {
        state::stage_query(query)?;
    }

    let result = execute_fallback_query(query)?;
    state::set_last_execution_mode("duckdb_cpu_fallback");
    Ok(result)
}

fn execute_fallback_query(query: &str) -> Result<MaterializedQueryResult, Box<dyn Error>> {
    state::with_catalog_connection(|connection| {
        let mut statement = connection.prepare(query)?;
        let batches = {
            let arrow = statement.query_arrow([])?;
            arrow.collect::<Vec<_>>()
        };
        let schema = statement.schema();
        Ok(MaterializedQueryResult::from_batches(schema, batches))
    })
}
