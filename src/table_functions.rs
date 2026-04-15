use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};

use crate::config::{to_i64, BufferConfig};
use crate::metal;
use crate::pinned::{self, PinSummary};
use crate::state::{self, InitializedBufferManager};
use crate::{query_cache_status, reset_query_caches};

pub struct GpuBufferInitTable;

#[derive(Debug, Clone)]
pub struct BufferInitBindData {
    config: BufferConfig,
}

#[derive(Debug)]
pub struct OneShotInitData<T> {
    emitted: AtomicBool,
    row: T,
}

impl VTab for GpuBufferInitTable {
    type InitData = OneShotInitData<InitializedBufferManager>;
    type BindData = BufferInitBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let cache_region_size = bind.get_parameter(0).to_string();
        let processing_region_size = bind.get_parameter(1).to_string();
        let pinned_memory_size = bind
            .get_named_parameter("pinned_memory_size")
            .map(|value| value.to_string());

        bind.add_result_column("status", LogicalTypeId::Varchar.into());
        bind.add_result_column("device_name", LogicalTypeId::Varchar.into());
        bind.add_result_column("cache_region_bytes", LogicalTypeId::Bigint.into());
        bind.add_result_column("processing_region_bytes", LogicalTypeId::Bigint.into());
        bind.add_result_column("pinned_region_bytes", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_backend", LogicalTypeId::Varchar.into());
        bind.set_cardinality(1, true);

        let config = BufferConfig::parse(
            cache_region_size.as_str(),
            processing_region_size.as_str(),
            pinned_memory_size.as_deref(),
        )?;

        Ok(BufferInitBindData { config })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        init.set_max_threads(1);
        let bind_data = unsafe { &*init.get_bind_data::<BufferInitBindData>() };
        let row = state::initialize(bind_data.config.clone())?;
        Ok(OneShotInitData {
            emitted: AtomicBool::new(false),
            row,
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        if init_data.emitted.swap(true, Ordering::SeqCst) {
            output.set_len(0);
            return Ok(());
        }

        write_string(output, 0, "initialized");
        write_string(output, 1, init_data.row.device.name.as_str());
        write_i64(
            output,
            2,
            to_i64(init_data.row.config.cache_region_bytes, "cache region")?,
        );
        write_i64(
            output,
            3,
            to_i64(
                init_data.row.config.processing_region_bytes,
                "processing region",
            )?,
        );
        write_i64(
            output,
            4,
            to_i64(init_data.row.config.pinned_region_bytes, "pinned region")?,
        );
        write_string(output, 5, init_data.row.backend_mode.as_str());
        output.set_len(1);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeId::Varchar.into(),
            LogicalTypeId::Varchar.into(),
        ])
    }

    fn named_parameters() -> Option<Vec<(String, LogicalTypeHandle)>> {
        Some(vec![(
            "pinned_memory_size".to_owned(),
            LogicalTypeId::Varchar.into(),
        )])
    }
}

fn write_string(output: &DataChunkHandle, column_index: usize, value: &str) {
    let column = output.flat_vector(column_index);
    column.insert(0, value);
}

fn write_i64(output: &DataChunkHandle, column_index: usize, value: i64) {
    let mut column = output.flat_vector(column_index);
    let data = column.as_mut_slice::<i64>();
    data[0] = value;
}

pub struct GpuRegisterTableTable;

#[derive(Debug, Clone)]
pub struct RegisterTableBindData {
    table_name: String,
}

impl VTab for GpuRegisterTableTable {
    type InitData = OneShotInitData<PinSummary>;
    type BindData = RegisterTableBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let table_name = bind.get_parameter(0).to_string();

        bind.add_result_column("table_name", LogicalTypeId::Varchar.into());
        bind.add_result_column("row_count", LogicalTypeId::Bigint.into());
        bind.add_result_column("byte_count", LogicalTypeId::Bigint.into());
        bind.add_result_column("skipped_columns", LogicalTypeId::Varchar.into());
        bind.set_cardinality(1, true);

        Ok(RegisterTableBindData { table_name })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        init.set_max_threads(1);
        let bind_data = unsafe { &*init.get_bind_data::<RegisterTableBindData>() };
        let summary = state::with_catalog_connection(|connection| {
            pinned::register_table(connection, bind_data.table_name.as_str())
        })?;
        reset_query_caches();
        Ok(OneShotInitData {
            emitted: AtomicBool::new(false),
            row: summary,
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        if init_data.emitted.swap(true, Ordering::SeqCst) {
            output.set_len(0);
            return Ok(());
        }

        write_string(output, 0, init_data.row.table_name.as_str());
        write_i64(
            output,
            1,
            to_i64(init_data.row.row_count as u64, "pinned row count")?,
        );
        write_i64(
            output,
            2,
            to_i64(init_data.row.byte_count as u64, "pinned byte count")?,
        );
        write_string(output, 3, init_data.row.skipped_columns.join(",").as_str());
        output.set_len(1);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeId::Varchar.into()])
    }
}

pub struct GpuCacheStatusTable;

#[derive(Debug, Clone)]
pub struct CacheStatusRow {
    last_execution_mode: Option<String>,
    metal_stats: metal::MetalCacheStats,
    source_stats: crate::source_cache::SourceCacheStats,
}

impl VTab for GpuCacheStatusTable {
    type InitData = OneShotInitData<CacheStatusRow>;
    type BindData = ();

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("last_execution_mode", LogicalTypeId::Varchar.into());
        bind.add_result_column("metal_entries", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_cached_bytes", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_hits", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_misses", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_evictions", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_bypasses", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_input_entries", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_input_cached_bytes", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_selection_entries", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_selection_cached_bytes", LogicalTypeId::Bigint.into());
        bind.add_result_column("metal_join_result_entries", LogicalTypeId::Bigint.into());
        bind.add_result_column(
            "metal_join_result_cached_bytes",
            LogicalTypeId::Bigint.into(),
        );
        bind.add_result_column("source_selection_entries", LogicalTypeId::Bigint.into());
        bind.add_result_column("source_selection_bytes", LogicalTypeId::Bigint.into());
        bind.add_result_column("grouped_topk_entries", LogicalTypeId::Bigint.into());
        bind.add_result_column("grouped_topk_bytes", LogicalTypeId::Bigint.into());
        bind.add_result_column("query_result_entries", LogicalTypeId::Bigint.into());
        bind.add_result_column("query_result_bytes", LogicalTypeId::Bigint.into());
        bind.set_cardinality(1, true);
        Ok(())
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        init.set_max_threads(1);
        let (metal_stats, source_stats) = query_cache_status();
        let row = CacheStatusRow {
            last_execution_mode: state::last_execution_mode(),
            metal_stats,
            source_stats,
        };
        Ok(OneShotInitData {
            emitted: AtomicBool::new(false),
            row,
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        if init_data.emitted.swap(true, Ordering::SeqCst) {
            output.set_len(0);
            return Ok(());
        }

        write_string(
            output,
            0,
            init_data.row.last_execution_mode.as_deref().unwrap_or(""),
        );
        write_i64(
            output,
            1,
            to_i64(init_data.row.metal_stats.entries, "metal entries")?,
        );
        write_i64(
            output,
            2,
            to_i64(init_data.row.metal_stats.cached_bytes, "metal cached bytes")?,
        );
        write_i64(
            output,
            3,
            to_i64(init_data.row.metal_stats.hits, "metal hits")?,
        );
        write_i64(
            output,
            4,
            to_i64(init_data.row.metal_stats.misses, "metal misses")?,
        );
        write_i64(
            output,
            5,
            to_i64(init_data.row.metal_stats.evictions, "metal evictions")?,
        );
        write_i64(
            output,
            6,
            to_i64(init_data.row.metal_stats.bypasses, "metal bypasses")?,
        );
        write_i64(
            output,
            7,
            to_i64(
                init_data.row.metal_stats.input_entries,
                "metal input entries",
            )?,
        );
        write_i64(
            output,
            8,
            to_i64(
                init_data.row.metal_stats.input_cached_bytes,
                "metal input cached bytes",
            )?,
        );
        write_i64(
            output,
            9,
            to_i64(
                init_data.row.metal_stats.selection_entries,
                "metal selection entries",
            )?,
        );
        write_i64(
            output,
            10,
            to_i64(
                init_data.row.metal_stats.selection_cached_bytes,
                "metal selection cached bytes",
            )?,
        );
        write_i64(
            output,
            11,
            to_i64(
                init_data.row.metal_stats.join_result_entries,
                "metal join result entries",
            )?,
        );
        write_i64(
            output,
            12,
            to_i64(
                init_data.row.metal_stats.join_result_cached_bytes,
                "metal join result cached bytes",
            )?,
        );
        write_i64(
            output,
            13,
            to_i64(
                init_data.row.source_stats.source_selection_entries,
                "source selection entries",
            )?,
        );
        write_i64(
            output,
            14,
            to_i64(
                init_data.row.source_stats.source_selection_bytes,
                "source selection bytes",
            )?,
        );
        write_i64(
            output,
            15,
            to_i64(
                init_data.row.source_stats.grouped_topk_entries,
                "grouped topk entries",
            )?,
        );
        write_i64(
            output,
            16,
            to_i64(
                init_data.row.source_stats.grouped_topk_bytes,
                "grouped topk bytes",
            )?,
        );
        write_i64(
            output,
            17,
            to_i64(
                init_data.row.source_stats.query_result_entries,
                "query result entries",
            )?,
        );
        write_i64(
            output,
            18,
            to_i64(
                init_data.row.source_stats.query_result_bytes,
                "query result bytes",
            )?,
        );
        output.set_len(1);
        Ok(())
    }
}

pub struct GpuCacheResetTable;

#[derive(Debug, Clone)]
pub struct CacheResetRow {
    status: String,
}

impl VTab for GpuCacheResetTable {
    type InitData = OneShotInitData<CacheResetRow>;
    type BindData = ();

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("status", LogicalTypeId::Varchar.into());
        bind.set_cardinality(1, true);
        Ok(())
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        init.set_max_threads(1);
        reset_query_caches();
        Ok(OneShotInitData {
            emitted: AtomicBool::new(false),
            row: CacheResetRow {
                status: "reset".to_owned(),
            },
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        if init_data.emitted.swap(true, Ordering::SeqCst) {
            output.set_len(0);
            return Ok(());
        }
        write_string(output, 0, init_data.row.status.as_str());
        output.set_len(1);
        Ok(())
    }
}
