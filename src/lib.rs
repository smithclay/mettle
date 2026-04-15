#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

mod config;
mod error;
mod host_plan;
mod metal;
mod pinned;
mod plan;
mod processing;
mod source_cache;
mod state;
mod table_functions;

use std::error::Error;

#[cfg(feature = "loadable-extension")]
use duckdb::duckdb_entrypoint_c_api;
use duckdb::Connection;

pub use config::BufferConfig;
pub use error::{MettleError, Result};
pub use metal::MetalDeviceInfo;
pub use state::last_execution_mode;

use processing::register_gpu_processing;
use table_functions::{
    GpuBufferInitTable, GpuCacheResetTable, GpuCacheStatusTable, GpuRegisterTableTable,
};

pub fn register_extension(connection: &Connection) -> std::result::Result<(), Box<dyn Error>> {
    state::set_catalog_connection(connection.try_clone()?);
    connection.register_table_function::<GpuBufferInitTable>("gpu_buffer_init")?;
    connection.register_table_function::<GpuRegisterTableTable>("gpu_register_table")?;
    connection.register_table_function::<GpuCacheStatusTable>("gpu_cache_status")?;
    connection.register_table_function::<GpuCacheResetTable>("gpu_cache_reset")?;
    register_gpu_processing(connection)?;
    Ok(())
}

#[doc(hidden)]
pub fn reset_runtime_state_for_tests() {
    state::reset_for_tests();
    pinned::reset_for_tests();
}

#[doc(hidden)]
pub(crate) fn reset_query_caches() {
    host_plan::reset_cache();
    source_cache::reset();
    state::clear_last_execution_mode();
    metal::reset_runtime_cache();
}

#[doc(hidden)]
pub fn reset_hot_caches_for_benchmark() {
    reset_query_caches();
}

#[doc(hidden)]
pub(crate) fn query_cache_status() -> (metal::MetalCacheStats, source_cache::SourceCacheStats) {
    (metal::cache_stats(), source_cache::stats())
}

#[cfg(feature = "loadable-extension")]
#[duckdb_entrypoint_c_api(ext_name = "mettle", min_duckdb_version = "v1.4.0")]
pub fn extension_entrypoint(connection: Connection) -> std::result::Result<(), Box<dyn Error>> {
    // `register_extension` already registers a cloned catalog connection via
    // `state::set_catalog_connection`. Do not re-register here.
    register_extension(&connection)?;
    Ok(())
}
