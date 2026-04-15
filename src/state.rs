use std::error::Error;
use std::sync::{Mutex, OnceLock, RwLock};

use duckdb::Connection;

use crate::config::BufferConfig;
use crate::error::{MettleError, Result};
use crate::host_plan;
use crate::metal::{self, MetalDeviceInfo};
use crate::source_cache;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitializedBufferManager {
    pub config: BufferConfig,
    pub device: MetalDeviceInfo,
    pub backend_mode: String,
}

static BUFFER_MANAGER: OnceLock<RwLock<Option<InitializedBufferManager>>> = OnceLock::new();
static LAST_EXECUTION_MODE: OnceLock<RwLock<Option<String>>> = OnceLock::new();
static CATALOG_CONNECTION: OnceLock<Mutex<Option<Connection>>> = OnceLock::new();

pub fn initialize(config: BufferConfig) -> Result<InitializedBufferManager> {
    metal::reset_runtime_cache();
    source_cache::reset();
    let device = metal::validate_runtime()?;
    let state = InitializedBufferManager {
        config,
        device,
        backend_mode: "metal_shared_buffer_column_cache".to_owned(),
    };

    let manager = BUFFER_MANAGER.get_or_init(|| RwLock::new(None));
    *manager.write().expect("buffer manager lock poisoned") = Some(state.clone());
    *LAST_EXECUTION_MODE
        .get_or_init(|| RwLock::new(None))
        .write()
        .expect("execution mode lock poisoned") = None;
    Ok(state)
}

pub fn stage_query(query: &str) -> Result<()> {
    let manager = BUFFER_MANAGER
        .get_or_init(|| RwLock::new(None))
        .read()
        .expect("buffer manager lock poisoned")
        .clone()
        .ok_or_else(|| {
            MettleError::NotInitialized("call gpu_buffer_init before gpu_processing".to_owned())
        })?;

    metal::stage_query(query, &manager.config)?;
    Ok(())
}

pub fn set_catalog_connection(connection: Connection) {
    host_plan::reset_cache();
    *CATALOG_CONNECTION
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("catalog connection lock poisoned") = Some(connection);
}

pub fn with_catalog_connection<T>(
    f: impl FnOnce(&Connection) -> std::result::Result<T, Box<dyn Error>>,
) -> std::result::Result<T, Box<dyn Error>> {
    let guard = CATALOG_CONNECTION
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("catalog connection lock poisoned");
    let connection = guard.as_ref().ok_or_else(|| {
        MettleError::Processing(
            "gpu_processing is missing the registered DuckDB catalog connection".to_owned(),
        )
    })?;
    f(connection)
}

pub fn current_config() -> Result<BufferConfig> {
    BUFFER_MANAGER
        .get_or_init(|| RwLock::new(None))
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_ref()
        .map(|manager| manager.config.clone())
        .ok_or_else(|| {
            MettleError::NotInitialized("call gpu_buffer_init before gpu_processing".to_owned())
        })
}

pub fn set_last_execution_mode(mode: &str) {
    *LAST_EXECUTION_MODE
        .get_or_init(|| RwLock::new(None))
        .write()
        .expect("execution mode lock poisoned") = Some(mode.to_owned());
}

pub fn clear_last_execution_mode() {
    *LAST_EXECUTION_MODE
        .get_or_init(|| RwLock::new(None))
        .write()
        .expect("execution mode lock poisoned") = None;
}

pub fn last_execution_mode() -> Option<String> {
    LAST_EXECUTION_MODE
        .get_or_init(|| RwLock::new(None))
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

pub fn reset_for_tests() {
    host_plan::reset_cache();
    source_cache::reset();
    *BUFFER_MANAGER
        .get_or_init(|| RwLock::new(None))
        .write()
        .expect("buffer manager lock poisoned") = None;
    *LAST_EXECUTION_MODE
        .get_or_init(|| RwLock::new(None))
        .write()
        .expect("execution mode lock poisoned") = None;
    *CATALOG_CONNECTION
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("catalog connection lock poisoned") = None;
    metal::reset_runtime_cache();
}
