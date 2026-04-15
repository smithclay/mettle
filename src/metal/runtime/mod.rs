use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::mem::size_of_val;
use std::ptr::NonNull;

use super::gpu_buffer::{allocate_buffer, make_buffer_from_slice, GpuBuffer, GpuVector};
use super::shaders::{compile_library, compile_pipeline, dispatch_threads, hash_values};
use super::{MetalCacheStats, MetalDeviceInfo, QuerySubmission};
use super::{CACHE_BYPASSES, CACHE_EVICTIONS, CACHE_HITS, CACHE_MISSES, RUNTIME_GENERATION};
use crate::config::BufferConfig;
use crate::error::{MettleError, Result};
use crate::pinned::PinnedColumn;

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLSize,
};

use std::sync::atomic::Ordering;

mod aggregate_ops;
mod gather_ops;
mod join_ops;
mod scalar_ops;

#[cfg(target_os = "macos")]
pub(super) const GV_THREADS_PER_GROUP: usize = 256;
#[cfg(target_os = "macos")]
pub(super) const MAX_GROUP_KEY_LANES: usize = 4;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct BufferCacheKey {
    pub(super) type_tag: u8,
    pub(super) len: usize,
    pub(super) fingerprint: u64,
}

#[cfg(target_os = "macos")]
pub(super) struct CachedInputBuffer {
    pub(super) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) byte_len: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct CachedSelectionVector {
    pub(super) vector: GpuVector,
    pub(super) byte_len: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct CachedJoinVectors {
    pub(super) representative_row_ids: GpuVector,
    pub(super) aggregate_row_ids: GpuVector,
    pub(super) byte_len: usize,
}

#[cfg(target_os = "macos")]
pub(super) const CACHE_TAG_I32: u8 = 1;
#[cfg(target_os = "macos")]
pub(super) const CACHE_TAG_I64: u8 = 2;
#[cfg(target_os = "macos")]
pub(super) const CACHE_TAG_F32: u8 = 3;
#[cfg(target_os = "macos")]
pub(super) const CACHE_TAG_DATE32: u8 = 4;

pub(super) struct MetalRuntime {
    pub(super) info: MetalDeviceInfo,
    #[cfg(target_os = "macos")]
    pub(super) device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(target_os = "macos")]
    pub(super) command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    #[cfg(target_os = "macos")]
    pub(super) library: Retained<ProtocolObject<dyn MTLLibrary>>,
    #[cfg(target_os = "macos")]
    pub(super) pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    #[cfg(target_os = "macos")]
    pub(super) input_cache: HashMap<BufferCacheKey, CachedInputBuffer>,
    #[cfg(target_os = "macos")]
    pub(super) input_cache_lru: VecDeque<BufferCacheKey>,
    #[cfg(target_os = "macos")]
    pub(super) generation: u64,
    #[cfg(target_os = "macos")]
    pub(super) cache_region_bytes: u64,
    #[cfg(target_os = "macos")]
    pub(super) cached_input_bytes: u64,
    #[cfg(target_os = "macos")]
    pub(super) selection_cache: HashMap<String, CachedSelectionVector>,
    #[cfg(target_os = "macos")]
    pub(super) selection_cache_lru: VecDeque<String>,
    #[cfg(target_os = "macos")]
    pub(super) cached_selection_bytes: u64,
    #[cfg(target_os = "macos")]
    pub(super) join_result_cache: HashMap<String, CachedJoinVectors>,
    #[cfg(target_os = "macos")]
    pub(super) join_result_cache_lru: VecDeque<String>,
    #[cfg(target_os = "macos")]
    pub(super) cached_join_result_bytes: u64,
    #[cfg(target_os = "macos")]
    pub(super) batch_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    #[cfg(target_os = "macos")]
    pub(super) column_gpu_cache: HashMap<usize, GpuBuffer>,
}

impl MetalRuntime {
    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn new(cache_region_bytes: u64) -> Result<Self> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return Err(MettleError::MetalUnavailable(
                "no system Metal device is available".to_owned(),
            ));
        };

        let Some(command_queue) = device.newCommandQueue() else {
            return Err(MettleError::MetalUnavailable(
                "the system Metal device did not provide a command queue".to_owned(),
            ));
        };

        let library = compile_library(&device)?;

        Ok(Self {
            info: MetalDeviceInfo {
                name: device.name().to_string(),
                registry_id: device.registryID(),
                unified_memory: device.hasUnifiedMemory(),
            },
            device,
            command_queue,
            library,
            pipelines: HashMap::new(),
            input_cache: HashMap::new(),
            input_cache_lru: VecDeque::new(),
            generation: RUNTIME_GENERATION.load(Ordering::Relaxed),
            cache_region_bytes,
            cached_input_bytes: 0,
            selection_cache: HashMap::new(),
            selection_cache_lru: VecDeque::new(),
            cached_selection_bytes: 0,
            join_result_cache: HashMap::new(),
            join_result_cache_lru: VecDeque::new(),
            cached_join_result_bytes: 0,
            batch_command_buffer: None,
            column_gpu_cache: HashMap::new(),
        })
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn cache_stats(&self) -> MetalCacheStats {
        let input_entries = self.input_cache.len() as u64;
        let selection_entries = self.selection_cache.len() as u64;
        let join_result_entries = self.join_result_cache.len() as u64;
        let input_cached_bytes = self.cached_input_bytes;
        let selection_cached_bytes = self.cached_selection_bytes;
        let join_result_cached_bytes = self.cached_join_result_bytes;
        MetalCacheStats {
            entries: input_entries + selection_entries + join_result_entries,
            cached_bytes: input_cached_bytes + selection_cached_bytes + join_result_cached_bytes,
            input_entries,
            input_cached_bytes,
            selection_entries,
            selection_cached_bytes,
            join_result_entries,
            join_result_cached_bytes,
            ..MetalCacheStats::default()
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn new(_cache_region_bytes: u64) -> Result<Self> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn begin_batch(&mut self) -> Result<()> {
        self.flush_batch()?;
        let cb = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create batch command buffer".to_owned())
        })?;
        self.batch_command_buffer = Some(cb);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn end_batch(&mut self) -> Result<()> {
        self.flush_batch()
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn flush_batch(&mut self) -> Result<()> {
        if let Some(cb) = self.batch_command_buffer.take() {
            cb.commit();
            cb.waitUntilCompleted();
            if let Some(error) = cb.error() {
                return Err(MettleError::Processing(format!(
                    "batch command buffer failed: {error}"
                )));
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn get_command_buffer(&self) -> Result<(Retained<ProtocolObject<dyn MTLCommandBuffer>>, bool)> {
        if let Some(ref cb) = self.batch_command_buffer {
            Ok((cb.clone(), true))
        } else {
            let cb = self.command_queue.commandBuffer().ok_or_else(|| {
                MettleError::MetalUnavailable("failed to create a Metal command buffer".to_owned())
            })?;
            Ok((cb, false))
        }
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn commit_if_standalone(
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        is_batch: bool,
        ctx: &str,
    ) -> Result<()> {
        if !is_batch {
            cb.commit();
            cb.waitUntilCompleted();
            if let Some(e) = cb.error() {
                return Err(MettleError::Processing(format!("{ctx} failed: {e}")));
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn empty_dispatch_args(&self) -> Result<GpuBuffer> {
        let buf = make_buffer_from_slice(&self.device, &[0u32, 1, 1], "empty dispatch args")?;
        Ok(GpuBuffer {
            buffer: buf,
            len: 3,
            element_size: 4,
        })
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn allocate_dispatch_args_buffer(&self, label: &str) -> Result<GpuBuffer> {
        let buffer = allocate_buffer(&self.device, 3 * std::mem::size_of::<u32>(), label)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                [0u32, 1, 1].as_ptr(),
                buffer.contents().as_ptr().cast::<u32>(),
                3,
            );
        }
        Ok(GpuBuffer {
            buffer,
            len: 3,
            element_size: 4,
        })
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn encode_dispatch_args_update_on(
        &mut self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        count: &GpuBuffer,
        dispatch_args: &GpuBuffer,
    ) -> Result<()> {
        let pipeline = self.pipeline("write_dispatch_args")?.clone();
        let threads_per_group = GV_THREADS_PER_GROUP as u32;
        let enc = cb.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&count.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&dispatch_args.buffer), 0, 1);
            enc.setBytes_length_atIndex(NonNull::from(&threads_per_group).cast::<c_void>(), 4, 2);
        }
        dispatch_threads(&enc, &pipeline, 1);
        enc.endEncoding();
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn dispatch_gv_indirect(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        dispatch_args: &GpuBuffer,
    ) {
        unsafe {
            encoder
                .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &dispatch_args.buffer,
                    0,
                    MTLSize {
                        width: GV_THREADS_PER_GROUP,
                        height: 1,
                        depth: 1,
                    },
                );
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn begin_batch(&mut self) -> Result<()> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn end_batch(&mut self) -> Result<()> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn stage_query(
        &mut self,
        query: &str,
        config: &BufferConfig,
    ) -> Result<QuerySubmission> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let query_bytes = u64::try_from(query.len())
            .map_err(|_| MettleError::Processing("query length overflowed u64".to_owned()))?;

        if query_bytes == 0 {
            return Err(MettleError::Processing(
                "gpu_processing requires a non-empty SQL string".to_owned(),
            ));
        }

        if query_bytes > config.processing_region_bytes {
            return Err(MettleError::Processing(format!(
                "query payload ({query_bytes} bytes) exceeded the configured GPU processing region ({})",
                config.processing_region_bytes
            )));
        }

        if query_bytes > config.pinned_region_bytes {
            return Err(MettleError::Processing(format!(
                "query payload ({query_bytes} bytes) exceeded the configured pinned memory region ({})",
                config.pinned_region_bytes
            )));
        }

        let buffer = allocate_buffer(&self.device, query.len(), "query staging buffer")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                query.as_ptr(),
                buffer.contents().as_ptr().cast::<u8>(),
                query.len(),
            );
        }

        let mirrored = unsafe {
            let ptr = buffer.contents().as_ptr().cast::<u8>();
            std::slice::from_raw_parts(ptr, query.len())
        };
        if mirrored != query.as_bytes() {
            return Err(MettleError::Processing(
                "Metal shared-buffer roundtrip validation failed".to_owned(),
            ));
        }

        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        let fingerprint_hex = format!("{:016x}", hasher.finish());

        Ok(QuerySubmission {
            device: self.info.clone(),
            query_bytes,
            fingerprint_hex,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn stage_query(
        &mut self,
        _query: &str,
        _config: &BufferConfig,
    ) -> Result<QuerySubmission> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    /// Upload a PinnedColumn to a GpuBuffer (i32), keyed by data pointer.
    /// Int64 columns are truncated to i32 (sufficient for join keys).
    /// Cached across queries — no per-query hashing.
    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn column_to_gpu_buffer(&mut self, col: &PinnedColumn) -> Result<GpuBuffer> {
        let ptr_key = match col {
            PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => v.as_ptr() as usize,
            PinnedColumn::Int64(v) => v.as_ptr() as usize,
            PinnedColumn::Float32(v) => v.as_ptr() as usize,
            PinnedColumn::Float64(v) => v.as_ptr() as usize,
            PinnedColumn::DictionaryI32 { codes, .. } => codes.as_ptr() as usize,
        };

        if let Some(cached) = self.column_gpu_cache.get(&ptr_key) {
            return Ok(GpuBuffer {
                buffer: cached.buffer.clone(),
                len: cached.len,
                element_size: cached.element_size,
            });
        }

        let buf = match col {
            PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
                let metal_buf = make_buffer_from_slice(&self.device, v, "col i32")?;
                GpuBuffer {
                    buffer: metal_buf,
                    len: v.len(),
                    element_size: 4,
                }
            }
            PinnedColumn::Int64(v) => {
                let converted: Vec<i32> = v.iter().map(|&x| x as i32).collect();
                let metal_buf = make_buffer_from_slice(&self.device, &converted, "col i64>i32")?;
                GpuBuffer {
                    buffer: metal_buf,
                    len: converted.len(),
                    element_size: 4,
                }
            }
            PinnedColumn::Float32(v) => {
                let metal_buf = make_buffer_from_slice(&self.device, v, "col f32")?;
                GpuBuffer {
                    buffer: metal_buf,
                    len: v.len(),
                    element_size: 4,
                }
            }
            PinnedColumn::Float64(v) => {
                let converted: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                let metal_buf = make_buffer_from_slice(&self.device, &converted, "col f64>f32")?;
                GpuBuffer {
                    buffer: metal_buf,
                    len: converted.len(),
                    element_size: 4,
                }
            }
            PinnedColumn::DictionaryI32 { codes, .. } => {
                let metal_buf = make_buffer_from_slice(&self.device, codes, "col dict>i32")?;
                GpuBuffer {
                    buffer: metal_buf,
                    len: codes.len(),
                    element_size: 4,
                }
            }
        };

        self.column_gpu_cache.insert(
            ptr_key,
            GpuBuffer {
                buffer: buf.buffer.clone(),
                len: buf.len,
                element_size: buf.element_size,
            },
        );
        Ok(buf)
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn column_to_gpu_buffer(&mut self, _: &PinnedColumn) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn pipeline(
        &mut self,
        function_name: &'static str,
    ) -> Result<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        if !self.pipelines.contains_key(function_name) {
            let pipeline = compile_pipeline(&self.device, &self.library, function_name)?;
            self.pipelines.insert(function_name, pipeline);
        }

        self.pipelines.get(function_name).ok_or_else(|| {
            MettleError::Processing(format!(
                "Metal pipeline cache did not retain '{function_name}'"
            ))
        })
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn cached_input_buffer<T: Copy>(
        &mut self,
        values: &[T],
        cache_tag: u8,
        label: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let byte_len = size_of_val(values);
        if byte_len == 0 {
            return allocate_buffer(&self.device, 0, label);
        }

        if self.cache_region_bytes == 0 || (byte_len as u64) > self.cache_region_bytes {
            CACHE_BYPASSES.fetch_add(1, Ordering::Relaxed);
            return make_buffer_from_slice(&self.device, values, label);
        }

        let key = BufferCacheKey {
            type_tag: cache_tag,
            len: values.len(),
            fingerprint: hash_values(values),
        };
        if let Some(buffer) = self.input_cache.get(&key).map(|entry| entry.buffer.clone()) {
            self.touch_lru(key);
            CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(buffer);
        }

        let buffer = make_buffer_from_slice(&self.device, values, label)?;
        CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
        self.insert_cached_input_buffer(
            key,
            CachedInputBuffer {
                buffer: buffer.clone(),
                byte_len,
            },
        );
        Ok(buffer)
    }

    #[cfg(target_os = "macos")]
    fn insert_cached_input_buffer(&mut self, key: BufferCacheKey, entry: CachedInputBuffer) {
        while self.cached_input_bytes + entry.byte_len as u64 > self.cache_region_bytes {
            let Some(oldest_key) = self.input_cache_lru.pop_front() else {
                break;
            };
            if let Some(evicted) = self.input_cache.remove(&oldest_key) {
                self.cached_input_bytes = self
                    .cached_input_bytes
                    .saturating_sub(evicted.byte_len as u64);
                CACHE_EVICTIONS.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.cached_input_bytes += entry.byte_len as u64;
        self.input_cache.insert(key, entry);
        self.touch_lru(key);
    }

    #[cfg(target_os = "macos")]
    fn touch_lru(&mut self, key: BufferCacheKey) {
        if let Some(position) = self
            .input_cache_lru
            .iter()
            .position(|candidate| *candidate == key)
        {
            self.input_cache_lru.remove(position);
        }
        self.input_cache_lru.push_back(key);
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn get_selection_cache_gv(&mut self, key: &str) -> Option<GpuVector> {
        let vector = self
            .selection_cache
            .get(key)
            .map(|entry| entry.vector.clone())?;
        self.touch_selection_lru(key);
        Some(vector)
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn put_selection_cache_gv(&mut self, key: String, vector: &GpuVector) {
        let budget = self.cache_region_bytes / 2;
        if budget == 0 {
            return;
        }

        let byte_len =
            vector.data.byte_len() + vector.count.byte_len() + vector.dispatch_args.byte_len();
        if (byte_len as u64) > budget {
            return;
        }

        if let Some(existing) = self.selection_cache.remove(&key) {
            self.cached_selection_bytes = self
                .cached_selection_bytes
                .saturating_sub(existing.byte_len as u64);
            if let Some(position) = self
                .selection_cache_lru
                .iter()
                .position(|candidate| candidate == &key)
            {
                self.selection_cache_lru.remove(position);
            }
        }

        while self.cached_selection_bytes + byte_len as u64 > budget {
            let Some(oldest_key) = self.selection_cache_lru.pop_front() else {
                break;
            };
            if let Some(evicted) = self.selection_cache.remove(&oldest_key) {
                self.cached_selection_bytes = self
                    .cached_selection_bytes
                    .saturating_sub(evicted.byte_len as u64);
            }
        }

        self.cached_selection_bytes += byte_len as u64;
        self.selection_cache.insert(
            key.clone(),
            CachedSelectionVector {
                vector: vector.clone(),
                byte_len,
            },
        );
        self.touch_selection_lru(&key);
    }

    #[cfg(target_os = "macos")]
    fn touch_selection_lru(&mut self, key: &str) {
        if let Some(position) = self
            .selection_cache_lru
            .iter()
            .position(|candidate| candidate == key)
        {
            self.selection_cache_lru.remove(position);
        }
        self.selection_cache_lru.push_back(key.to_owned());
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn get_join_result_cache_gv(&mut self, key: &str) -> Option<(GpuVector, GpuVector)> {
        let entry = self.join_result_cache.get(key)?;
        let representative = entry.representative_row_ids.clone();
        let aggregate = entry.aggregate_row_ids.clone();
        self.touch_join_result_lru(key);
        Some((representative, aggregate))
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn put_join_result_cache_gv(
        &mut self,
        key: String,
        representative_row_ids: &GpuVector,
        aggregate_row_ids: &GpuVector,
    ) {
        let budget = self.cache_region_bytes / 4;
        if budget == 0 {
            return;
        }

        let byte_len = representative_row_ids.data.byte_len()
            + representative_row_ids.count.byte_len()
            + representative_row_ids.dispatch_args.byte_len()
            + aggregate_row_ids.data.byte_len()
            + aggregate_row_ids.count.byte_len()
            + aggregate_row_ids.dispatch_args.byte_len();
        if (byte_len as u64) > budget {
            return;
        }

        if let Some(existing) = self.join_result_cache.remove(&key) {
            self.cached_join_result_bytes = self
                .cached_join_result_bytes
                .saturating_sub(existing.byte_len as u64);
            if let Some(position) = self
                .join_result_cache_lru
                .iter()
                .position(|candidate| candidate == &key)
            {
                self.join_result_cache_lru.remove(position);
            }
        }

        while self.cached_join_result_bytes + byte_len as u64 > budget {
            let Some(oldest_key) = self.join_result_cache_lru.pop_front() else {
                break;
            };
            if let Some(evicted) = self.join_result_cache.remove(&oldest_key) {
                self.cached_join_result_bytes = self
                    .cached_join_result_bytes
                    .saturating_sub(evicted.byte_len as u64);
            }
        }

        self.cached_join_result_bytes += byte_len as u64;
        self.join_result_cache.insert(
            key.clone(),
            CachedJoinVectors {
                representative_row_ids: representative_row_ids.clone(),
                aggregate_row_ids: aggregate_row_ids.clone(),
                byte_len,
            },
        );
        self.touch_join_result_lru(&key);
    }

    #[cfg(target_os = "macos")]
    fn touch_join_result_lru(&mut self, key: &str) {
        if let Some(position) = self
            .join_result_cache_lru
            .iter()
            .position(|candidate| candidate == key)
        {
            self.join_result_cache_lru.remove(position);
        }
        self.join_result_cache_lru.push_back(key.to_owned());
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn get_selection_cache_gv(&mut self, _: &str) -> Option<GpuVector> {
        None
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn put_selection_cache_gv(&mut self, _: String, _: &GpuVector) {}

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn get_join_result_cache_gv(&mut self, _: &str) -> Option<(GpuVector, GpuVector)> {
        None
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn put_join_result_cache_gv(&mut self, _: String, _: &GpuVector, _: &GpuVector) {}
}
