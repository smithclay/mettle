use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::mem::size_of_val;
use std::ptr::NonNull;

use super::gpu_buffer::{allocate_buffer, make_buffer_from_slice, GpuBuffer, GpuVector};
use super::shaders::{compile_library, compile_pipeline, dispatch_threads, hash_values};
use super::{
    GpuDenseGroupedRows, GpuGroupedAggregateRows, GpuGroupedAggregateSlotResult,
    GpuGroupedAggregateSpec, GpuGroupedAggregateStateKind, GpuGroupedTopKRows, MetalCacheStats,
    MetalDeviceInfo, QuerySubmission, ScalarCompareOp, ScalarProjectOp,
};
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

#[cfg(target_os = "macos")]
const GV_THREADS_PER_GROUP: usize = 256;
#[cfg(target_os = "macos")]
const MAX_GROUP_KEY_LANES: usize = 4;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferCacheKey {
    type_tag: u8,
    len: usize,
    fingerprint: u64,
}

#[cfg(target_os = "macos")]
struct CachedInputBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    byte_len: usize,
}

#[cfg(target_os = "macos")]
struct CachedSelectionVector {
    vector: GpuVector,
    byte_len: usize,
}

#[cfg(target_os = "macos")]
struct CachedJoinVectors {
    representative_row_ids: GpuVector,
    aggregate_row_ids: GpuVector,
    byte_len: usize,
}

#[cfg(target_os = "macos")]
const CACHE_TAG_I32: u8 = 1;
#[cfg(target_os = "macos")]
const CACHE_TAG_I64: u8 = 2;
#[cfg(target_os = "macos")]
const CACHE_TAG_F32: u8 = 3;
#[cfg(target_os = "macos")]
const CACHE_TAG_DATE32: u8 = 4;

pub(super) struct MetalRuntime {
    pub(super) info: MetalDeviceInfo,
    #[cfg(target_os = "macos")]
    pub(super) device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(target_os = "macos")]
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    #[cfg(target_os = "macos")]
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    #[cfg(target_os = "macos")]
    pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    #[cfg(target_os = "macos")]
    input_cache: HashMap<BufferCacheKey, CachedInputBuffer>,
    #[cfg(target_os = "macos")]
    input_cache_lru: VecDeque<BufferCacheKey>,
    #[cfg(target_os = "macos")]
    pub(super) generation: u64,
    #[cfg(target_os = "macos")]
    pub(super) cache_region_bytes: u64,
    #[cfg(target_os = "macos")]
    cached_input_bytes: u64,
    #[cfg(target_os = "macos")]
    selection_cache: HashMap<String, CachedSelectionVector>,
    #[cfg(target_os = "macos")]
    selection_cache_lru: VecDeque<String>,
    #[cfg(target_os = "macos")]
    cached_selection_bytes: u64,
    #[cfg(target_os = "macos")]
    join_result_cache: HashMap<String, CachedJoinVectors>,
    #[cfg(target_os = "macos")]
    join_result_cache_lru: VecDeque<String>,
    #[cfg(target_os = "macos")]
    cached_join_result_bytes: u64,
    #[cfg(target_os = "macos")]
    batch_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    #[cfg(target_os = "macos")]
    column_gpu_cache: HashMap<usize, GpuBuffer>,
}

impl MetalRuntime {
    #[cfg(target_os = "macos")]
    pub(super) fn new(cache_region_bytes: u64) -> Result<Self> {
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
    pub(super) fn cache_stats(&self) -> MetalCacheStats {
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
    pub(super) fn new(_cache_region_bytes: u64) -> Result<Self> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn begin_batch(&mut self) -> Result<()> {
        self.flush_batch()?;
        let cb = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create batch command buffer".to_owned())
        })?;
        self.batch_command_buffer = Some(cb);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(super) fn end_batch(&mut self) -> Result<()> {
        self.flush_batch()
    }

    #[cfg(target_os = "macos")]
    fn flush_batch(&mut self) -> Result<()> {
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
    fn get_command_buffer(&self) -> Result<(Retained<ProtocolObject<dyn MTLCommandBuffer>>, bool)> {
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
    fn commit_if_standalone(
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
    fn empty_dispatch_args(&self) -> Result<GpuBuffer> {
        let buf = make_buffer_from_slice(&self.device, &[0u32, 1, 1], "empty dispatch args")?;
        Ok(GpuBuffer {
            buffer: buf,
            len: 3,
            element_size: 4,
        })
    }

    #[cfg(target_os = "macos")]
    fn allocate_dispatch_args_buffer(&self, label: &str) -> Result<GpuBuffer> {
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
    fn encode_dispatch_args_update_on(
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
    fn dispatch_gv_indirect(
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
    pub(super) fn begin_batch(&mut self) -> Result<()> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn end_batch(&mut self) -> Result<()> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn stage_query(
        &mut self,
        query: &str,
        config: &BufferConfig,
    ) -> Result<QuerySubmission> {
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
    pub(super) fn stage_query(
        &mut self,
        _query: &str,
        _config: &BufferConfig,
    ) -> Result<QuerySubmission> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn project_f32_values(
        &mut self,
        values: &[f32],
        scalar: f32,
        op: ScalarProjectOp,
    ) -> Result<Vec<f32>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let pipeline = self.pipeline("project_f32_scalar")?.clone();
        let input =
            self.cached_input_buffer(values, CACHE_TAG_F32, "gpu float projection input")?;
        let output = allocate_buffer(
            &self.device,
            std::mem::size_of_val(values),
            "gpu float projection output",
        )?;

        let command_buffer = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal command buffer".to_owned())
        })?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;

        encode_f32_scalar_projection(
            &encoder,
            &pipeline,
            &input,
            &output,
            scalar,
            values.len(),
            op,
        )?;

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MettleError::Processing(format!(
                "Metal float projection command buffer failed: {error}"
            )));
        }

        let result = unsafe {
            std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), values.len())
        };
        Ok(result.to_vec())
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compare_i32_values(
        &mut self,
        values: &[i32],
        scalar: i32,
        op: ScalarCompareOp,
    ) -> Result<Vec<bool>> {
        self.compare_i32_values_with_tag(
            values,
            scalar,
            op,
            CACHE_TAG_I32,
            "gpu int comparison input",
        )
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compare_date32_values(
        &mut self,
        values: &[i32],
        scalar: i32,
        op: ScalarCompareOp,
    ) -> Result<Vec<bool>> {
        self.compare_i32_values_with_tag(
            values,
            scalar,
            op,
            CACHE_TAG_DATE32,
            "gpu date comparison input",
        )
    }

    #[cfg(target_os = "macos")]
    fn compare_i32_values_with_tag(
        &mut self,
        values: &[i32],
        scalar: i32,
        op: ScalarCompareOp,
        cache_tag: u8,
        input_label: &str,
    ) -> Result<Vec<bool>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let pipeline = self.pipeline("compare_i32_scalar")?.clone();
        let input = self.cached_input_buffer(values, cache_tag, input_label)?;
        let output = allocate_buffer(&self.device, values.len(), "gpu int comparison output")?;

        let command_buffer = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal command buffer".to_owned())
        })?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;

        encode_i32_scalar_compare(
            &encoder,
            &pipeline,
            &input,
            &output,
            scalar,
            values.len(),
            op,
        )?;

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MettleError::Processing(format!(
                "Metal int comparison command buffer failed: {error}"
            )));
        }

        let result = unsafe {
            std::slice::from_raw_parts(output.contents().as_ptr().cast::<u8>(), values.len())
        };
        Ok(result.iter().map(|value| *value != 0).collect())
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compare_i64_values(
        &mut self,
        values: &[i64],
        scalar: i64,
        op: ScalarCompareOp,
    ) -> Result<Vec<bool>> {
        self.compare_i64_values_with_tag(
            values,
            scalar,
            op,
            CACHE_TAG_I64,
            "gpu bigint comparison input",
        )
    }

    #[cfg(target_os = "macos")]
    fn compare_i64_values_with_tag(
        &mut self,
        values: &[i64],
        scalar: i64,
        op: ScalarCompareOp,
        cache_tag: u8,
        input_label: &str,
    ) -> Result<Vec<bool>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let pipeline = self.pipeline("compare_i64_scalar")?.clone();
        let input = self.cached_input_buffer(values, cache_tag, input_label)?;
        let output = allocate_buffer(&self.device, values.len(), "gpu bigint comparison output")?;

        let command_buffer = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal command buffer".to_owned())
        })?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;

        encode_i64_scalar_compare(
            &encoder,
            &pipeline,
            &input,
            &output,
            scalar,
            values.len(),
            op,
        )?;

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MettleError::Processing(format!(
                "Metal bigint comparison command buffer failed: {error}"
            )));
        }

        let result = unsafe {
            std::slice::from_raw_parts(output.contents().as_ptr().cast::<u8>(), values.len())
        };
        Ok(result.iter().map(|value| *value != 0).collect())
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compare_f32_values(
        &mut self,
        values: &[f32],
        scalar: f32,
        op: ScalarCompareOp,
    ) -> Result<Vec<bool>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let pipeline = self.pipeline("compare_f32_scalar")?.clone();
        let input =
            self.cached_input_buffer(values, CACHE_TAG_F32, "gpu float comparison input")?;
        let output = allocate_buffer(&self.device, values.len(), "gpu float comparison output")?;

        let command_buffer = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal command buffer".to_owned())
        })?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;

        encode_f32_scalar_compare(
            &encoder,
            &pipeline,
            &input,
            &output,
            scalar,
            values.len(),
            op,
        )?;

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MettleError::Processing(format!(
                "Metal float comparison command buffer failed: {error}"
            )));
        }

        let result = unsafe {
            std::slice::from_raw_parts(output.contents().as_ptr().cast::<u8>(), values.len())
        };
        Ok(result.iter().map(|value| *value != 0).collect())
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compare_i32_to_gpu(
        &mut self,
        values: &[i32],
        scalar: i32,
        op: ScalarCompareOp,
    ) -> Result<GpuBuffer> {
        if values.is_empty() {
            return Ok(GpuBuffer {
                buffer: allocate_buffer(&self.device, 1, "empty mask")?,
                len: 0,
                element_size: 1,
            });
        }
        let pipeline = self.pipeline("compare_i32_scalar")?.clone();
        let input = self.cached_input_buffer(values, CACHE_TAG_I32, "gpu i32 compare input")?;
        let output = allocate_buffer(&self.device, values.len(), "gpu i32 compare mask")?;

        let (cb, is_batch) = self.get_command_buffer()?;
        let encoder = cb.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;
        encode_i32_scalar_compare(
            &encoder,
            &pipeline,
            &input,
            &output,
            scalar,
            values.len(),
            op,
        )?;
        encoder.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "Metal compare")?;
        Ok(GpuBuffer {
            buffer: output,
            len: values.len(),
            element_size: 1,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compare_i64_to_gpu(
        &mut self,
        values: &[i64],
        scalar: i64,
        op: ScalarCompareOp,
    ) -> Result<GpuBuffer> {
        if values.is_empty() {
            return Ok(GpuBuffer {
                buffer: allocate_buffer(&self.device, 1, "empty mask")?,
                len: 0,
                element_size: 1,
            });
        }
        let pipeline = self.pipeline("compare_i64_scalar")?.clone();
        let input = self.cached_input_buffer(values, CACHE_TAG_I64, "gpu i64 compare input")?;
        let output = allocate_buffer(&self.device, values.len(), "gpu i64 compare mask")?;

        let (cb, is_batch) = self.get_command_buffer()?;
        let encoder = cb.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;
        encode_i64_scalar_compare(
            &encoder,
            &pipeline,
            &input,
            &output,
            scalar,
            values.len(),
            op,
        )?;
        encoder.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "Metal compare")?;
        Ok(GpuBuffer {
            buffer: output,
            len: values.len(),
            element_size: 1,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compare_f32_to_gpu(
        &mut self,
        values: &[f32],
        scalar: f32,
        op: ScalarCompareOp,
    ) -> Result<GpuBuffer> {
        if values.is_empty() {
            return Ok(GpuBuffer {
                buffer: allocate_buffer(&self.device, 1, "empty mask")?,
                len: 0,
                element_size: 1,
            });
        }
        let pipeline = self.pipeline("compare_f32_scalar")?.clone();
        let input = self.cached_input_buffer(values, CACHE_TAG_F32, "gpu f32 compare input")?;
        let output = allocate_buffer(&self.device, values.len(), "gpu f32 compare mask")?;

        let (cb, is_batch) = self.get_command_buffer()?;
        let encoder = cb.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;
        encode_f32_scalar_compare(
            &encoder,
            &pipeline,
            &input,
            &output,
            scalar,
            values.len(),
            op,
        )?;
        encoder.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "Metal compare")?;
        Ok(GpuBuffer {
            buffer: output,
            len: values.len(),
            element_size: 1,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn and_masks_gpu(&mut self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        assert_eq!(a.len, b.len, "and_masks_gpu: length mismatch");
        if a.len == 0 {
            return Ok(GpuBuffer {
                buffer: allocate_buffer(&self.device, 1, "empty mask")?,
                len: 0,
                element_size: 1,
            });
        }
        let pipeline = self.pipeline("and_masks")?.clone();
        let output = allocate_buffer(&self.device, a.len, "gpu and_masks output")?;
        let len = a.len as u32;

        let (cb, is_batch) = self.get_command_buffer()?;
        let encoder = cb.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&a.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&b.buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&len).cast::<c_void>(),
                std::mem::size_of::<u32>(),
                3,
            );
        }
        dispatch_threads(&encoder, &pipeline, a.len);
        encoder.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "Metal and_masks")?;
        Ok(GpuBuffer {
            buffer: output,
            len: a.len,
            element_size: 1,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn filter_to_indices_gpu(&mut self, mask: &GpuBuffer) -> Result<GpuBuffer> {
        // Must flush any pending batch — we need to read the atomic counter after completion.
        self.flush_batch()?;
        if mask.len == 0 {
            return Ok(GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty indices")?,
                len: 0,
                element_size: 4,
            });
        }
        let pipeline = self.pipeline("filter_to_indices")?.clone();
        // Allocate output for worst case (all rows pass).
        let output = allocate_buffer(
            &self.device,
            mask.len * std::mem::size_of::<u32>(),
            "gpu filter_to_indices output",
        )?;
        // Atomic counter for compaction.
        let count_buf =
            allocate_buffer(&self.device, std::mem::size_of::<u32>(), "gpu filter count")?;
        unsafe {
            std::ptr::write_bytes(
                count_buf.contents().as_ptr().cast::<u8>(),
                0,
                std::mem::size_of::<u32>(),
            );
        }
        let len = mask.len as u32;

        let command_buffer = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal command buffer".to_owned())
        })?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&mask.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buf), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&len).cast::<c_void>(),
                std::mem::size_of::<u32>(),
                3,
            );
        }
        dispatch_threads(&encoder, &pipeline, mask.len);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        if let Some(error) = command_buffer.error() {
            return Err(MettleError::Processing(format!(
                "Metal filter_to_indices failed: {error}"
            )));
        }
        // Read actual count from the atomic counter.
        let actual_count = unsafe { *count_buf.contents().as_ptr().cast::<u32>() } as usize;
        Ok(GpuBuffer {
            buffer: output,
            len: actual_count,
            element_size: 4,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn elementwise_f64(
        &mut self,
        a: &[f64],
        b: &[f64],
        op: ScalarProjectOp,
    ) -> Result<Vec<f64>> {
        if a.is_empty() {
            return Ok(Vec::new());
        }
        assert_eq!(a.len(), b.len(), "elementwise_f64: input length mismatch");

        // Metal doesn't support f64 in device buffers — downcast to f32 for GPU,
        // then widen back. Precision loss is acceptable for analytics workloads.
        let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();

        let pipeline = self.pipeline("elementwise_f32")?.clone();
        let input_a = make_buffer_from_slice(&self.device, &a_f32, "gpu elementwise input a")?;
        let input_b = make_buffer_from_slice(&self.device, &b_f32, "gpu elementwise input b")?;
        let output = allocate_buffer(
            &self.device,
            a.len() * std::mem::size_of::<f32>(),
            "gpu elementwise output",
        )?;

        let len = u32::try_from(a.len()).map_err(|_| {
            MettleError::Processing("elementwise_f64 length overflowed u32".to_owned())
        })?;
        let op_code = op.code();

        let command_buffer = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal command buffer".to_owned())
        })?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&input_a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&input_b), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::from(&len).cast::<c_void>(),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&op_code).cast::<c_void>(),
                std::mem::size_of::<u32>(),
                4,
            );
        }
        dispatch_threads(&encoder, &pipeline, a.len());
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MettleError::Processing(format!(
                "Metal elementwise command buffer failed: {error}"
            )));
        }

        let result = unsafe {
            std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), a.len())
        };
        Ok(result.iter().map(|&v| v as f64).collect())
    }

    /// Upload a PinnedColumn to a GpuBuffer (i32), keyed by data pointer.
    /// Int64 columns are truncated to i32 (sufficient for join keys).
    /// Cached across queries — no per-query hashing.
    #[cfg(target_os = "macos")]
    pub(super) fn column_to_gpu_buffer(&mut self, col: &PinnedColumn) -> Result<GpuBuffer> {
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
    pub(super) fn column_to_gpu_buffer(&mut self, _: &PinnedColumn) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }

    // ── GpuVector (count-passing) methods ──────────────────────────────
    // These encode into the batch command buffer without flushing.
    // The count stays on GPU — no CPU readback until end_batch().

    #[cfg(target_os = "macos")]
    pub(super) fn filter_to_indices_gv(&mut self, mask: &GpuBuffer) -> Result<GpuVector> {
        if mask.len == 0 {
            let data = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            let count_buf = allocate_buffer(&self.device, 4, "empty count")?;
            unsafe {
                std::ptr::write_bytes(count_buf.contents().as_ptr().cast::<u8>(), 0, 4);
            }
            let count = GpuBuffer {
                buffer: count_buf,
                len: 1,
                element_size: 4,
            };
            let dispatch_args = self.empty_dispatch_args()?;
            return Ok(GpuVector {
                data,
                count,
                dispatch_args,
                max_len: 0,
            });
        }
        let pipeline = self.pipeline("filter_to_indices")?.clone();
        let output = allocate_buffer(&self.device, mask.len * 4, "gv filter output")?;
        let count_buf = allocate_buffer(&self.device, 4, "gv filter count")?;
        unsafe {
            std::ptr::write_bytes(count_buf.contents().as_ptr().cast::<u8>(), 0, 4);
        }
        let dispatch_args = self.allocate_dispatch_args_buffer("gv filter dispatch args")?;
        let len = mask.len as u32;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&mask.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&count_buf), 0, 2);
            enc.setBytes_length_atIndex(NonNull::from(&len).cast::<c_void>(), 4, 3);
        }
        dispatch_threads(&enc, &pipeline, mask.len);
        enc.endEncoding();
        self.encode_dispatch_args_update_on(
            &cb,
            &GpuBuffer {
                buffer: count_buf.clone(),
                len: 1,
                element_size: 4,
            },
            &dispatch_args,
        )?;
        Self::commit_if_standalone(&cb, is_batch, "filter_to_indices_gv")?;
        let data = GpuBuffer {
            buffer: output,
            len: mask.len,
            element_size: 4,
        };
        let count = GpuBuffer {
            buffer: count_buf,
            len: 1,
            element_size: 4,
        };
        Ok(GpuVector {
            data,
            count,
            dispatch_args,
            max_len: mask.len,
        })
    }

    #[cfg(target_os = "macos")]
    fn gather_gv_impl(
        &mut self,
        kernel: &'static str,
        data: &GpuBuffer,
        indices: &GpuVector,
    ) -> Result<GpuVector> {
        if indices.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &indices.count,
                &indices.dispatch_args,
                0,
            ));
        }
        let pipeline = self.pipeline(kernel)?.clone();
        let output = allocate_buffer(&self.device, indices.max_len * 4, "gv gather output")?;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&indices.data.buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&indices.count.buffer), 0, 3);
        }
        Self::dispatch_gv_indirect(&enc, &indices.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, kernel)?;
        let out = GpuBuffer {
            buffer: output,
            len: indices.max_len,
            element_size: 4,
        };
        Ok(GpuVector::inherited_with_dispatch(
            out,
            &indices.count,
            &indices.dispatch_args,
            indices.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn gather_i32_gv(
        &mut self,
        data: &GpuBuffer,
        indices: &GpuVector,
    ) -> Result<GpuVector> {
        self.gather_gv_impl("gather_i32_v", data, indices)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn gather_f32_gv(
        &mut self,
        data: &GpuBuffer,
        indices: &GpuVector,
    ) -> Result<GpuVector> {
        self.gather_gv_impl("gather_f32_v", data, indices)
    }

    /// Gather i32 data as f32 (cast during gather). For integer columns in expressions.
    #[cfg(target_os = "macos")]
    pub(super) fn gather_i32_as_f32_gv(
        &mut self,
        data: &GpuBuffer,
        indices: &GpuVector,
    ) -> Result<GpuVector> {
        self.gather_gv_impl("gather_i32_as_f32_v", data, indices)
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn gather_i32_as_f32_gv(
        &mut self,
        _: &GpuBuffer,
        _: &GpuVector,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn reindex_gv(
        &mut self,
        base: &GpuVector,
        selection: &GpuVector,
    ) -> Result<GpuVector> {
        if selection.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &selection.count,
                &selection.dispatch_args,
                0,
            ));
        }
        let pipeline = self.pipeline("reindex_v")?.clone();
        let output = allocate_buffer(&self.device, selection.max_len * 4, "gv reindex output")?;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&base.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&selection.data.buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&selection.count.buffer), 0, 3);
        }
        Self::dispatch_gv_indirect(&enc, &selection.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "reindex_gv")?;
        let out = GpuBuffer {
            buffer: output,
            len: selection.max_len,
            element_size: 4,
        };
        Ok(GpuVector::inherited_with_dispatch(
            out,
            &selection.count,
            &selection.dispatch_args,
            selection.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compact_matches_gv(
        &mut self,
        matches: &GpuVector,
    ) -> Result<(GpuVector, GpuVector)> {
        if matches.max_len == 0 {
            let empty = |label: &str| -> Result<GpuVector> {
                let data = GpuBuffer {
                    buffer: allocate_buffer(&self.device, 4, label)?,
                    len: 0,
                    element_size: 4,
                };
                let count_buf = allocate_buffer(&self.device, 4, "empty count")?;
                unsafe {
                    std::ptr::write_bytes(count_buf.contents().as_ptr().cast::<u8>(), 0, 4);
                }
                let count = GpuBuffer {
                    buffer: count_buf,
                    len: 1,
                    element_size: 4,
                };
                Ok(GpuVector {
                    data,
                    count,
                    dispatch_args: self.empty_dispatch_args()?,
                    max_len: 0,
                })
            };
            return Ok((empty("empty probe")?, empty("empty build")?));
        }
        let pipeline = self.pipeline("compact_matches_v")?.clone();
        let probe_kept = allocate_buffer(&self.device, matches.max_len * 4, "gv probe_kept")?;
        let build_kept = allocate_buffer(&self.device, matches.max_len * 4, "gv build_kept")?;
        let out_count_buf = allocate_buffer(&self.device, 4, "gv compact count")?;
        unsafe {
            std::ptr::write_bytes(out_count_buf.contents().as_ptr().cast::<u8>(), 0, 4);
        }
        let out_dispatch_args = self.allocate_dispatch_args_buffer("gv compact dispatch args")?;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&matches.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&probe_kept), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&build_kept), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&out_count_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&matches.count.buffer), 0, 4);
        }
        Self::dispatch_gv_indirect(&enc, &matches.dispatch_args);
        enc.endEncoding();
        self.encode_dispatch_args_update_on(
            &cb,
            &GpuBuffer {
                buffer: out_count_buf.clone(),
                len: 1,
                element_size: 4,
            },
            &out_dispatch_args,
        )?;
        Self::commit_if_standalone(&cb, is_batch, "compact_matches_gv")?;
        let out_count = GpuBuffer {
            buffer: out_count_buf,
            len: 1,
            element_size: 4,
        };
        let pk = GpuVector {
            data: GpuBuffer {
                buffer: probe_kept,
                len: matches.max_len,
                element_size: 4,
            },
            count: GpuBuffer {
                buffer: out_count.buffer.clone(),
                len: 1,
                element_size: 4,
            },
            dispatch_args: GpuBuffer {
                buffer: out_dispatch_args.buffer.clone(),
                len: out_dispatch_args.len,
                element_size: out_dispatch_args.element_size,
            },
            max_len: matches.max_len,
        };
        let bk = GpuVector {
            data: GpuBuffer {
                buffer: build_kept,
                len: matches.max_len,
                element_size: 4,
            },
            count: out_count,
            dispatch_args: out_dispatch_args,
            max_len: matches.max_len,
        };
        Ok((pk, bk))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn hash_join_build_gv(
        &mut self,
        keys: &GpuVector,
        ht_capacity: u32,
    ) -> Result<(GpuBuffer, GpuBuffer)> {
        let cap = ht_capacity as usize;
        let ht_byte_len = cap * 4;
        let ht_keys_buf = allocate_buffer(&self.device, ht_byte_len, "gv hj keys")?;
        let ht_idx_buf = allocate_buffer(&self.device, ht_byte_len, "gv hj idx")?;
        unsafe {
            std::ptr::write_bytes(
                ht_keys_buf.contents().as_ptr().cast::<u8>(),
                0xFF,
                ht_byte_len,
            );
            std::ptr::write_bytes(
                ht_idx_buf.contents().as_ptr().cast::<u8>(),
                0xFF,
                ht_byte_len,
            );
        }
        if keys.max_len > 0 {
            let pipeline = self.pipeline("hash_join_build_i32_v")?.clone();
            let (cb, is_batch) = self.get_command_buffer()?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
            enc.setComputePipelineState(&pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&keys.data.buffer), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&ht_keys_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&ht_idx_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&keys.count.buffer), 0, 3);
                enc.setBytes_length_atIndex(NonNull::from(&ht_capacity).cast::<c_void>(), 4, 4);
            }
            Self::dispatch_gv_indirect(&enc, &keys.dispatch_args);
            enc.endEncoding();
            Self::commit_if_standalone(&cb, is_batch, "hash_join_build_gv")?;
        }
        Ok((
            GpuBuffer {
                buffer: ht_keys_buf,
                len: cap,
                element_size: 4,
            },
            GpuBuffer {
                buffer: ht_idx_buf,
                len: cap,
                element_size: 4,
            },
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn hash_join_probe_gv(
        &mut self,
        probe_keys: &GpuVector,
        ht_keys: &GpuBuffer,
        ht_idx: &GpuBuffer,
        ht_capacity: u32,
    ) -> Result<GpuVector> {
        if probe_keys.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &probe_keys.count,
                &probe_keys.dispatch_args,
                0,
            ));
        }
        let pipeline = self.pipeline("hash_join_probe_i32_v")?.clone();
        let output = allocate_buffer(&self.device, probe_keys.max_len * 4, "gv hj output")?;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&probe_keys.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&ht_keys.buffer), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&ht_idx.buffer), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&probe_keys.count.buffer), 0, 4);
            enc.setBytes_length_atIndex(NonNull::from(&ht_capacity).cast::<c_void>(), 4, 5);
        }
        Self::dispatch_gv_indirect(&enc, &probe_keys.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "hash_join_probe_gv")?;
        let out = GpuBuffer {
            buffer: output,
            len: probe_keys.max_len,
            element_size: 4,
        };
        Ok(GpuVector::inherited_with_dispatch(
            out,
            &probe_keys.count,
            &probe_keys.dispatch_args,
            probe_keys.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn dense_lookup_build_i32_gv(
        &mut self,
        keys: &GpuVector,
        min_key: i32,
        slot_count: u32,
    ) -> Result<GpuBuffer> {
        if slot_count == 0 {
            return Err(MettleError::Processing(
                "dense lookup slot count must be > 0".to_owned(),
            ));
        }
        let lookup = allocate_buffer(
            &self.device,
            slot_count as usize * std::mem::size_of::<i32>(),
            "dense lookup rows",
        )?;
        unsafe {
            std::ptr::write_bytes(
                lookup.contents().as_ptr().cast::<u8>(),
                0xFFu8,
                slot_count as usize * std::mem::size_of::<i32>(),
            );
        }
        if keys.max_len > 0 {
            let pipeline = self.pipeline("dense_lookup_build_i32_v")?.clone();
            let (cb, is_batch) = self.get_command_buffer()?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
            enc.setComputePipelineState(&pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&keys.data.buffer), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&lookup), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&keys.count.buffer), 0, 2);
                enc.setBytes_length_atIndex(NonNull::from(&min_key).cast::<c_void>(), 4, 3);
                enc.setBytes_length_atIndex(NonNull::from(&slot_count).cast::<c_void>(), 4, 4);
            }
            Self::dispatch_gv_indirect(&enc, &keys.dispatch_args);
            enc.endEncoding();
            Self::commit_if_standalone(&cb, is_batch, "dense_lookup_build_i32_gv")?;
        }
        Ok(GpuBuffer {
            buffer: lookup,
            len: slot_count as usize,
            element_size: 4,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn dense_lookup_probe_i32_gv(
        &mut self,
        probe_keys: &GpuVector,
        lookup_rows: &GpuBuffer,
        min_key: i32,
        slot_count: u32,
    ) -> Result<GpuVector> {
        if probe_keys.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &probe_keys.count,
                &probe_keys.dispatch_args,
                0,
            ));
        }
        let pipeline = self.pipeline("dense_lookup_probe_i32_v")?.clone();
        let output = allocate_buffer(
            &self.device,
            probe_keys.max_len * std::mem::size_of::<i32>(),
            "dense lookup probe output",
        )?;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&probe_keys.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&lookup_rows.buffer), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&probe_keys.count.buffer), 0, 3);
            enc.setBytes_length_atIndex(NonNull::from(&min_key).cast::<c_void>(), 4, 4);
            enc.setBytes_length_atIndex(NonNull::from(&slot_count).cast::<c_void>(), 4, 5);
        }
        Self::dispatch_gv_indirect(&enc, &probe_keys.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "dense_lookup_probe_i32_gv")?;
        let out = GpuBuffer {
            buffer: output,
            len: probe_keys.max_len,
            element_size: 4,
        };
        Ok(GpuVector::inherited_with_dispatch(
            out,
            &probe_keys.count,
            &probe_keys.dispatch_args,
            probe_keys.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn dense_lookup_build_from_indices_i32_gv(
        &mut self,
        column: &GpuBuffer,
        row_ids: &GpuVector,
        min_key: i32,
        slot_count: u32,
    ) -> Result<GpuBuffer> {
        if slot_count == 0 {
            return Err(MettleError::Processing(
                "dense lookup slot count must be > 0".to_owned(),
            ));
        }
        let lookup = allocate_buffer(
            &self.device,
            slot_count as usize * std::mem::size_of::<i32>(),
            "dense lookup rows",
        )?;
        unsafe {
            std::ptr::write_bytes(
                lookup.contents().as_ptr().cast::<u8>(),
                0xFFu8,
                slot_count as usize * std::mem::size_of::<i32>(),
            );
        }
        if row_ids.max_len > 0 {
            let pipeline = self
                .pipeline("dense_lookup_build_from_indices_i32_v")?
                .clone();
            let (cb, is_batch) = self.get_command_buffer()?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
            enc.setComputePipelineState(&pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&column.buffer), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&row_ids.data.buffer), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&lookup), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&row_ids.count.buffer), 0, 3);
                enc.setBytes_length_atIndex(NonNull::from(&min_key).cast::<c_void>(), 4, 4);
                enc.setBytes_length_atIndex(NonNull::from(&slot_count).cast::<c_void>(), 4, 5);
            }
            Self::dispatch_gv_indirect(&enc, &row_ids.dispatch_args);
            enc.endEncoding();
            Self::commit_if_standalone(&cb, is_batch, "dense_lookup_build_from_indices_i32_gv")?;
        }
        Ok(GpuBuffer {
            buffer: lookup,
            len: slot_count as usize,
            element_size: 4,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn dense_lookup_probe_from_indices_i32_gv(
        &mut self,
        column: &GpuBuffer,
        row_ids: &GpuVector,
        lookup_rows: &GpuBuffer,
        min_key: i32,
        slot_count: u32,
    ) -> Result<GpuVector> {
        if row_ids.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &row_ids.count,
                &row_ids.dispatch_args,
                0,
            ));
        }
        let pipeline = self
            .pipeline("dense_lookup_probe_from_indices_i32_v")?
            .clone();
        let output = allocate_buffer(
            &self.device,
            row_ids.max_len * std::mem::size_of::<i32>(),
            "dense lookup probe output",
        )?;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&column.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&row_ids.data.buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&lookup_rows.buffer), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&row_ids.count.buffer), 0, 4);
            enc.setBytes_length_atIndex(NonNull::from(&min_key).cast::<c_void>(), 4, 5);
            enc.setBytes_length_atIndex(NonNull::from(&slot_count).cast::<c_void>(), 4, 6);
        }
        Self::dispatch_gv_indirect(&enc, &row_ids.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "dense_lookup_probe_from_indices_i32_gv")?;
        let out = GpuBuffer {
            buffer: output,
            len: row_ids.max_len,
            element_size: 4,
        };
        Ok(GpuVector::inherited_with_dispatch(
            out,
            &row_ids.count,
            &row_ids.dispatch_args,
            row_ids.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn build_membership_filter_gv(
        &mut self,
        keys: &GpuVector,
        bit_count: u32,
    ) -> Result<GpuBuffer> {
        if bit_count < 32 || !bit_count.is_power_of_two() {
            return Err(MettleError::Processing(
                "membership filter bit count must be a power of two >= 32".to_owned(),
            ));
        }
        let word_count = (bit_count / 32) as usize;
        let word_bytes = word_count * 4;
        let membership = allocate_buffer(&self.device, word_bytes, "membership filter words")?;
        unsafe {
            std::ptr::write_bytes(membership.contents().as_ptr().cast::<u8>(), 0, word_bytes);
        }
        if keys.max_len == 0 {
            return Ok(GpuBuffer {
                buffer: membership,
                len: word_count,
                element_size: 4,
            });
        }
        let pipeline = self.pipeline("build_membership_filter_v")?.clone();
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&keys.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&membership), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&keys.count.buffer), 0, 2);
            enc.setBytes_length_atIndex(NonNull::from(&bit_count).cast::<c_void>(), 4, 3);
        }
        Self::dispatch_gv_indirect(&enc, &keys.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "build_membership_filter_gv")?;
        Ok(GpuBuffer {
            buffer: membership,
            len: word_count,
            element_size: 4,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn membership_filter_mask_gv(
        &mut self,
        probe_keys: &GpuVector,
        membership: &GpuBuffer,
        bit_count: u32,
    ) -> Result<GpuBuffer> {
        if bit_count < 32 || !bit_count.is_power_of_two() {
            return Err(MettleError::Processing(
                "membership filter bit count must be a power of two >= 32".to_owned(),
            ));
        }
        if probe_keys.max_len == 0 {
            return Ok(GpuBuffer {
                buffer: allocate_buffer(&self.device, 1, "empty membership mask")?,
                len: 0,
                element_size: 1,
            });
        }
        let pipeline = self.pipeline("membership_filter_mask_v")?.clone();
        let mask = allocate_buffer(&self.device, probe_keys.max_len, "membership filter mask")?;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&probe_keys.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&membership.buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&mask), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&probe_keys.count.buffer), 0, 3);
            enc.setBytes_length_atIndex(NonNull::from(&bit_count).cast::<c_void>(), 4, 4);
        }
        Self::dispatch_gv_indirect(&enc, &probe_keys.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "membership_filter_mask_gv")?;
        Ok(GpuBuffer {
            buffer: mask,
            len: probe_keys.max_len,
            element_size: 1,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn project_f32_scalar_gv(
        &mut self,
        input: &GpuVector,
        scalar: f32,
        op: ScalarProjectOp,
    ) -> Result<GpuVector> {
        if input.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &input.count,
                &input.dispatch_args,
                0,
            ));
        }
        let pipeline = self.pipeline("project_f32_scalar_v")?.clone();
        let output = allocate_buffer(&self.device, input.max_len * 4, "gv project output")?;
        let op_code = op.code();
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&input.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 1);
            enc.setBytes_length_atIndex(NonNull::from(&scalar).cast::<c_void>(), 4, 2);
            enc.setBuffer_offset_atIndex(Some(&input.count.buffer), 0, 3);
            enc.setBytes_length_atIndex(NonNull::from(&op_code).cast::<c_void>(), 4, 4);
        }
        Self::dispatch_gv_indirect(&enc, &input.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "project_f32_scalar_gv")?;
        let out = GpuBuffer {
            buffer: output,
            len: input.max_len,
            element_size: 4,
        };
        Ok(GpuVector::inherited_with_dispatch(
            out,
            &input.count,
            &input.dispatch_args,
            input.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn elementwise_f32_gv(
        &mut self,
        a: &GpuVector,
        b: &GpuVector,
        op: ScalarProjectOp,
    ) -> Result<GpuVector> {
        debug_assert!(
            b.max_len >= a.max_len,
            "elementwise_f32_gv: b shorter than a"
        );
        if a.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &a.count,
                &a.dispatch_args,
                0,
            ));
        }
        let pipeline = self.pipeline("elementwise_f32_v")?.clone();
        let output = allocate_buffer(&self.device, a.max_len * 4, "gv elementwise output")?;
        let op_code = op.code();
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b.data.buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&a.count.buffer), 0, 3);
            enc.setBytes_length_atIndex(NonNull::from(&op_code).cast::<c_void>(), 4, 4);
        }
        Self::dispatch_gv_indirect(&enc, &a.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "elementwise_f32_gv")?;
        let out = GpuBuffer {
            buffer: output,
            len: a.max_len,
            element_size: 4,
        };
        Ok(GpuVector::inherited_with_dispatch(
            out,
            &a.count,
            &a.dispatch_args,
            a.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn project_revenue_from_indices_gv(
        &mut self,
        extendedprice: &GpuBuffer,
        discount: &GpuBuffer,
        indices: &GpuVector,
    ) -> Result<GpuVector> {
        if indices.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &indices.count,
                &indices.dispatch_args,
                0,
            ));
        }
        let pipeline = self.pipeline("project_revenue_from_indices_v")?.clone();
        let output = allocate_buffer(&self.device, indices.max_len * 4, "gv revenue output")?;
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&extendedprice.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&discount.buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&indices.data.buffer), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&indices.count.buffer), 0, 4);
        }
        Self::dispatch_gv_indirect(&enc, &indices.dispatch_args);
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "project_revenue_from_indices_gv")?;
        let out = GpuBuffer {
            buffer: output,
            len: indices.max_len,
            element_size: 4,
        };
        Ok(GpuVector::inherited_with_dispatch(
            out,
            &indices.count,
            &indices.dispatch_args,
            indices.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn accumulate_dense_slots_gv(
        &mut self,
        slot_ids: &GpuVector,
        values: &GpuVector,
        dense_rows: &GpuVector,
    ) -> Result<GpuVector> {
        debug_assert_eq!(slot_ids.max_len, values.max_len);
        if dense_rows.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &dense_rows.count,
                &dense_rows.dispatch_args,
                0,
            ));
        }

        let slot_count = u32::try_from(dense_rows.max_len)
            .map_err(|_| MettleError::Processing("dense slot count overflowed u32".to_owned()))?;
        let byte_len = dense_rows.max_len * std::mem::size_of::<u32>();
        let sums = allocate_buffer(&self.device, byte_len, "dense slot sums")?;
        unsafe {
            std::ptr::write_bytes(sums.contents().as_ptr().cast::<u8>(), 0, byte_len);
        }

        if slot_ids.max_len > 0 {
            let pipeline = self.pipeline("accumulate_dense_slots_v")?.clone();
            let (cb, is_batch) = self.get_command_buffer()?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
            enc.setComputePipelineState(&pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&slot_ids.data.buffer), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&values.data.buffer), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&sums), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&slot_ids.count.buffer), 0, 3);
                enc.setBytes_length_atIndex(NonNull::from(&slot_count).cast::<c_void>(), 4, 4);
            }
            Self::dispatch_gv_indirect(&enc, &slot_ids.dispatch_args);
            enc.endEncoding();
            Self::commit_if_standalone(&cb, is_batch, "accumulate_dense_slots_gv")?;
        }

        Ok(GpuVector::inherited_with_dispatch(
            GpuBuffer {
                buffer: sums,
                len: dense_rows.max_len,
                element_size: 4,
            },
            &dense_rows.count,
            &dense_rows.dispatch_args,
            dense_rows.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn accumulate_revenue_dense_slots_from_indices_gv(
        &mut self,
        slot_ids: &GpuVector,
        extendedprice: &GpuBuffer,
        discount: &GpuBuffer,
        row_ids: &GpuVector,
        dense_rows: &GpuVector,
    ) -> Result<GpuVector> {
        debug_assert_eq!(slot_ids.max_len, row_ids.max_len);
        if dense_rows.max_len == 0 {
            let out = GpuBuffer {
                buffer: allocate_buffer(&self.device, 4, "empty")?,
                len: 0,
                element_size: 4,
            };
            return Ok(GpuVector::inherited_with_dispatch(
                out,
                &dense_rows.count,
                &dense_rows.dispatch_args,
                0,
            ));
        }

        let slot_count = u32::try_from(dense_rows.max_len)
            .map_err(|_| MettleError::Processing("dense slot count overflowed u32".to_owned()))?;
        let byte_len = dense_rows.max_len * std::mem::size_of::<u32>();
        let sums = allocate_buffer(&self.device, byte_len, "dense revenue slot sums")?;
        unsafe {
            std::ptr::write_bytes(sums.contents().as_ptr().cast::<u8>(), 0, byte_len);
        }

        if slot_ids.max_len > 0 {
            let pipeline = self
                .pipeline("accumulate_revenue_dense_slots_from_indices_v")?
                .clone();
            let (cb, is_batch) = self.get_command_buffer()?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
            enc.setComputePipelineState(&pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&slot_ids.data.buffer), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&extendedprice.buffer), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&discount.buffer), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&row_ids.data.buffer), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&sums), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&row_ids.count.buffer), 0, 5);
                enc.setBytes_length_atIndex(NonNull::from(&slot_count).cast::<c_void>(), 4, 6);
            }
            Self::dispatch_gv_indirect(&enc, &row_ids.dispatch_args);
            enc.endEncoding();
            Self::commit_if_standalone(
                &cb,
                is_batch,
                "accumulate_revenue_dense_slots_from_indices_gv",
            )?;
        }

        Ok(GpuVector::inherited_with_dispatch(
            GpuBuffer {
                buffer: sums,
                len: dense_rows.max_len,
                element_size: 4,
            },
            &dense_rows.count,
            &dense_rows.dispatch_args,
            dense_rows.max_len,
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn grouped_aggregate_exact_gv(
        &mut self,
        key_lanes: &[GpuVector],
        specs: &[GpuGroupedAggregateSpec<'_>],
    ) -> Result<GpuGroupedAggregateRows> {
        if key_lanes.is_empty() || specs.is_empty() {
            return Ok(GpuGroupedAggregateRows {
                key_lanes: Vec::new(),
                slots: Vec::new(),
            });
        }
        if key_lanes.len() > MAX_GROUP_KEY_LANES {
            return Err(MettleError::Processing(format!(
                "grouped aggregate supports at most {MAX_GROUP_KEY_LANES} key lanes"
            )));
        }
        let max_len = key_lanes[0].max_len;
        if max_len == 0 {
            return Ok(GpuGroupedAggregateRows {
                key_lanes: vec![Vec::new(); key_lanes.len()],
                slots: specs
                    .iter()
                    .map(|spec| GpuGroupedAggregateSlotResult {
                        values: match spec.state_kind {
                            GpuGroupedAggregateStateKind::Count => None,
                            GpuGroupedAggregateStateKind::Sum
                            | GpuGroupedAggregateStateKind::Min
                            | GpuGroupedAggregateStateKind::Max
                            | GpuGroupedAggregateStateKind::Avg => Some(Vec::new()),
                        },
                        counts: match spec.state_kind {
                            GpuGroupedAggregateStateKind::Count
                            | GpuGroupedAggregateStateKind::Avg => Some(Vec::new()),
                            GpuGroupedAggregateStateKind::Sum
                            | GpuGroupedAggregateStateKind::Min
                            | GpuGroupedAggregateStateKind::Max => None,
                        },
                    })
                    .collect(),
            });
        }

        for key_lane in key_lanes.iter().skip(1) {
            debug_assert_eq!(key_lane.max_len, max_len);
        }

        let ht_capacity = (max_len.saturating_mul(2).max(16)).next_power_of_two();
        let ht_capacity_u32 = u32::try_from(ht_capacity).map_err(|_| {
            MettleError::Processing("grouped aggregate capacity overflowed u32".to_owned())
        })?;
        let key_lane_count_u32 = u32::try_from(key_lanes.len()).map_err(|_| {
            MettleError::Processing("grouped aggregate key-lane count overflowed u32".to_owned())
        })?;
        let byte_len = ht_capacity * std::mem::size_of::<u32>();

        let ht_states = allocate_buffer(&self.device, byte_len, "grouped aggregate states")?;
        unsafe {
            std::ptr::write_bytes(ht_states.contents().as_ptr().cast::<u8>(), 0x00u8, byte_len);
        }
        let mut ht_key_buffers = Vec::with_capacity(MAX_GROUP_KEY_LANES);
        for lane_idx in 0..MAX_GROUP_KEY_LANES {
            let buffer = allocate_buffer(
                &self.device,
                byte_len,
                &format!("grouped aggregate key lane {lane_idx}"),
            )?;
            unsafe {
                std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0x00u8, byte_len);
            }
            ht_key_buffers.push(buffer);
        }
        let zero_key = make_buffer_from_slice(&self.device, &[0i32], "grouped aggregate zero key")?;

        let (cb, is_batch) = self.get_command_buffer()?;

        let build_pipeline = self.pipeline("grouped_aggregate_build_keys_v")?.clone();
        let build_enc = cb.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable(
                "failed to create grouped aggregate build encoder".to_owned(),
            )
        })?;
        build_enc.setComputePipelineState(&build_pipeline);
        for lane_idx in 0..MAX_GROUP_KEY_LANES {
            let buffer = key_lanes
                .get(lane_idx)
                .map(|lane| &lane.data.buffer)
                .unwrap_or(&zero_key);
            unsafe {
                build_enc.setBuffer_offset_atIndex(Some(buffer), 0, lane_idx);
            }
        }
        unsafe {
            build_enc.setBuffer_offset_atIndex(Some(&ht_states), 0, 4);
            build_enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[0]), 0, 5);
            build_enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[1]), 0, 6);
            build_enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[2]), 0, 7);
            build_enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[3]), 0, 8);
            build_enc.setBuffer_offset_atIndex(Some(&key_lanes[0].count.buffer), 0, 9);
            build_enc.setBytes_length_atIndex(
                NonNull::from(&ht_capacity_u32).cast::<c_void>(),
                4,
                10,
            );
            build_enc.setBytes_length_atIndex(
                NonNull::from(&key_lane_count_u32).cast::<c_void>(),
                4,
                11,
            );
        }
        Self::dispatch_gv_indirect(&build_enc, &key_lanes[0].dispatch_args);
        build_enc.endEncoding();

        enum SlotBuffers {
            Count(Retained<ProtocolObject<dyn MTLBuffer>>),
            Value(Retained<ProtocolObject<dyn MTLBuffer>>),
            Avg {
                sums: Retained<ProtocolObject<dyn MTLBuffer>>,
                counts: Retained<ProtocolObject<dyn MTLBuffer>>,
            },
        }

        let mut slot_buffers = Vec::with_capacity(specs.len());
        for spec in specs {
            let values_buffer = spec.values.map(|values| &values.data.buffer);
            match spec.state_kind {
                GpuGroupedAggregateStateKind::Count => {
                    let counts =
                        allocate_buffer(&self.device, byte_len, "grouped aggregate counts")?;
                    unsafe {
                        std::ptr::write_bytes(
                            counts.contents().as_ptr().cast::<u8>(),
                            0x00u8,
                            byte_len,
                        );
                    }
                    let pipeline = self.pipeline("grouped_aggregate_update_count_v")?.clone();
                    let enc = cb.computeCommandEncoder().ok_or_else(|| {
                        MettleError::MetalUnavailable(
                            "failed to create grouped aggregate count encoder".to_owned(),
                        )
                    })?;
                    enc.setComputePipelineState(&pipeline);
                    for lane_idx in 0..MAX_GROUP_KEY_LANES {
                        let buffer = key_lanes
                            .get(lane_idx)
                            .map(|lane| &lane.data.buffer)
                            .unwrap_or(&zero_key);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(buffer), 0, lane_idx);
                        }
                    }
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&ht_states), 0, 4);
                        enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[0]), 0, 5);
                        enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[1]), 0, 6);
                        enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[2]), 0, 7);
                        enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[3]), 0, 8);
                        enc.setBuffer_offset_atIndex(Some(&counts), 0, 9);
                        enc.setBuffer_offset_atIndex(Some(&key_lanes[0].count.buffer), 0, 10);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&ht_capacity_u32).cast::<c_void>(),
                            4,
                            11,
                        );
                        enc.setBytes_length_atIndex(
                            NonNull::from(&key_lane_count_u32).cast::<c_void>(),
                            4,
                            12,
                        );
                    }
                    Self::dispatch_gv_indirect(&enc, &key_lanes[0].dispatch_args);
                    enc.endEncoding();
                    slot_buffers.push(SlotBuffers::Count(counts));
                }
                GpuGroupedAggregateStateKind::Sum
                | GpuGroupedAggregateStateKind::Min
                | GpuGroupedAggregateStateKind::Max
                | GpuGroupedAggregateStateKind::Avg => {
                    let Some(values_buffer) = values_buffer else {
                        return Err(MettleError::Processing(
                            "grouped aggregate value slot is missing its input vector".to_owned(),
                        ));
                    };
                    let sums = allocate_buffer(&self.device, byte_len, "grouped aggregate values")?;
                    let init_bits = match spec.state_kind {
                        GpuGroupedAggregateStateKind::Sum | GpuGroupedAggregateStateKind::Avg => {
                            0x00000000u32
                        }
                        GpuGroupedAggregateStateKind::Min => f32::INFINITY.to_bits(),
                        GpuGroupedAggregateStateKind::Max => f32::NEG_INFINITY.to_bits(),
                        GpuGroupedAggregateStateKind::Count => unreachable!(),
                    };
                    unsafe {
                        let ptr = sums.contents().as_ptr().cast::<u32>();
                        for slot in 0..ht_capacity {
                            *ptr.add(slot) = init_bits;
                        }
                    }

                    let (pipeline_name, maybe_counts) = match spec.state_kind {
                        GpuGroupedAggregateStateKind::Sum => {
                            ("grouped_aggregate_update_sum_v", None)
                        }
                        GpuGroupedAggregateStateKind::Min => {
                            ("grouped_aggregate_update_min_v", None)
                        }
                        GpuGroupedAggregateStateKind::Max => {
                            ("grouped_aggregate_update_max_v", None)
                        }
                        GpuGroupedAggregateStateKind::Avg => {
                            let counts = allocate_buffer(
                                &self.device,
                                byte_len,
                                "grouped aggregate avg counts",
                            )?;
                            unsafe {
                                std::ptr::write_bytes(
                                    counts.contents().as_ptr().cast::<u8>(),
                                    0x00u8,
                                    byte_len,
                                );
                            }
                            ("grouped_aggregate_update_avg_v", Some(counts))
                        }
                        GpuGroupedAggregateStateKind::Count => unreachable!(),
                    };

                    let pipeline = self.pipeline(pipeline_name)?.clone();
                    let enc = cb.computeCommandEncoder().ok_or_else(|| {
                        MettleError::MetalUnavailable(
                            "failed to create grouped aggregate value encoder".to_owned(),
                        )
                    })?;
                    enc.setComputePipelineState(&pipeline);
                    for lane_idx in 0..MAX_GROUP_KEY_LANES {
                        let buffer = key_lanes
                            .get(lane_idx)
                            .map(|lane| &lane.data.buffer)
                            .unwrap_or(&zero_key);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(buffer), 0, lane_idx);
                        }
                    }
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(values_buffer), 0, 4);
                        enc.setBuffer_offset_atIndex(Some(&ht_states), 0, 5);
                        enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[0]), 0, 6);
                        enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[1]), 0, 7);
                        enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[2]), 0, 8);
                        enc.setBuffer_offset_atIndex(Some(&ht_key_buffers[3]), 0, 9);
                        enc.setBuffer_offset_atIndex(Some(&sums), 0, 10);
                        let count_buffer = maybe_counts.as_ref().unwrap_or(&zero_key);
                        enc.setBuffer_offset_atIndex(Some(count_buffer), 0, 11);
                        enc.setBuffer_offset_atIndex(Some(&key_lanes[0].count.buffer), 0, 12);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&ht_capacity_u32).cast::<c_void>(),
                            4,
                            13,
                        );
                        enc.setBytes_length_atIndex(
                            NonNull::from(&key_lane_count_u32).cast::<c_void>(),
                            4,
                            14,
                        );
                    }
                    Self::dispatch_gv_indirect(&enc, &key_lanes[0].dispatch_args);
                    enc.endEncoding();

                    if let Some(counts) = maybe_counts {
                        slot_buffers.push(SlotBuffers::Avg { sums, counts });
                    } else {
                        slot_buffers.push(SlotBuffers::Value(sums));
                    }
                }
            }
        }

        if !is_batch {
            cb.commit();
            cb.waitUntilCompleted();
            if let Some(error) = cb.error() {
                return Err(MettleError::Processing(format!(
                    "grouped aggregate command buffer failed: {error}"
                )));
            }
        }

        let states_ptr = ht_states.contents().as_ptr().cast::<u32>();
        let key_ptrs: Vec<*const u32> = ht_key_buffers
            .iter()
            .map(|buffer| buffer.contents().as_ptr().cast::<u32>() as *const u32)
            .collect();
        let live_slots: Vec<usize> = (0..ht_capacity)
            .filter(|&slot| unsafe { *states_ptr.add(slot) } == 2u32)
            .collect();

        let mut key_results = vec![Vec::with_capacity(live_slots.len()); key_lanes.len()];
        for &slot in &live_slots {
            for lane_idx in 0..key_lanes.len() {
                let key = unsafe { *key_ptrs[lane_idx].add(slot) } as i32;
                key_results[lane_idx].push(key);
            }
        }

        let mut slot_results = Vec::with_capacity(slot_buffers.len());
        for buffers in slot_buffers {
            match buffers {
                SlotBuffers::Count(counts) => {
                    let ptr = counts.contents().as_ptr().cast::<u32>();
                    slot_results.push(GpuGroupedAggregateSlotResult {
                        values: None,
                        counts: Some(
                            live_slots
                                .iter()
                                .map(|&slot| unsafe { *ptr.add(slot) })
                                .collect(),
                        ),
                    });
                }
                SlotBuffers::Value(values) => {
                    let ptr = values.contents().as_ptr().cast::<u32>();
                    slot_results.push(GpuGroupedAggregateSlotResult {
                        values: Some(
                            live_slots
                                .iter()
                                .map(|&slot| f32::from_bits(unsafe { *ptr.add(slot) }))
                                .collect(),
                        ),
                        counts: None,
                    });
                }
                SlotBuffers::Avg { sums, counts } => {
                    let sum_ptr = sums.contents().as_ptr().cast::<u32>();
                    let count_ptr = counts.contents().as_ptr().cast::<u32>();
                    slot_results.push(GpuGroupedAggregateSlotResult {
                        values: Some(
                            live_slots
                                .iter()
                                .map(|&slot| f32::from_bits(unsafe { *sum_ptr.add(slot) }))
                                .collect(),
                        ),
                        counts: Some(
                            live_slots
                                .iter()
                                .map(|&slot| unsafe { *count_ptr.add(slot) })
                                .collect(),
                        ),
                    });
                }
            }
        }

        Ok(GpuGroupedAggregateRows {
            key_lanes: key_results,
            slots: slot_results,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn compact_grouped_rows(
        &mut self,
        ht_row_ids: &GpuBuffer,
        ht_secondary_order_values: &GpuBuffer,
        ht_aggregate_values: &GpuBuffer,
        ht_capacity: u32,
    ) -> Result<GpuDenseGroupedRows> {
        let max_len = ht_capacity as usize;
        let dense_row_ids =
            allocate_buffer(&self.device, max_len.max(1) * 4, "dense grouped row ids")?;
        let dense_secondary =
            allocate_buffer(&self.device, max_len.max(1) * 4, "dense grouped secondary")?;
        let dense_sums = allocate_buffer(&self.device, max_len.max(1) * 4, "dense grouped sums")?;
        let dense_count_buf = allocate_buffer(&self.device, 4, "dense grouped count")?;
        unsafe {
            std::ptr::write_bytes(dense_count_buf.contents().as_ptr().cast::<u8>(), 0, 4);
        }
        let dense_dispatch_args =
            self.allocate_dispatch_args_buffer("dense grouped dispatch args")?;
        let (cb, is_batch) = self.get_command_buffer()?;

        if max_len > 0 {
            let pipeline = self.pipeline("compact_grouped_rows")?.clone();
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| MettleError::MetalUnavailable("no encoder".into()))?;
            enc.setComputePipelineState(&pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&ht_row_ids.buffer), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&ht_secondary_order_values.buffer), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&ht_aggregate_values.buffer), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&dense_row_ids), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&dense_secondary), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&dense_sums), 0, 5);
                enc.setBuffer_offset_atIndex(Some(&dense_count_buf), 0, 6);
                enc.setBytes_length_atIndex(NonNull::from(&ht_capacity).cast::<c_void>(), 4, 7);
            }
            dispatch_threads(&enc, &pipeline, max_len);
            enc.endEncoding();
            self.encode_dispatch_args_update_on(
                &cb,
                &GpuBuffer {
                    buffer: dense_count_buf.clone(),
                    len: 1,
                    element_size: 4,
                },
                &dense_dispatch_args,
            )?;
            Self::commit_if_standalone(&cb, is_batch, "compact_grouped_rows")?;
        }

        let count = GpuBuffer {
            buffer: dense_count_buf,
            len: 1,
            element_size: 4,
        };
        let dispatch = GpuBuffer {
            buffer: dense_dispatch_args.buffer.clone(),
            len: dense_dispatch_args.len,
            element_size: dense_dispatch_args.element_size,
        };
        Ok(GpuDenseGroupedRows {
            representative_row_ids: GpuVector {
                data: GpuBuffer {
                    buffer: dense_row_ids,
                    len: max_len,
                    element_size: 4,
                },
                count: GpuBuffer {
                    buffer: count.buffer.clone(),
                    len: 1,
                    element_size: 4,
                },
                dispatch_args: GpuBuffer {
                    buffer: dispatch.buffer.clone(),
                    len: dispatch.len,
                    element_size: dispatch.element_size,
                },
                max_len,
            },
            secondary_order_values: GpuVector {
                data: GpuBuffer {
                    buffer: dense_secondary,
                    len: max_len,
                    element_size: 4,
                },
                count: GpuBuffer {
                    buffer: count.buffer.clone(),
                    len: 1,
                    element_size: 4,
                },
                dispatch_args: GpuBuffer {
                    buffer: dispatch.buffer.clone(),
                    len: dispatch.len,
                    element_size: dispatch.element_size,
                },
                max_len,
            },
            aggregate_values: GpuVector {
                data: GpuBuffer {
                    buffer: dense_sums,
                    len: max_len,
                    element_size: 4,
                },
                count,
                dispatch_args: dispatch,
                max_len,
            },
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn topk_dense_grouped_rows(
        &mut self,
        rows: &GpuDenseGroupedRows,
        limit: usize,
        use_secondary_order: bool,
    ) -> Result<GpuGroupedTopKRows> {
        if limit == 0 || limit > 64 {
            return Err(MettleError::Processing(
                "topk_dense_grouped_rows requires 1 <= limit <= 64".to_owned(),
            ));
        }
        let limit_u32 = u32::try_from(limit).map_err(|_| {
            MettleError::Processing("grouped top-k limit overflowed u32".to_owned())
        })?;
        let secondary_flag = if use_secondary_order { 1u32 } else { 0u32 };
        let threads_per_group = GV_THREADS_PER_GROUP as u32;

        let out_row_ids = allocate_buffer(&self.device, limit * 4, "grouped topk row ids")?;
        let out_sums = allocate_buffer(&self.device, limit * 4, "grouped topk sums")?;
        let out_count_buf = allocate_buffer(&self.device, 4, "grouped topk count")?;
        let out_dispatch_args = self.allocate_dispatch_args_buffer("grouped topk dispatch args")?;
        unsafe {
            std::ptr::write_bytes(out_count_buf.contents().as_ptr().cast::<u8>(), 0, 4);
        }

        // Parallel kernel budget: tg_size * limit * 12 bytes (sums+dates+row_ids)
        // + tg_size * 4 bytes (counts) must fit in 32 KB threadgroup memory.
        // For tg_size=128 this caps limit at 20. Larger limits fall back to
        // the single-threaded kernel.
        const PARALLEL_TG_SIZE: usize = 128;
        const PARALLEL_LIMIT_CAP: usize = 20;
        let use_parallel = limit <= PARALLEL_LIMIT_CAP;

        let pipeline_name = if use_parallel {
            "topk_dense_grouped_rows_parallel_v"
        } else {
            "topk_dense_grouped_rows"
        };
        let pipeline = self.pipeline(pipeline_name)?.clone();
        let (cb, is_batch) = self.get_command_buffer()?;
        let enc = cb.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create grouped top-k encoder".to_owned())
        })?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&rows.representative_row_ids.data.buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&rows.secondary_order_values.data.buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&rows.aggregate_values.data.buffer), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&rows.aggregate_values.count.buffer), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&out_row_ids), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&out_sums), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&out_count_buf), 0, 6);
            enc.setBuffer_offset_atIndex(Some(&out_dispatch_args.buffer), 0, 7);
            enc.setBytes_length_atIndex(NonNull::from(&limit_u32).cast::<c_void>(), 4, 8);
            enc.setBytes_length_atIndex(NonNull::from(&secondary_flag).cast::<c_void>(), 4, 9);
            enc.setBytes_length_atIndex(NonNull::from(&threads_per_group).cast::<c_void>(), 4, 10);
        }
        if use_parallel {
            // Threadgroup memory: 3 arrays of (tg_size * limit) at 4 bytes each,
            // plus 1 array of tg_size counts at 4 bytes each.
            let per_slot_bytes = PARALLEL_TG_SIZE * limit * 4;
            let counts_bytes = PARALLEL_TG_SIZE * 4;
            unsafe {
                enc.setThreadgroupMemoryLength_atIndex(per_slot_bytes, 0); // tg_sums
                enc.setThreadgroupMemoryLength_atIndex(per_slot_bytes, 1); // tg_dates
                enc.setThreadgroupMemoryLength_atIndex(per_slot_bytes, 2); // tg_row_ids
                enc.setThreadgroupMemoryLength_atIndex(counts_bytes, 3); // tg_counts
            }
            // Dispatch exactly 1 threadgroup of PARALLEL_TG_SIZE threads.
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: PARALLEL_TG_SIZE,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: PARALLEL_TG_SIZE,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            dispatch_threads(&enc, &pipeline, 1);
        }
        enc.endEncoding();
        Self::commit_if_standalone(&cb, is_batch, "topk_dense_grouped_rows")?;

        let count = GpuBuffer {
            buffer: out_count_buf,
            len: 1,
            element_size: 4,
        };
        Ok(GpuGroupedTopKRows {
            representative_row_ids: GpuVector::inherited_with_dispatch(
                GpuBuffer {
                    buffer: out_row_ids,
                    len: limit,
                    element_size: 4,
                },
                &count,
                &out_dispatch_args,
                limit,
            ),
            aggregate_values: GpuVector::inherited_with_dispatch(
                GpuBuffer {
                    buffer: out_sums,
                    len: limit,
                    element_size: 4,
                },
                &count,
                &out_dispatch_args,
                limit,
            ),
        })
    }

    #[cfg(target_os = "macos")]
    fn dense_live_row_sums_to_topk(
        &mut self,
        live_row_ids: &GpuBuffer,
        live_count: &GpuBuffer,
        live_dispatch_args: &GpuBuffer,
        sums: &GpuBuffer,
        secondary_order_values: &GpuBuffer,
        live_capacity: usize,
        limit: usize,
        use_secondary_order: bool,
    ) -> Result<GpuGroupedTopKRows> {
        let (cb, is_batch) = self.get_command_buffer()?;
        debug_assert!(is_batch);

        let dense_secondary = allocate_buffer(
            &self.device,
            live_capacity.max(1) * 4,
            "dense grouped secondary",
        )?;
        let dense_sums =
            allocate_buffer(&self.device, live_capacity.max(1) * 4, "dense grouped sums")?;
        let gather_pipeline = self.pipeline("gather_live_grouped_rows_v")?.clone();
        let gather_enc = cb.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable(
                "failed to create dense grouped gather encoder".to_owned(),
            )
        })?;
        gather_enc.setComputePipelineState(&gather_pipeline);
        unsafe {
            gather_enc.setBuffer_offset_atIndex(Some(&live_row_ids.buffer), 0, 0);
            gather_enc.setBuffer_offset_atIndex(Some(&secondary_order_values.buffer), 0, 1);
            gather_enc.setBuffer_offset_atIndex(Some(&sums.buffer), 0, 2);
            gather_enc.setBuffer_offset_atIndex(Some(&dense_secondary), 0, 3);
            gather_enc.setBuffer_offset_atIndex(Some(&dense_sums), 0, 4);
            gather_enc.setBuffer_offset_atIndex(Some(&live_count.buffer), 0, 5);
        }
        Self::dispatch_gv_indirect(&gather_enc, live_dispatch_args);
        gather_enc.endEncoding();

        let dense_rows = GpuDenseGroupedRows {
            representative_row_ids: GpuVector::inherited_with_dispatch(
                GpuBuffer {
                    buffer: live_row_ids.buffer.clone(),
                    len: live_capacity,
                    element_size: 4,
                },
                live_count,
                live_dispatch_args,
                live_capacity,
            ),
            secondary_order_values: GpuVector::inherited_with_dispatch(
                GpuBuffer {
                    buffer: dense_secondary,
                    len: live_capacity,
                    element_size: 4,
                },
                live_count,
                live_dispatch_args,
                live_capacity,
            ),
            aggregate_values: GpuVector::inherited_with_dispatch(
                GpuBuffer {
                    buffer: dense_sums,
                    len: live_capacity,
                    element_size: 4,
                },
                live_count,
                live_dispatch_args,
                live_capacity,
            ),
        };
        self.topk_dense_grouped_rows(&dense_rows, limit, use_secondary_order)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn aggregate_topk_grouped_rows_gv(
        &mut self,
        representative_row_ids: &GpuVector,
        secondary_order_values: &GpuBuffer,
        values: &GpuVector,
        limit: usize,
        use_secondary_order: bool,
    ) -> Result<GpuGroupedTopKRows> {
        if limit == 0 || limit > 64 {
            return Err(MettleError::Processing(
                "aggregate_topk_grouped_rows_gv requires 1 <= limit <= 64".to_owned(),
            ));
        }
        debug_assert_eq!(representative_row_ids.max_len, values.max_len);
        let secondary_flag = if use_secondary_order { 1u32 } else { 0u32 };
        let row_capacity = secondary_order_values.len;
        let use_dense_row_ids = row_capacity > 0
            && row_capacity <= (1 << 23)
            && row_capacity
                <= representative_row_ids
                    .max_len
                    .saturating_mul(32)
                    .max(1 << 18);
        let started_temp_batch = if self.batch_command_buffer.is_none() {
            self.begin_batch()?;
            true
        } else {
            false
        };
        let (cb, is_batch) = self.get_command_buffer()?;
        let result = if use_dense_row_ids {
            let row_capacity_u32 = u32::try_from(row_capacity).map_err(|_| {
                MettleError::Processing("dense row-id aggregate capacity overflowed u32".to_owned())
            })?;
            let live_capacity = representative_row_ids.max_len.min(row_capacity).max(1);
            let byte_len = row_capacity * std::mem::size_of::<u32>();
            let seen = allocate_buffer(&self.device, byte_len, "dense row-id seen")?;
            let sums = allocate_buffer(&self.device, byte_len, "dense row-id sums")?;
            let live_row_ids =
                allocate_buffer(&self.device, live_capacity * 4, "dense row-id live rows")?;
            let live_count_buf = allocate_buffer(&self.device, 4, "dense row-id live count")?;
            let live_dispatch_args =
                self.allocate_dispatch_args_buffer("dense row-id live dispatch args")?;
            unsafe {
                std::ptr::write_bytes(seen.contents().as_ptr().cast::<u8>(), 0, byte_len);
                std::ptr::write_bytes(sums.contents().as_ptr().cast::<u8>(), 0, byte_len);
                std::ptr::write_bytes(live_count_buf.contents().as_ptr().cast::<u8>(), 0, 4);
            }

            let agg_pipeline = self.pipeline("dense_row_id_sum_track_live_v")?.clone();
            let agg_enc = cb.computeCommandEncoder().ok_or_else(|| {
                MettleError::MetalUnavailable(
                    "failed to create dense row-id aggregate encoder".to_owned(),
                )
            })?;
            agg_enc.setComputePipelineState(&agg_pipeline);
            unsafe {
                agg_enc.setBuffer_offset_atIndex(Some(&representative_row_ids.data.buffer), 0, 0);
                agg_enc.setBuffer_offset_atIndex(Some(&values.data.buffer), 0, 1);
                agg_enc.setBuffer_offset_atIndex(Some(&seen), 0, 2);
                agg_enc.setBuffer_offset_atIndex(Some(&sums), 0, 3);
                agg_enc.setBuffer_offset_atIndex(Some(&live_row_ids), 0, 4);
                agg_enc.setBuffer_offset_atIndex(Some(&live_count_buf), 0, 5);
                agg_enc.setBuffer_offset_atIndex(Some(&values.count.buffer), 0, 6);
                agg_enc.setBytes_length_atIndex(
                    NonNull::from(&row_capacity_u32).cast::<c_void>(),
                    4,
                    7,
                );
            }
            Self::dispatch_gv_indirect(&agg_enc, &values.dispatch_args);
            agg_enc.endEncoding();
            self.encode_dispatch_args_update_on(
                &cb,
                &GpuBuffer {
                    buffer: live_count_buf.clone(),
                    len: 1,
                    element_size: 4,
                },
                &live_dispatch_args,
            )?;
            self.dense_live_row_sums_to_topk(
                &GpuBuffer {
                    buffer: live_row_ids,
                    len: live_capacity,
                    element_size: 4,
                },
                &GpuBuffer {
                    buffer: live_count_buf,
                    len: 1,
                    element_size: 4,
                },
                &live_dispatch_args,
                &GpuBuffer {
                    buffer: sums,
                    len: row_capacity,
                    element_size: 4,
                },
                secondary_order_values,
                live_capacity,
                limit,
                secondary_flag != 0,
            )?
        } else {
            let ht_capacity =
                (representative_row_ids.max_len.saturating_mul(2).max(16)).next_power_of_two();
            let ht_capacity_u32 = u32::try_from(ht_capacity).map_err(|_| {
                MettleError::Processing("grouped aggregate capacity overflowed u32".to_owned())
            })?;
            let byte_len = ht_capacity * std::mem::size_of::<u32>();

            let ht_row_ids = allocate_buffer(&self.device, byte_len, "grouped agg row ids")?;
            let ht_secondary = allocate_buffer(&self.device, byte_len, "grouped agg secondary")?;
            let ht_sums = allocate_buffer(&self.device, byte_len, "grouped agg sums")?;
            unsafe {
                std::ptr::write_bytes(ht_row_ids.contents().as_ptr().cast::<u8>(), 0xFF, byte_len);
                std::ptr::write_bytes(ht_sums.contents().as_ptr().cast::<u8>(), 0, byte_len);
            }

            let agg_pipeline = self.pipeline("hash_agg_grouped_sum_v")?.clone();
            let agg_enc = cb.computeCommandEncoder().ok_or_else(|| {
                MettleError::MetalUnavailable(
                    "failed to create grouped aggregate encoder".to_owned(),
                )
            })?;
            agg_enc.setComputePipelineState(&agg_pipeline);
            unsafe {
                agg_enc.setBuffer_offset_atIndex(Some(&representative_row_ids.data.buffer), 0, 0);
                agg_enc.setBuffer_offset_atIndex(Some(&secondary_order_values.buffer), 0, 1);
                agg_enc.setBuffer_offset_atIndex(Some(&values.data.buffer), 0, 2);
                agg_enc.setBuffer_offset_atIndex(Some(&ht_row_ids), 0, 3);
                agg_enc.setBuffer_offset_atIndex(Some(&ht_secondary), 0, 4);
                agg_enc.setBuffer_offset_atIndex(Some(&ht_sums), 0, 5);
                agg_enc.setBuffer_offset_atIndex(Some(&values.count.buffer), 0, 6);
                agg_enc.setBytes_length_atIndex(
                    NonNull::from(&ht_capacity_u32).cast::<c_void>(),
                    4,
                    7,
                );
            }
            Self::dispatch_gv_indirect(&agg_enc, &values.dispatch_args);
            agg_enc.endEncoding();
            debug_assert!(is_batch);

            let dense_rows = self.compact_grouped_rows(
                &GpuBuffer {
                    buffer: ht_row_ids,
                    len: ht_capacity,
                    element_size: 4,
                },
                &GpuBuffer {
                    buffer: ht_secondary,
                    len: ht_capacity,
                    element_size: 4,
                },
                &GpuBuffer {
                    buffer: ht_sums,
                    len: ht_capacity,
                    element_size: 4,
                },
                ht_capacity_u32,
            )?;
            self.topk_dense_grouped_rows(&dense_rows, limit, secondary_flag != 0)?
        };
        if started_temp_batch {
            self.end_batch()?;
        }
        Ok(result)
    }

    // Non-macOS stubs for GpuVector methods
    #[cfg(not(target_os = "macos"))]
    pub(super) fn filter_to_indices_gv(&mut self, _: &GpuBuffer) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn gather_i32_gv(&mut self, _: &GpuBuffer, _: &GpuVector) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn gather_f32_gv(&mut self, _: &GpuBuffer, _: &GpuVector) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn reindex_gv(&mut self, _: &GpuVector, _: &GpuVector) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn compact_matches_gv(&mut self, _: &GpuVector) -> Result<(GpuVector, GpuVector)> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn hash_join_build_gv(
        &mut self,
        _: &GpuVector,
        _: u32,
    ) -> Result<(GpuBuffer, GpuBuffer)> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn hash_join_probe_gv(
        &mut self,
        _: &GpuVector,
        _: &GpuBuffer,
        _: &GpuBuffer,
        _: u32,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn dense_lookup_build_i32_gv(
        &mut self,
        _: &GpuVector,
        _: i32,
        _: u32,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn dense_lookup_probe_i32_gv(
        &mut self,
        _: &GpuVector,
        _: &GpuBuffer,
        _: i32,
        _: u32,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn dense_lookup_build_from_indices_i32_gv(
        &mut self,
        _: &GpuBuffer,
        _: &GpuVector,
        _: i32,
        _: u32,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn dense_lookup_probe_from_indices_i32_gv(
        &mut self,
        _: &GpuBuffer,
        _: &GpuVector,
        _: &GpuBuffer,
        _: i32,
        _: u32,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn build_membership_filter_gv(
        &mut self,
        _: &GpuVector,
        _: u32,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn membership_filter_mask_gv(
        &mut self,
        _: &GpuVector,
        _: &GpuBuffer,
        _: u32,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn project_f32_scalar_gv(
        &mut self,
        _: &GpuVector,
        _: f32,
        _: ScalarProjectOp,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn elementwise_f32_gv(
        &mut self,
        _: &GpuVector,
        _: &GpuVector,
        _: ScalarProjectOp,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn project_revenue_from_indices_gv(
        &mut self,
        _: &GpuBuffer,
        _: &GpuBuffer,
        _: &GpuVector,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn accumulate_dense_slots_gv(
        &mut self,
        _: &GpuVector,
        _: &GpuVector,
        _: &GpuVector,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn accumulate_revenue_dense_slots_from_indices_gv(
        &mut self,
        _: &GpuVector,
        _: &GpuBuffer,
        _: &GpuBuffer,
        _: &GpuVector,
        _: &GpuVector,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn grouped_aggregate_exact_gv(
        &mut self,
        _: &[GpuVector],
        _: &[GpuGroupedAggregateSpec<'_>],
    ) -> Result<GpuGroupedAggregateRows> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn compact_grouped_rows(
        &mut self,
        _: &GpuBuffer,
        _: &GpuBuffer,
        _: &GpuBuffer,
        _: u32,
    ) -> Result<GpuDenseGroupedRows> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn topk_dense_grouped_rows(
        &mut self,
        _: &GpuDenseGroupedRows,
        _: usize,
        _: bool,
    ) -> Result<GpuGroupedTopKRows> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn aggregate_topk_grouped_rows_gv(
        &mut self,
        _: &GpuVector,
        _: &GpuBuffer,
        _: &GpuVector,
        _: usize,
        _: bool,
    ) -> Result<GpuGroupedTopKRows> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn hash_join_probe_i32(
        &mut self,
        build_keys: &[i32],
        probe_keys: &[i32],
        ht_capacity: u32,
    ) -> Result<Vec<i32>> {
        if probe_keys.is_empty() {
            return Ok(Vec::new());
        }
        if ht_capacity < 2 || !ht_capacity.is_power_of_two() {
            return Err(MettleError::Processing(
                "gpu hash join capacity must be a power of two >= 2".to_owned(),
            ));
        }

        let ht_byte_len = (ht_capacity as usize) * std::mem::size_of::<u32>();
        let ht_keys_buf = allocate_buffer(&self.device, ht_byte_len, "gpu hash join keys table")?;
        let ht_idx_buf = allocate_buffer(&self.device, ht_byte_len, "gpu hash join index table")?;
        unsafe {
            std::ptr::write_bytes(
                ht_keys_buf.contents().as_ptr().cast::<u8>(),
                0xFFu8,
                ht_byte_len,
            );
            std::ptr::write_bytes(
                ht_idx_buf.contents().as_ptr().cast::<u8>(),
                0xFFu8,
                ht_byte_len,
            );
        }

        let build_pipeline = self.pipeline("hash_join_build_i32")?.clone();
        let probe_pipeline = self.pipeline("hash_join_probe_i32")?.clone();
        let probe_input =
            make_buffer_from_slice(&self.device, probe_keys, "gpu hash join probe input")?;
        let output = allocate_buffer(
            &self.device,
            std::mem::size_of_val(probe_keys),
            "gpu hash join output",
        )?;

        let command_buffer = self.command_queue.commandBuffer().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal command buffer".to_owned())
        })?;

        // Build phase.
        if !build_keys.is_empty() {
            let build_input =
                make_buffer_from_slice(&self.device, build_keys, "gpu hash join build input")?;
            let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
                MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
            })?;
            let build_len = u32::try_from(build_keys.len()).map_err(|_| {
                MettleError::Processing("hash join build len overflowed u32".to_owned())
            })?;
            encoder.setComputePipelineState(&build_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&build_input), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&ht_keys_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&ht_idx_buf), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&build_len).cast::<c_void>(),
                    std::mem::size_of::<u32>(),
                    3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::from(&ht_capacity).cast::<c_void>(),
                    std::mem::size_of::<u32>(),
                    4,
                );
            }
            dispatch_threads(&encoder, &build_pipeline, build_keys.len());
            encoder.endEncoding();
        }

        // Probe phase.
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MettleError::MetalUnavailable("failed to create a Metal compute encoder".to_owned())
        })?;
        let probe_len = u32::try_from(probe_keys.len()).map_err(|_| {
            MettleError::Processing("hash join probe len overflowed u32".to_owned())
        })?;
        encoder.setComputePipelineState(&probe_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&probe_input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&ht_keys_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&ht_idx_buf), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::from(&probe_len).cast::<c_void>(),
                std::mem::size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&ht_capacity).cast::<c_void>(),
                std::mem::size_of::<u32>(),
                5,
            );
        }
        dispatch_threads(&encoder, &probe_pipeline, probe_keys.len());
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MettleError::Processing(format!(
                "Metal hash join command buffer failed: {error}"
            )));
        }

        let result = unsafe {
            std::slice::from_raw_parts(output.contents().as_ptr().cast::<i32>(), probe_keys.len())
        };
        Ok(result.to_vec())
    }

    #[cfg(target_os = "macos")]
    fn pipeline(
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
    fn cached_input_buffer<T: Copy>(
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
    pub(super) fn get_selection_cache_gv(&mut self, key: &str) -> Option<GpuVector> {
        let vector = self
            .selection_cache
            .get(key)
            .map(|entry| entry.vector.clone())?;
        self.touch_selection_lru(key);
        Some(vector)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn put_selection_cache_gv(&mut self, key: String, vector: &GpuVector) {
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
    pub(super) fn get_join_result_cache_gv(&mut self, key: &str) -> Option<(GpuVector, GpuVector)> {
        let entry = self.join_result_cache.get(key)?;
        let representative = entry.representative_row_ids.clone();
        let aggregate = entry.aggregate_row_ids.clone();
        self.touch_join_result_lru(key);
        Some((representative, aggregate))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn put_join_result_cache_gv(
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
    pub(super) fn get_selection_cache_gv(&mut self, _: &str) -> Option<GpuVector> {
        None
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn put_selection_cache_gv(&mut self, _: String, _: &GpuVector) {}

    #[cfg(not(target_os = "macos"))]
    pub(super) fn get_join_result_cache_gv(&mut self, _: &str) -> Option<(GpuVector, GpuVector)> {
        None
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn put_join_result_cache_gv(&mut self, _: String, _: &GpuVector, _: &GpuVector) {}

    #[cfg(not(target_os = "macos"))]
    pub(super) fn project_f32_values(
        &mut self,
        _values: &[f32],
        _scalar: f32,
        _op: ScalarProjectOp,
    ) -> Result<Vec<f32>> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn compare_i32_values(
        &mut self,
        _values: &[i32],
        _scalar: i32,
        _op: ScalarCompareOp,
    ) -> Result<Vec<bool>> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn compare_date32_values(
        &mut self,
        _values: &[i32],
        _scalar: i32,
        _op: ScalarCompareOp,
    ) -> Result<Vec<bool>> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn compare_i64_values(
        &mut self,
        _values: &[i64],
        _scalar: i64,
        _op: ScalarCompareOp,
    ) -> Result<Vec<bool>> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn compare_f32_values(
        &mut self,
        _values: &[f32],
        _scalar: f32,
        _op: ScalarCompareOp,
    ) -> Result<Vec<bool>> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn compare_i32_to_gpu(
        &mut self,
        _: &[i32],
        _: i32,
        _: ScalarCompareOp,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn compare_i64_to_gpu(
        &mut self,
        _: &[i64],
        _: i64,
        _: ScalarCompareOp,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn compare_f32_to_gpu(
        &mut self,
        _: &[f32],
        _: f32,
        _: ScalarCompareOp,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn and_masks_gpu(&mut self, _: &GpuBuffer, _: &GpuBuffer) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(super) fn filter_to_indices_gpu(&mut self, _: &GpuBuffer) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn elementwise_f64(
        &mut self,
        _a: &[f64],
        _b: &[f64],
        _op: ScalarProjectOp,
    ) -> Result<Vec<f64>> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }
}

#[cfg(target_os = "macos")]
fn encode_f32_scalar_projection(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    scalar: f32,
    len: usize,
    op: ScalarProjectOp,
) -> Result<()> {
    let len_u32 = u32::try_from(len)
        .map_err(|_| MettleError::Processing("projection length overflowed u32".to_owned()))?;
    let op_code = op.code();

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::from(&scalar).cast::<c_void>(),
            std::mem::size_of::<f32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&len_u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&op_code).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    dispatch_threads(encoder, pipeline, len);
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_i32_scalar_compare(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    scalar: i32,
    len: usize,
    op: ScalarCompareOp,
) -> Result<()> {
    let len_u32 = u32::try_from(len)
        .map_err(|_| MettleError::Processing("comparison length overflowed u32".to_owned()))?;
    let op_code = op.code();

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::from(&scalar).cast::<c_void>(),
            std::mem::size_of::<i32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&len_u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&op_code).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    dispatch_threads(encoder, pipeline, len);
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_i64_scalar_compare(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    scalar: i64,
    len: usize,
    op: ScalarCompareOp,
) -> Result<()> {
    let len_u32 = u32::try_from(len)
        .map_err(|_| MettleError::Processing("comparison length overflowed u32".to_owned()))?;
    let op_code = op.code();

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::from(&scalar).cast::<c_void>(),
            std::mem::size_of::<i64>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&len_u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&op_code).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    dispatch_threads(encoder, pipeline, len);
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_f32_scalar_compare(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    scalar: f32,
    len: usize,
    op: ScalarCompareOp,
) -> Result<()> {
    let len_u32 = u32::try_from(len)
        .map_err(|_| MettleError::Processing("comparison length overflowed u32".to_owned()))?;
    let op_code = op.code();

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::from(&scalar).cast::<c_void>(),
            std::mem::size_of::<f32>(),
            2,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&len_u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&op_code).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    dispatch_threads(encoder, pipeline, len);
    Ok(())
}
