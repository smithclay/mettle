use super::MetalRuntime;
use crate::error::{MettleError, Result};

#[cfg(target_os = "macos")]
use super::{CACHE_TAG_F32, CACHE_TAG_I32, CACHE_TAG_I64, CACHE_TAG_DATE32};
#[cfg(target_os = "macos")]
use crate::metal::gpu_buffer::{allocate_buffer, make_buffer_from_slice, GpuBuffer, GpuVector};
#[cfg(target_os = "macos")]
use crate::metal::shaders::dispatch_threads;
#[cfg(target_os = "macos")]
use crate::metal::{ScalarCompareOp, ScalarProjectOp};
#[cfg(not(target_os = "macos"))]
use crate::metal::gpu_buffer::{GpuBuffer, GpuVector};
#[cfg(not(target_os = "macos"))]
use crate::metal::{ScalarCompareOp, ScalarProjectOp};

#[cfg(target_os = "macos")]
use std::ffi::c_void;
#[cfg(target_os = "macos")]
use std::ptr::NonNull;

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState,
};

impl MetalRuntime {
    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn project_f32_values(
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
    pub(in crate::metal) fn compare_i32_values(
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
    pub(in crate::metal) fn compare_date32_values(
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
    pub(in crate::metal) fn compare_i64_values(
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
    pub(in crate::metal) fn compare_f32_values(
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
    pub(in crate::metal) fn compare_i32_to_gpu(
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
    pub(in crate::metal) fn compare_i64_to_gpu(
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
    pub(in crate::metal) fn compare_f32_to_gpu(
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
    pub(in crate::metal) fn and_masks_gpu(&mut self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
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
    pub(in crate::metal) fn elementwise_f64(
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

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn project_f32_scalar_gv(
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
    pub(in crate::metal) fn elementwise_f32_gv(
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
    pub(in crate::metal) fn project_revenue_from_indices_gv(
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

    // Non-macOS stubs
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn project_f32_values(
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
    pub(in crate::metal) fn compare_i32_values(
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
    pub(in crate::metal) fn compare_date32_values(
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
    pub(in crate::metal) fn compare_i64_values(
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
    pub(in crate::metal) fn compare_f32_values(
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
    pub(in crate::metal) fn compare_i32_to_gpu(
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
    pub(in crate::metal) fn compare_i64_to_gpu(
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
    pub(in crate::metal) fn compare_f32_to_gpu(
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
    pub(in crate::metal) fn and_masks_gpu(&mut self, _: &GpuBuffer, _: &GpuBuffer) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn elementwise_f64(
        &mut self,
        _a: &[f64],
        _b: &[f64],
        _op: ScalarProjectOp,
    ) -> Result<Vec<f64>> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn project_f32_scalar_gv(
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
    pub(in crate::metal) fn elementwise_f32_gv(
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
    pub(in crate::metal) fn project_revenue_from_indices_gv(
        &mut self,
        _: &GpuBuffer,
        _: &GpuBuffer,
        _: &GpuVector,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
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
