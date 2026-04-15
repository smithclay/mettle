use super::MetalRuntime;
use crate::error::{MettleError, Result};
use crate::metal::gpu_buffer::GpuBuffer;
use crate::metal::gpu_buffer::GpuVector;

#[cfg(target_os = "macos")]
use crate::metal::gpu_buffer::allocate_buffer;
#[cfg(target_os = "macos")]
use crate::metal::shaders::dispatch_threads;

#[cfg(target_os = "macos")]
use std::ffi::c_void;
#[cfg(target_os = "macos")]
use std::ptr::NonNull;

#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

impl MetalRuntime {
    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn filter_to_indices_gpu(&mut self, mask: &GpuBuffer) -> Result<GpuBuffer> {
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
    pub(in crate::metal) fn filter_to_indices_gv(&mut self, mask: &GpuBuffer) -> Result<GpuVector> {
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
    pub(in crate::metal) fn gather_i32_gv(
        &mut self,
        data: &GpuBuffer,
        indices: &GpuVector,
    ) -> Result<GpuVector> {
        self.gather_gv_impl("gather_i32_v", data, indices)
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn gather_f32_gv(
        &mut self,
        data: &GpuBuffer,
        indices: &GpuVector,
    ) -> Result<GpuVector> {
        self.gather_gv_impl("gather_f32_v", data, indices)
    }

    /// Gather i32 data as f32 (cast during gather). For integer columns in expressions.
    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn gather_i32_as_f32_gv(
        &mut self,
        data: &GpuBuffer,
        indices: &GpuVector,
    ) -> Result<GpuVector> {
        self.gather_gv_impl("gather_i32_as_f32_v", data, indices)
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn gather_i32_as_f32_gv(
        &mut self,
        _: &GpuBuffer,
        _: &GpuVector,
    ) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn reindex_gv(
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
    pub(in crate::metal) fn compact_matches_gv(
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

    // Non-macOS stubs
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn filter_to_indices_gpu(&mut self, _: &GpuBuffer) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".to_owned(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn filter_to_indices_gv(&mut self, _: &GpuBuffer) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn gather_i32_gv(&mut self, _: &GpuBuffer, _: &GpuVector) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn gather_f32_gv(&mut self, _: &GpuBuffer, _: &GpuVector) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn reindex_gv(&mut self, _: &GpuVector, _: &GpuVector) -> Result<GpuVector> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn compact_matches_gv(&mut self, _: &GpuVector) -> Result<(GpuVector, GpuVector)> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
}
