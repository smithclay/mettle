use super::MetalRuntime;
use crate::error::{MettleError, Result};
use crate::metal::gpu_buffer::{GpuBuffer, GpuVector};

#[cfg(target_os = "macos")]
use crate::metal::gpu_buffer::{allocate_buffer, make_buffer_from_slice};
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
    pub(in crate::metal) fn hash_join_build_gv(
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
    pub(in crate::metal) fn hash_join_probe_gv(
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
    pub(in crate::metal) fn dense_lookup_build_i32_gv(
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
    pub(in crate::metal) fn dense_lookup_probe_i32_gv(
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
    pub(in crate::metal) fn dense_lookup_build_from_indices_i32_gv(
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
    pub(in crate::metal) fn dense_lookup_probe_from_indices_i32_gv(
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
    pub(in crate::metal) fn build_membership_filter_gv(
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
    pub(in crate::metal) fn membership_filter_mask_gv(
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
    pub(in crate::metal) fn hash_join_probe_i32(
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

    // Non-macOS stubs
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn hash_join_build_gv(
        &mut self,
        _: &GpuVector,
        _: u32,
    ) -> Result<(GpuBuffer, GpuBuffer)> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn hash_join_probe_gv(
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
    pub(in crate::metal) fn dense_lookup_build_i32_gv(
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
    pub(in crate::metal) fn dense_lookup_probe_i32_gv(
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
    pub(in crate::metal) fn dense_lookup_build_from_indices_i32_gv(
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
    pub(in crate::metal) fn dense_lookup_probe_from_indices_i32_gv(
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
    pub(in crate::metal) fn build_membership_filter_gv(
        &mut self,
        _: &GpuVector,
        _: u32,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn membership_filter_mask_gv(
        &mut self,
        _: &GpuVector,
        _: &GpuBuffer,
        _: u32,
    ) -> Result<GpuBuffer> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }

}
