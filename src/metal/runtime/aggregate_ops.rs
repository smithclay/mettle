use super::MetalRuntime;
use crate::error::{MettleError, Result};
use crate::metal::gpu_buffer::{GpuBuffer, GpuVector};
use crate::metal::{
    GpuDenseGroupedRows, GpuGroupedAggregateRows, GpuGroupedAggregateSlotResult,
    GpuGroupedAggregateSpec, GpuGroupedAggregateStateKind, GpuGroupedTopKRows,
};

#[cfg(target_os = "macos")]
use super::{GV_THREADS_PER_GROUP, MAX_GROUP_KEY_LANES};
#[cfg(target_os = "macos")]
use crate::metal::gpu_buffer::{allocate_buffer, make_buffer_from_slice};
#[cfg(target_os = "macos")]
use crate::metal::shaders::dispatch_threads;

#[cfg(target_os = "macos")]
use std::ffi::c_void;
#[cfg(target_os = "macos")]
use std::ptr::NonNull;

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};

impl MetalRuntime {
    #[cfg(target_os = "macos")]
    pub(in crate::metal) fn accumulate_dense_slots_gv(
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
    pub(in crate::metal) fn accumulate_revenue_dense_slots_from_indices_gv(
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
    pub(in crate::metal) fn grouped_aggregate_exact_gv(
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
    pub(in crate::metal) fn compact_grouped_rows(
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
    pub(in crate::metal) fn topk_dense_grouped_rows(
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
    pub(in crate::metal) fn aggregate_topk_grouped_rows_gv(
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

    // Non-macOS stubs
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn accumulate_dense_slots_gv(
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
    pub(in crate::metal) fn accumulate_revenue_dense_slots_from_indices_gv(
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
    pub(in crate::metal) fn grouped_aggregate_exact_gv(
        &mut self,
        _: &[GpuVector],
        _: &[GpuGroupedAggregateSpec<'_>],
    ) -> Result<GpuGroupedAggregateRows> {
        Err(MettleError::MetalUnavailable(
            "Metal is only available on Apple platforms".into(),
        ))
    }
    #[cfg(not(target_os = "macos"))]
    pub(in crate::metal) fn compact_grouped_rows(
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
    pub(in crate::metal) fn topk_dense_grouped_rows(
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
    pub(in crate::metal) fn aggregate_topk_grouped_rows_gv(
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
}
