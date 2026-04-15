use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::size_of_val;

use crate::error::{MettleError, Result};

use super::{ScalarCompareOp, ScalarProjectOp};

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_foundation::NSString;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLSize,
};

#[cfg(target_os = "macos")]
pub(super) fn hash_values<T: Copy>(values: &[T]) -> u64 {
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) };
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

#[cfg(target_os = "macos")]
pub(super) fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let source = NSString::from_str(METAL_KERNEL_SOURCE);
    let options = MTLCompileOptions::new();
    device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| {
            MettleError::Processing(format!("failed to compile Metal library: {error}"))
        })
}

#[cfg(target_os = "macos")]
pub(super) fn compile_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    function_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let function_name = NSString::from_str(function_name);
    let function = library.newFunctionWithName(&function_name).ok_or_else(|| {
        MettleError::Processing(format!(
            "Metal function '{function_name}' was not found in the compiled library"
        ))
    })?;

    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| {
            MettleError::Processing(format!(
                "failed to compile Metal compute pipeline '{function_name}': {error}"
            ))
        })
}
#[cfg(target_os = "macos")]
pub(super) fn dispatch_threads(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    len: usize,
) {
    // Use `maxTotalThreadsPerThreadgroup` (typically 1024 on M-series) as the
    // threadgroup width rather than `threadExecutionWidth` (the 32-thread SIMD
    // width) — the latter ran every kernel at 1/32 occupancy. Capped to `len`
    // for small inputs so we don't launch empty lanes.
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup().max(1);
    let threads_per_group = max_threads.min(len.max(1));
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: len,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
}

impl ScalarProjectOp {
    pub(super) fn code(self) -> u32 {
        match self {
            Self::Add => 0,
            Self::Multiply => 1,
            Self::Subtract => 2,
            Self::SubtractFrom => 3,
        }
    }
}

impl ScalarCompareOp {
    pub(super) fn code(self) -> u32 {
        match self {
            Self::Eq => 0,
            Self::Lt => 1,
            Self::Le => 2,
            Self::Gt => 3,
            Self::Ge => 4,
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) const METAL_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void project_i32_scalar(
    device const int* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant int& scalar [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }

    int value = input[gid];
    switch (op_code) {
        case 0: output[gid] = value + scalar; break;
        case 1: output[gid] = value * scalar; break;
        case 2: output[gid] = value - scalar; break;
        case 3: output[gid] = scalar - value; break;
        default: output[gid] = value; break;
    }
}

kernel void project_i64_scalar(
    device const long* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant long& scalar [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }

    long value = input[gid];
    switch (op_code) {
        case 0: output[gid] = value + scalar; break;
        case 1: output[gid] = value * scalar; break;
        case 2: output[gid] = value - scalar; break;
        case 3: output[gid] = scalar - value; break;
        default: output[gid] = value; break;
    }
}

kernel void project_f32_scalar(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }

    float value = input[gid];
    switch (op_code) {
        case 0: output[gid] = value + scalar; break;
        case 1: output[gid] = value * scalar; break;
        case 2: output[gid] = value - scalar; break;
        case 3: output[gid] = scalar - value; break;
        default: output[gid] = value; break;
    }
}

kernel void compare_i32_scalar(
    device const int* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant int& scalar [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }

    int value = input[gid];
    switch (op_code) {
        case 0:
            output[gid] = value == scalar;
            break;
        case 1:
            output[gid] = value < scalar;
            break;
        case 2:
            output[gid] = value <= scalar;
            break;
        case 3:
            output[gid] = value > scalar;
            break;
        case 4:
            output[gid] = value >= scalar;
            break;
        default:
            output[gid] = 0;
            break;
    }
}

kernel void compare_i64_scalar(
    device const long* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant long& scalar [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }

    long value = input[gid];
    switch (op_code) {
        case 0:
            output[gid] = value == scalar;
            break;
        case 1:
            output[gid] = value < scalar;
            break;
        case 2:
            output[gid] = value <= scalar;
            break;
        case 3:
            output[gid] = value > scalar;
            break;
        case 4:
            output[gid] = value >= scalar;
            break;
        default:
            output[gid] = 0;
            break;
    }
}

kernel void compare_f32_scalar(
    device const float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }

    float value = input[gid];
    switch (op_code) {
        case 0:
            output[gid] = value == scalar;
            break;
        case 1:
            output[gid] = value < scalar;
            break;
        case 2:
            output[gid] = value <= scalar;
            break;
        case 3:
            output[gid] = value > scalar;
            break;
        case 4:
            output[gid] = value >= scalar;
            break;
        default:
            output[gid] = 0;
            break;
    }
}

kernel void hash_join_build_i32(
    device const int* keys [[buffer(0)]],
    device atomic_uint* ht_keys [[buffer(1)]],
    device atomic_int* ht_row_indices [[buffer(2)]],
    constant uint& input_len [[buffer(3)]],
    constant uint& ht_capacity [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= input_len) {
        return;
    }

    uint key = uint(keys[gid]);
    int row_idx = int(gid);
    uint mask = ht_capacity - 1u;
    uint slot = key & mask;

    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint expected = 0xFFFFFFFFu;
        bool claimed = atomic_compare_exchange_weak_explicit(
            &ht_keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed
        );
        if (claimed) {
            atomic_store_explicit(&ht_row_indices[slot], row_idx, memory_order_relaxed);
            return;
        }
        if (expected == key) {
            // Duplicate key — overwrite (PK-FK assumption: keys are unique on build side).
            atomic_store_explicit(&ht_row_indices[slot], row_idx, memory_order_relaxed);
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

kernel void hash_join_probe_i32(
    device const int* probe_keys [[buffer(0)]],
    device int* output [[buffer(1)]],
    device const uint* ht_keys [[buffer(2)]],
    device const int* ht_row_indices [[buffer(3)]],
    constant uint& probe_len [[buffer(4)]],
    constant uint& ht_capacity [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= probe_len) {
        return;
    }

    uint key = uint(probe_keys[gid]);
    uint mask = ht_capacity - 1u;
    uint slot = key & mask;

    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint bucket_key = ht_keys[slot];
        if (bucket_key == 0xFFFFFFFFu) {
            output[gid] = -1;
            return;
        }
        if (bucket_key == key) {
            output[gid] = ht_row_indices[slot];
            return;
        }
        slot = (slot + 1u) & mask;
    }
    output[gid] = -1;
}

// Atomic float add via compare-and-swap (works on all Metal versions).
void atomic_add_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = current + val;
        uint desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

void atomic_min_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = min(current, val);
        uint desired_bits = as_type<uint>(desired);
        if (desired_bits == expected) {
            return;
        }
        if (atomic_compare_exchange_weak_explicit(
                addr, &expected, desired_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

void atomic_max_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = max(current, val);
        uint desired_bits = as_type<uint>(desired);
        if (desired_bits == expected) {
            return;
        }
        if (atomic_compare_exchange_weak_explicit(
                addr, &expected, desired_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

inline uint grouped_key_hash(uint key0, uint key1, uint key2, uint key3, uint key_lane_count) {
    uint h = 2166136261u;
    h = (h ^ key0) * 16777619u;
    if (key_lane_count > 1u) h = (h ^ key1) * 16777619u;
    if (key_lane_count > 2u) h = (h ^ key2) * 16777619u;
    if (key_lane_count > 3u) h = (h ^ key3) * 16777619u;
    return h;
}

inline bool grouped_keys_equal(
    device const uint* ht_key0,
    device const uint* ht_key1,
    device const uint* ht_key2,
    device const uint* ht_key3,
    uint slot,
    uint key0,
    uint key1,
    uint key2,
    uint key3,
    uint key_lane_count
) {
    if (ht_key0[slot] != key0) return false;
    if (key_lane_count > 1u && ht_key1[slot] != key1) return false;
    if (key_lane_count > 2u && ht_key2[slot] != key2) return false;
    if (key_lane_count > 3u && ht_key3[slot] != key3) return false;
    return true;
}

inline uint probe_grouped_slot(
    device atomic_uint* ht_states,
    device const uint* ht_key0,
    device const uint* ht_key1,
    device const uint* ht_key2,
    device const uint* ht_key3,
    uint key0,
    uint key1,
    uint key2,
    uint key3,
    uint ht_capacity,
    uint key_lane_count
) {
    uint mask = ht_capacity - 1u;
    uint slot = grouped_key_hash(key0, key1, key2, key3, key_lane_count) & mask;
    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint state = atomic_load_explicit(&ht_states[slot], memory_order_relaxed);
        while (state == 1u) {
            state = atomic_load_explicit(&ht_states[slot], memory_order_relaxed);
        }
        if (state == 0u) {
            return 0xFFFFFFFFu;
        }
        if (grouped_keys_equal(
                ht_key0, ht_key1, ht_key2, ht_key3,
                slot, key0, key1, key2, key3, key_lane_count)) {
            return slot;
        }
        slot = (slot + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// GPU grouped aggregate: for each (key, value) pair, find-or-insert the key
// in an open-addressing hash table, then atomically add the value to the
// accumulator slot. Keys are i32 (cast to uint), values are f32.
kernel void hash_agg_sum_f32(
    device const int* keys [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device atomic_uint* ht_keys [[buffer(2)]],
    device atomic_uint* ht_sums [[buffer(3)]],
    constant uint& input_len [[buffer(4)]],
    constant uint& ht_capacity [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= input_len) {
        return;
    }

    uint key = uint(keys[gid]);
    float val = values[gid];
    uint mask = ht_capacity - 1u;
    uint slot = key & mask;

    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint expected = 0xFFFFFFFFu;
        bool claimed = atomic_compare_exchange_weak_explicit(
            &ht_keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed
        );
        if (claimed || expected == key) {
            atomic_add_float(&ht_sums[slot], val);
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

kernel void hash_agg_single_key_sum_v(
    device const int* keys [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device atomic_uint* ht_keys [[buffer(2)]],
    device atomic_uint* ht_sums [[buffer(3)]],
    device const uint* count [[buffer(4)]],
    constant uint& ht_capacity [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;

    uint key = as_type<uint>(keys[gid]);
    float val = values[gid];
    uint mask = ht_capacity - 1u;
    uint slot = (key * 2654435761u) & mask;

    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint expected = 0xFFFFFFFFu;
        bool claimed = atomic_compare_exchange_weak_explicit(
            &ht_keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed
        );
        if (claimed || expected == key) {
            atomic_add_float(&ht_sums[slot], val);
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

kernel void grouped_aggregate_build_keys_v(
    device const int* key0_in [[buffer(0)]],
    device const int* key1_in [[buffer(1)]],
    device const int* key2_in [[buffer(2)]],
    device const int* key3_in [[buffer(3)]],
    device atomic_uint* ht_states [[buffer(4)]],
    device uint* ht_key0 [[buffer(5)]],
    device uint* ht_key1 [[buffer(6)]],
    device uint* ht_key2 [[buffer(7)]],
    device uint* ht_key3 [[buffer(8)]],
    device const uint* count [[buffer(9)]],
    constant uint& ht_capacity [[buffer(10)]],
    constant uint& key_lane_count [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;

    uint key0 = as_type<uint>(key0_in[gid]);
    uint key1 = key_lane_count > 1u ? as_type<uint>(key1_in[gid]) : 0u;
    uint key2 = key_lane_count > 2u ? as_type<uint>(key2_in[gid]) : 0u;
    uint key3 = key_lane_count > 3u ? as_type<uint>(key3_in[gid]) : 0u;

    uint mask = ht_capacity - 1u;
    uint slot = grouped_key_hash(key0, key1, key2, key3, key_lane_count) & mask;

    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint state = atomic_load_explicit(&ht_states[slot], memory_order_relaxed);
        while (state == 1u) {
            state = atomic_load_explicit(&ht_states[slot], memory_order_relaxed);
        }
        if (state == 2u) {
            if (grouped_keys_equal(ht_key0, ht_key1, ht_key2, ht_key3, slot, key0, key1, key2, key3, key_lane_count)) {
                return;
            }
            slot = (slot + 1u) & mask;
            continue;
        }

        uint expected = 0u;
        bool claimed = atomic_compare_exchange_weak_explicit(
            &ht_states[slot], &expected, 1u,
            memory_order_relaxed, memory_order_relaxed
        );
        if (claimed) {
            ht_key0[slot] = key0;
            if (key_lane_count > 1u) ht_key1[slot] = key1;
            if (key_lane_count > 2u) ht_key2[slot] = key2;
            if (key_lane_count > 3u) ht_key3[slot] = key3;
            atomic_store_explicit(&ht_states[slot], 2u, memory_order_relaxed);
            return;
        }

        state = expected;
        while (state == 1u) {
            state = atomic_load_explicit(&ht_states[slot], memory_order_relaxed);
        }
        if (state == 2u &&
            grouped_keys_equal(ht_key0, ht_key1, ht_key2, ht_key3, slot, key0, key1, key2, key3, key_lane_count)) {
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

kernel void grouped_aggregate_update_count_v(
    device const int* key0_in [[buffer(0)]],
    device const int* key1_in [[buffer(1)]],
    device const int* key2_in [[buffer(2)]],
    device const int* key3_in [[buffer(3)]],
    device atomic_uint* ht_states [[buffer(4)]],
    device const uint* ht_key0 [[buffer(5)]],
    device const uint* ht_key1 [[buffer(6)]],
    device const uint* ht_key2 [[buffer(7)]],
    device const uint* ht_key3 [[buffer(8)]],
    device atomic_uint* counts [[buffer(9)]],
    device const uint* count [[buffer(10)]],
    constant uint& ht_capacity [[buffer(11)]],
    constant uint& key_lane_count [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    uint key0 = as_type<uint>(key0_in[gid]);
    uint key1 = key_lane_count > 1u ? as_type<uint>(key1_in[gid]) : 0u;
    uint key2 = key_lane_count > 2u ? as_type<uint>(key2_in[gid]) : 0u;
    uint key3 = key_lane_count > 3u ? as_type<uint>(key3_in[gid]) : 0u;
    uint slot = probe_grouped_slot(ht_states, ht_key0, ht_key1, ht_key2, ht_key3, key0, key1, key2, key3, ht_capacity, key_lane_count);
    if (slot == 0xFFFFFFFFu) return;
    atomic_fetch_add_explicit(&counts[slot], 1u, memory_order_relaxed);
}

kernel void grouped_aggregate_update_sum_v(
    device const int* key0_in [[buffer(0)]],
    device const int* key1_in [[buffer(1)]],
    device const int* key2_in [[buffer(2)]],
    device const int* key3_in [[buffer(3)]],
    device const float* values [[buffer(4)]],
    device atomic_uint* ht_states [[buffer(5)]],
    device const uint* ht_key0 [[buffer(6)]],
    device const uint* ht_key1 [[buffer(7)]],
    device const uint* ht_key2 [[buffer(8)]],
    device const uint* ht_key3 [[buffer(9)]],
    device atomic_uint* sums [[buffer(10)]],
    device atomic_uint* unused_counts [[buffer(11)]],
    device const uint* count [[buffer(12)]],
    constant uint& ht_capacity [[buffer(13)]],
    constant uint& key_lane_count [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    (void)unused_counts;
    uint key0 = as_type<uint>(key0_in[gid]);
    uint key1 = key_lane_count > 1u ? as_type<uint>(key1_in[gid]) : 0u;
    uint key2 = key_lane_count > 2u ? as_type<uint>(key2_in[gid]) : 0u;
    uint key3 = key_lane_count > 3u ? as_type<uint>(key3_in[gid]) : 0u;
    uint slot = probe_grouped_slot(ht_states, ht_key0, ht_key1, ht_key2, ht_key3, key0, key1, key2, key3, ht_capacity, key_lane_count);
    if (slot == 0xFFFFFFFFu) return;
    atomic_add_float(&sums[slot], values[gid]);
}

kernel void grouped_aggregate_update_min_v(
    device const int* key0_in [[buffer(0)]],
    device const int* key1_in [[buffer(1)]],
    device const int* key2_in [[buffer(2)]],
    device const int* key3_in [[buffer(3)]],
    device const float* values [[buffer(4)]],
    device atomic_uint* ht_states [[buffer(5)]],
    device const uint* ht_key0 [[buffer(6)]],
    device const uint* ht_key1 [[buffer(7)]],
    device const uint* ht_key2 [[buffer(8)]],
    device const uint* ht_key3 [[buffer(9)]],
    device atomic_uint* mins [[buffer(10)]],
    device atomic_uint* unused_counts [[buffer(11)]],
    device const uint* count [[buffer(12)]],
    constant uint& ht_capacity [[buffer(13)]],
    constant uint& key_lane_count [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    (void)unused_counts;
    uint key0 = as_type<uint>(key0_in[gid]);
    uint key1 = key_lane_count > 1u ? as_type<uint>(key1_in[gid]) : 0u;
    uint key2 = key_lane_count > 2u ? as_type<uint>(key2_in[gid]) : 0u;
    uint key3 = key_lane_count > 3u ? as_type<uint>(key3_in[gid]) : 0u;
    uint slot = probe_grouped_slot(ht_states, ht_key0, ht_key1, ht_key2, ht_key3, key0, key1, key2, key3, ht_capacity, key_lane_count);
    if (slot == 0xFFFFFFFFu) return;
    atomic_min_float(&mins[slot], values[gid]);
}

kernel void grouped_aggregate_update_max_v(
    device const int* key0_in [[buffer(0)]],
    device const int* key1_in [[buffer(1)]],
    device const int* key2_in [[buffer(2)]],
    device const int* key3_in [[buffer(3)]],
    device const float* values [[buffer(4)]],
    device atomic_uint* ht_states [[buffer(5)]],
    device const uint* ht_key0 [[buffer(6)]],
    device const uint* ht_key1 [[buffer(7)]],
    device const uint* ht_key2 [[buffer(8)]],
    device const uint* ht_key3 [[buffer(9)]],
    device atomic_uint* maxs [[buffer(10)]],
    device atomic_uint* unused_counts [[buffer(11)]],
    device const uint* count [[buffer(12)]],
    constant uint& ht_capacity [[buffer(13)]],
    constant uint& key_lane_count [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    (void)unused_counts;
    uint key0 = as_type<uint>(key0_in[gid]);
    uint key1 = key_lane_count > 1u ? as_type<uint>(key1_in[gid]) : 0u;
    uint key2 = key_lane_count > 2u ? as_type<uint>(key2_in[gid]) : 0u;
    uint key3 = key_lane_count > 3u ? as_type<uint>(key3_in[gid]) : 0u;
    uint slot = probe_grouped_slot(ht_states, ht_key0, ht_key1, ht_key2, ht_key3, key0, key1, key2, key3, ht_capacity, key_lane_count);
    if (slot == 0xFFFFFFFFu) return;
    atomic_max_float(&maxs[slot], values[gid]);
}

kernel void grouped_aggregate_update_avg_v(
    device const int* key0_in [[buffer(0)]],
    device const int* key1_in [[buffer(1)]],
    device const int* key2_in [[buffer(2)]],
    device const int* key3_in [[buffer(3)]],
    device const float* values [[buffer(4)]],
    device atomic_uint* ht_states [[buffer(5)]],
    device const uint* ht_key0 [[buffer(6)]],
    device const uint* ht_key1 [[buffer(7)]],
    device const uint* ht_key2 [[buffer(8)]],
    device const uint* ht_key3 [[buffer(9)]],
    device atomic_uint* sums [[buffer(10)]],
    device atomic_uint* counts [[buffer(11)]],
    device const uint* count [[buffer(12)]],
    constant uint& ht_capacity [[buffer(13)]],
    constant uint& key_lane_count [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    uint key0 = as_type<uint>(key0_in[gid]);
    uint key1 = key_lane_count > 1u ? as_type<uint>(key1_in[gid]) : 0u;
    uint key2 = key_lane_count > 2u ? as_type<uint>(key2_in[gid]) : 0u;
    uint key3 = key_lane_count > 3u ? as_type<uint>(key3_in[gid]) : 0u;
    uint slot = probe_grouped_slot(ht_states, ht_key0, ht_key1, ht_key2, ht_key3, key0, key1, key2, key3, ht_capacity, key_lane_count);
    if (slot == 0xFFFFFFFFu) return;
    atomic_add_float(&sums[slot], values[gid]);
    atomic_fetch_add_explicit(&counts[slot], 1u, memory_order_relaxed);
}

kernel void gather_i32(
    device const int* data [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device int* output [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    output[gid] = data[indices[gid]];
}

kernel void gather_f32(
    device const float* data [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    output[gid] = data[indices[gid]];
}

kernel void reindex(
    device const uint* base [[buffer(0)]],
    device const uint* selection [[buffer(1)]],
    device uint* output [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    output[gid] = base[selection[gid]];
}

kernel void compact_matches(
    device const int* matches [[buffer(0)]],
    device uint* probe_kept [[buffer(1)]],
    device uint* build_kept [[buffer(2)]],
    device atomic_uint* count [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    int match_idx = matches[gid];
    if (match_idx >= 0) {
        uint slot = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        probe_kept[slot] = gid;
        build_kept[slot] = uint(match_idx);
    }
}

kernel void and_masks(
    device const uchar* a [[buffer(0)]],
    device const uchar* b [[buffer(1)]],
    device uchar* output [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    output[gid] = a[gid] & b[gid];
}

kernel void filter_to_indices(
    device const uchar* mask [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device atomic_uint* count [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    if (mask[gid] != 0) {
        uint slot = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        output[slot] = gid;
    }
}

kernel void elementwise_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }

    float va = a[gid];
    float vb = b[gid];
    switch (op_code) {
        case 0: output[gid] = va + vb; break;
        case 1: output[gid] = va * vb; break;
        case 2: output[gid] = va - vb; break;
        default: output[gid] = va; break;
    }
}

// ── Count-passing _v variants ─────────────────────────────────────────
// These read their length from a GPU-resident count buffer instead of a
// CPU-provided constant. Used by the GPU-native pipeline to avoid
// intermediate waitUntilCompleted stalls.

kernel void gather_i32_as_f32_v(
    device const int* data [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    output[gid] = float(data[indices[gid]]);
}

kernel void gather_i32_v(
    device const int* data [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device int* output [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    output[gid] = data[indices[gid]];
}

kernel void gather_f32_v(
    device const float* data [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    output[gid] = data[indices[gid]];
}

kernel void reindex_v(
    device const uint* base [[buffer(0)]],
    device const uint* selection [[buffer(1)]],
    device uint* output [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    output[gid] = base[selection[gid]];
}

kernel void compact_matches_v(
    device const int* matches [[buffer(0)]],
    device uint* probe_kept [[buffer(1)]],
    device uint* build_kept [[buffer(2)]],
    device atomic_uint* out_count [[buffer(3)]],
    device const uint* in_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= in_count[0]) return;
    int match_idx = matches[gid];
    if (match_idx >= 0) {
        uint slot = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
        probe_kept[slot] = gid;
        build_kept[slot] = uint(match_idx);
    }
}

kernel void hash_join_build_i32_v(
    device const int* keys [[buffer(0)]],
    device atomic_uint* ht_keys [[buffer(1)]],
    device atomic_int* ht_row_indices [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    constant uint& ht_capacity [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    uint key = uint(keys[gid]);
    int row_idx = int(gid);
    uint mask = ht_capacity - 1u;
    uint slot = key & mask;
    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint expected = 0xFFFFFFFFu;
        bool claimed = atomic_compare_exchange_weak_explicit(
            &ht_keys[slot], &expected, key,
            memory_order_relaxed, memory_order_relaxed
        );
        if (claimed) {
            atomic_store_explicit(&ht_row_indices[slot], row_idx, memory_order_relaxed);
            return;
        }
        if (expected == key) {
            atomic_store_explicit(&ht_row_indices[slot], row_idx, memory_order_relaxed);
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

kernel void hash_join_probe_i32_v(
    device const int* probe_keys [[buffer(0)]],
    device int* output [[buffer(1)]],
    device const uint* ht_keys [[buffer(2)]],
    device const int* ht_row_indices [[buffer(3)]],
    device const uint* count [[buffer(4)]],
    constant uint& ht_capacity [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    uint key = uint(probe_keys[gid]);
    uint mask = ht_capacity - 1u;
    uint slot = key & mask;
    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint bucket_key = ht_keys[slot];
        if (bucket_key == 0xFFFFFFFFu) { output[gid] = -1; return; }
        if (bucket_key == key) { output[gid] = ht_row_indices[slot]; return; }
        slot = (slot + 1u) & mask;
    }
    output[gid] = -1;
}

kernel void dense_lookup_build_i32_v(
    device const int* keys [[buffer(0)]],
    device int* lookup_rows [[buffer(1)]],
    device const uint* count [[buffer(2)]],
    constant int& min_key [[buffer(3)]],
    constant uint& slot_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;

    int slot = keys[gid] - min_key;
    if (slot < 0 || uint(slot) >= slot_count) return;
    lookup_rows[slot] = int(gid);
}

kernel void dense_lookup_probe_i32_v(
    device const int* probe_keys [[buffer(0)]],
    device int* output [[buffer(1)]],
    device const int* lookup_rows [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    constant int& min_key [[buffer(4)]],
    constant uint& slot_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;

    int slot = probe_keys[gid] - min_key;
    if (slot < 0 || uint(slot) >= slot_count) {
        output[gid] = -1;
        return;
    }
    output[gid] = lookup_rows[slot];
}

kernel void dense_lookup_build_from_indices_i32_v(
    device const int* column [[buffer(0)]],
    device const uint* row_ids [[buffer(1)]],
    device int* lookup_rows [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    constant int& min_key [[buffer(4)]],
    constant uint& slot_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;

    int slot = column[row_ids[gid]] - min_key;
    if (slot < 0 || uint(slot) >= slot_count) return;
    lookup_rows[slot] = int(gid);
}

kernel void dense_lookup_probe_from_indices_i32_v(
    device const int* column [[buffer(0)]],
    device const uint* row_ids [[buffer(1)]],
    device int* output [[buffer(2)]],
    device const int* lookup_rows [[buffer(3)]],
    device const uint* count [[buffer(4)]],
    constant int& min_key [[buffer(5)]],
    constant uint& slot_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;

    int slot = column[row_ids[gid]] - min_key;
    if (slot < 0 || uint(slot) >= slot_count) {
        output[gid] = -1;
        return;
    }
    output[gid] = lookup_rows[slot];
}

inline uint membership_hash0(uint key) {
    return key * 0x9E3779B1u;
}

inline uint membership_hash1(uint key) {
    return (key ^ 0x85EBCA77u) * 0xC2B2AE3Du;
}

kernel void build_membership_filter_v(
    device const int* keys [[buffer(0)]],
    device atomic_uint* membership [[buffer(1)]],
    device const uint* count [[buffer(2)]],
    constant uint& bit_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    uint key = as_type<uint>(keys[gid]);
    uint bit_mask = bit_count - 1u;

    uint bit0 = membership_hash0(key) & bit_mask;
    uint bit1 = membership_hash1(key) & bit_mask;

    atomic_fetch_or_explicit(&membership[bit0 >> 5], 1u << (bit0 & 31u), memory_order_relaxed);
    atomic_fetch_or_explicit(&membership[bit1 >> 5], 1u << (bit1 & 31u), memory_order_relaxed);
}

kernel void membership_filter_mask_v(
    device const int* probe_keys [[buffer(0)]],
    device const atomic_uint* membership [[buffer(1)]],
    device uchar* mask [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    constant uint& bit_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    uint key = as_type<uint>(probe_keys[gid]);
    uint bit_mask = bit_count - 1u;

    uint bit0 = membership_hash0(key) & bit_mask;
    uint bit1 = membership_hash1(key) & bit_mask;

    uint word0 = atomic_load_explicit(&membership[bit0 >> 5], memory_order_relaxed);
    uint word1 = atomic_load_explicit(&membership[bit1 >> 5], memory_order_relaxed);
    bool present =
        ((word0 >> (bit0 & 31u)) & 1u) != 0u &&
        ((word1 >> (bit1 & 31u)) & 1u) != 0u;
    mask[gid] = present ? uchar(1) : uchar(0);
}

kernel void project_f32_scalar_v(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    float value = input[gid];
    switch (op_code) {
        case 0: output[gid] = value + scalar; break;
        case 1: output[gid] = value * scalar; break;
        case 2: output[gid] = value - scalar; break;
        case 3: output[gid] = scalar - value; break;
        default: output[gid] = value; break;
    }
}

kernel void elementwise_f32_v(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    float va = a[gid];
    float vb = b[gid];
    switch (op_code) {
        case 0: output[gid] = va + vb; break;
        case 1: output[gid] = va * vb; break;
        case 2: output[gid] = va - vb; break;
        default: output[gid] = va; break;
    }
}

kernel void project_revenue_from_indices_v(
    device const float* extendedprice [[buffer(0)]],
    device const float* discount [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    uint row = indices[gid];
    output[gid] = extendedprice[row] * (1.0f - discount[row]);
}

kernel void accumulate_dense_slots_v(
    device const int* slot_ids [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device atomic_uint* sums [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    constant uint& slot_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    int slot = slot_ids[gid];
    if (slot < 0 || uint(slot) >= slot_count) return;
    atomic_add_float(&sums[slot], values[gid]);
}

kernel void accumulate_revenue_dense_slots_from_indices_v(
    device const int* slot_ids [[buffer(0)]],
    device const float* extendedprice [[buffer(1)]],
    device const float* discount [[buffer(2)]],
    device const uint* row_ids [[buffer(3)]],
    device atomic_uint* sums [[buffer(4)]],
    device const uint* count [[buffer(5)]],
    constant uint& slot_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    int slot = slot_ids[gid];
    if (slot < 0 || uint(slot) >= slot_count) return;
    uint row = row_ids[gid];
    atomic_add_float(&sums[slot], extendedprice[row] * (1.0f - discount[row]));
}

kernel void write_dispatch_args(
    device const uint* count [[buffer(0)]],
    device uint* dispatch_args [[buffer(1)]],
    constant uint& threads_per_group [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint rows = count[0];
    uint groups = rows == 0u ? 1u : ((rows + threads_per_group - 1u) / threads_per_group);
    dispatch_args[0] = groups;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

kernel void hash_agg_grouped_sum_v(
    device const uint* representative_row_ids [[buffer(0)]],
    device const int* secondary_order_values [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device atomic_uint* ht_row_ids [[buffer(3)]],
    device int* ht_secondary [[buffer(4)]],
    device atomic_uint* ht_sums [[buffer(5)]],
    device const uint* count [[buffer(6)]],
    constant uint& ht_capacity [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;
    uint representative_row_id = representative_row_ids[gid];
    float row_value = values[gid];
    uint mask = ht_capacity - 1u;
    uint slot = (representative_row_id * 2654435761u) & mask;

    for (uint probe = 0u; probe < ht_capacity; probe++) {
        uint expected = 0xFFFFFFFFu;
        bool claimed = atomic_compare_exchange_weak_explicit(
            &ht_row_ids[slot], &expected, representative_row_id,
            memory_order_relaxed, memory_order_relaxed
        );
        if (claimed) {
            ht_secondary[slot] = secondary_order_values[representative_row_id];
            atomic_add_float(&ht_sums[slot], row_value);
            return;
        }
        if (expected == representative_row_id) {
            atomic_add_float(&ht_sums[slot], row_value);
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

kernel void dense_row_id_sum_track_live_v(
    device const uint* representative_row_ids [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device atomic_uint* seen [[buffer(2)]],
    device atomic_uint* sums [[buffer(3)]],
    device uint* live_row_ids [[buffer(4)]],
    device atomic_uint* live_count [[buffer(5)]],
    device const uint* count [[buffer(6)]],
    constant uint& row_capacity [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count[0]) return;

    uint row_id = representative_row_ids[gid];
    if (row_id >= row_capacity) return;

    uint prior_seen = atomic_exchange_explicit(&seen[row_id], 1u, memory_order_relaxed);
    if (prior_seen == 0u) {
        uint live_slot = atomic_fetch_add_explicit(live_count, 1u, memory_order_relaxed);
        live_row_ids[live_slot] = row_id;
    }
    atomic_add_float(&sums[row_id], values[gid]);
}

constexpr constant uint MAX_GROUPED_TOPK = 64u;

inline bool grouped_candidate_better(
    float lhs_sum,
    int lhs_date,
    uint lhs_row_id,
    float rhs_sum,
    int rhs_date,
    uint rhs_row_id,
    bool use_secondary
) {
    if (lhs_sum > rhs_sum) return true;
    if (lhs_sum < rhs_sum) return false;
    if (use_secondary) {
        if (lhs_date < rhs_date) return true;
        if (lhs_date > rhs_date) return false;
    }
    return lhs_row_id < rhs_row_id;
}

kernel void compact_grouped_rows(
    device const uint* ht_row_ids [[buffer(0)]],
    device const int* ht_secondary [[buffer(1)]],
    device const atomic_uint* ht_sums [[buffer(2)]],
    device uint* dense_row_ids [[buffer(3)]],
    device int* dense_secondary [[buffer(4)]],
    device float* dense_sums [[buffer(5)]],
    device atomic_uint* dense_count [[buffer(6)]],
    constant uint& ht_capacity [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= ht_capacity) return;

    uint representative = ht_row_ids[gid];
    if (representative == 0xFFFFFFFFu) return;

    uint dense_slot = atomic_fetch_add_explicit(dense_count, 1u, memory_order_relaxed);
    dense_row_ids[dense_slot] = representative;
    dense_secondary[dense_slot] = ht_secondary[gid];
    dense_sums[dense_slot] = as_type<float>(
        atomic_load_explicit(&ht_sums[gid], memory_order_relaxed)
    );
}

kernel void gather_live_grouped_rows_v(
    device const uint* live_row_ids [[buffer(0)]],
    device const int* secondary_order_values [[buffer(1)]],
    device const atomic_uint* sums [[buffer(2)]],
    device int* dense_secondary [[buffer(3)]],
    device float* dense_sums [[buffer(4)]],
    device const uint* live_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= live_count[0]) return;

    uint row_id = live_row_ids[gid];
    dense_secondary[gid] = secondary_order_values[row_id];
    dense_sums[gid] = as_type<float>(
        atomic_load_explicit(&sums[row_id], memory_order_relaxed)
    );
}

kernel void topk_dense_grouped_rows(
    device const uint* dense_row_ids [[buffer(0)]],
    device const int* dense_secondary [[buffer(1)]],
    device const float* dense_sums [[buffer(2)]],
    device const uint* dense_count [[buffer(3)]],
    device uint* out_row_ids [[buffer(4)]],
    device float* out_sums [[buffer(5)]],
    device uint* out_count [[buffer(6)]],
    device uint* out_dispatch_args [[buffer(7)]],
    constant uint& limit [[buffer(8)]],
    constant uint& use_secondary [[buffer(9)]],
    constant uint& threads_per_group [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    thread float best_sums[MAX_GROUPED_TOPK];
    thread uint best_row_ids[MAX_GROUPED_TOPK];
    thread int best_dates[MAX_GROUPED_TOPK];
    uint top_count = 0u;
    bool secondary = use_secondary != 0u;
    uint capped_limit = min(limit, MAX_GROUPED_TOPK);
    uint input_count = dense_count[0];

    for (uint row = 0u; row < input_count; row++) {
        uint representative = dense_row_ids[row];
        float candidate_sum = dense_sums[row];
        int candidate_date = dense_secondary[row];

        uint insert_at = top_count;
        for (uint i = 0u; i < top_count; i++) {
            if (grouped_candidate_better(
                    candidate_sum, candidate_date, representative,
                    best_sums[i], best_dates[i], best_row_ids[i],
                    secondary)) {
                insert_at = i;
                break;
            }
        }

        if (insert_at >= capped_limit) {
            if (top_count < capped_limit) {
                best_sums[top_count] = candidate_sum;
                best_row_ids[top_count] = representative;
                best_dates[top_count] = candidate_date;
                top_count++;
            }
            continue;
        }

        uint shift_end = top_count < capped_limit ? top_count : (capped_limit - 1u);
        while (shift_end > insert_at) {
            best_sums[shift_end] = best_sums[shift_end - 1u];
            best_row_ids[shift_end] = best_row_ids[shift_end - 1u];
            best_dates[shift_end] = best_dates[shift_end - 1u];
            shift_end--;
        }

        best_sums[insert_at] = candidate_sum;
        best_row_ids[insert_at] = representative;
        best_dates[insert_at] = candidate_date;
        if (top_count < capped_limit) {
            top_count++;
        }
    }

    out_count[0] = top_count;
    uint groups = top_count == 0u ? 1u : ((top_count + threads_per_group - 1u) / threads_per_group);
    out_dispatch_args[0] = groups;
    out_dispatch_args[1] = 1u;
    out_dispatch_args[2] = 1u;
    for (uint i = 0u; i < top_count; i++) {
        out_row_ids[i] = best_row_ids[i];
        out_sums[i] = best_sums[i];
    }
}

// Parallel top-K using a single threadgroup. Each thread maintains a thread-local
// top-K in registers while striding through the dense rows. After the scan, all
// threads publish their partial top-K to threadgroup memory and thread 0 merges
// the (tg_size * limit) candidates into the final top-K.
//
// The caller must:
//   - dispatch exactly one threadgroup of `tg_size` threads
//   - supply threadgroup memory of size tg_size * limit * (sum+date+row_id) bytes
//     plus tg_size * sizeof(uint) for the per-thread counts, via setThreadgroupMemoryLength
//   - ensure limit * tg_size * 12 + tg_size * 4 <= 32 KB
kernel void topk_dense_grouped_rows_parallel_v(
    device const uint*   dense_row_ids     [[buffer(0)]],
    device const int*    dense_secondary   [[buffer(1)]],
    device const float*  dense_sums        [[buffer(2)]],
    device const uint*   dense_count       [[buffer(3)]],
    device uint*         out_row_ids       [[buffer(4)]],
    device float*        out_sums          [[buffer(5)]],
    device uint*         out_count         [[buffer(6)]],
    device uint*         out_dispatch_args [[buffer(7)]],
    constant uint&       limit             [[buffer(8)]],
    constant uint&       use_secondary     [[buffer(9)]],
    constant uint&       threads_per_group [[buffer(10)]],
    threadgroup float*   tg_sums           [[threadgroup(0)]],
    threadgroup int*     tg_dates          [[threadgroup(1)]],
    threadgroup uint*    tg_row_ids        [[threadgroup(2)]],
    threadgroup uint*    tg_counts         [[threadgroup(3)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint tgsz                              [[threads_per_threadgroup]]
) {
    uint input_count = dense_count[0];
    bool secondary = use_secondary != 0u;
    uint capped_limit = min(limit, MAX_GROUPED_TOPK);

    thread float best_sums[MAX_GROUPED_TOPK];
    thread uint  best_row_ids[MAX_GROUPED_TOPK];
    thread int   best_dates[MAX_GROUPED_TOPK];
    uint top_count = 0u;

    // Strided scan: thread `tid` handles rows tid, tid+tgsz, tid+2*tgsz, ...
    for (uint row = tid; row < input_count; row += tgsz) {
        float cs = dense_sums[row];
        int   cd = dense_secondary[row];
        uint  cr = dense_row_ids[row];

        // Fast reject against current worst when at capacity.
        if (top_count == capped_limit) {
            if (!grouped_candidate_better(cs, cd, cr,
                    best_sums[capped_limit - 1u], best_dates[capped_limit - 1u], best_row_ids[capped_limit - 1u],
                    secondary)) {
                continue;
            }
        }

        uint insert_at = top_count;
        for (uint i = 0u; i < top_count; i++) {
            if (grouped_candidate_better(cs, cd, cr,
                    best_sums[i], best_dates[i], best_row_ids[i], secondary)) {
                insert_at = i;
                break;
            }
        }
        if (insert_at >= capped_limit) {
            if (top_count < capped_limit) {
                best_sums[top_count]    = cs;
                best_row_ids[top_count] = cr;
                best_dates[top_count]   = cd;
                top_count++;
            }
            continue;
        }
        uint shift_end = top_count < capped_limit ? top_count : (capped_limit - 1u);
        while (shift_end > insert_at) {
            best_sums[shift_end]    = best_sums[shift_end - 1u];
            best_row_ids[shift_end] = best_row_ids[shift_end - 1u];
            best_dates[shift_end]   = best_dates[shift_end - 1u];
            shift_end--;
        }
        best_sums[insert_at]    = cs;
        best_row_ids[insert_at] = cr;
        best_dates[insert_at]   = cd;
        if (top_count < capped_limit) top_count++;
    }

    // Publish thread-local top-K to threadgroup memory.
    for (uint i = 0u; i < top_count; i++) {
        tg_sums[tid * capped_limit + i]    = best_sums[i];
        tg_dates[tid * capped_limit + i]   = best_dates[i];
        tg_row_ids[tid * capped_limit + i] = best_row_ids[i];
    }
    tg_counts[tid] = top_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 performs the final merge of tg_size partial top-K lists.
    if (tid != 0u) return;

    top_count = 0u;  // reset to rebuild from partials
    for (uint s = 0u; s < tgsz; s++) {
        uint slot_count = tg_counts[s];
        for (uint j = 0u; j < slot_count; j++) {
            float cs = tg_sums[s * capped_limit + j];
            int   cd = tg_dates[s * capped_limit + j];
            uint  cr = tg_row_ids[s * capped_limit + j];

            if (top_count == capped_limit) {
                if (!grouped_candidate_better(cs, cd, cr,
                        best_sums[capped_limit - 1u], best_dates[capped_limit - 1u], best_row_ids[capped_limit - 1u],
                        secondary)) {
                    continue;
                }
            }

            uint insert_at = top_count;
            for (uint i = 0u; i < top_count; i++) {
                if (grouped_candidate_better(cs, cd, cr,
                        best_sums[i], best_dates[i], best_row_ids[i], secondary)) {
                    insert_at = i;
                    break;
                }
            }
            if (insert_at >= capped_limit) {
                if (top_count < capped_limit) {
                    best_sums[top_count]    = cs;
                    best_row_ids[top_count] = cr;
                    best_dates[top_count]   = cd;
                    top_count++;
                }
                continue;
            }
            uint shift_end = top_count < capped_limit ? top_count : (capped_limit - 1u);
            while (shift_end > insert_at) {
                best_sums[shift_end]    = best_sums[shift_end - 1u];
                best_row_ids[shift_end] = best_row_ids[shift_end - 1u];
                best_dates[shift_end]   = best_dates[shift_end - 1u];
                shift_end--;
            }
            best_sums[insert_at]    = cs;
            best_row_ids[insert_at] = cr;
            best_dates[insert_at]   = cd;
            if (top_count < capped_limit) top_count++;
        }
    }

    out_count[0] = top_count;
    uint groups = top_count == 0u ? 1u : ((top_count + threads_per_group - 1u) / threads_per_group);
    out_dispatch_args[0] = groups;
    out_dispatch_args[1] = 1u;
    out_dispatch_args[2] = 1u;
    for (uint i = 0u; i < top_count; i++) {
        out_row_ids[i] = best_row_ids[i];
        out_sums[i]    = best_sums[i];
    }
}

"#;
