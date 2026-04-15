//! Metal library/pipeline compilation and dispatch helpers.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::size_of_val;

use crate::error::{MettleError, Result};

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
use super::METAL_KERNEL_SOURCE;

#[cfg(target_os = "macos")]
pub(in crate::metal) fn hash_values<T: Copy>(values: &[T]) -> u64 {
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) };
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

#[cfg(target_os = "macos")]
pub(in crate::metal) fn compile_library(
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
pub(in crate::metal) fn compile_pipeline(
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
pub(in crate::metal) fn dispatch_threads(
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
