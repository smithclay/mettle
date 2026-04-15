use crate::error::{MettleError, Result};

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// GPU-resident buffer handle. Wraps a Metal buffer so data stays on GPU
/// between operations without copying back to CPU.
#[derive(Clone)]
pub struct GpuBuffer {
    #[cfg(target_os = "macos")]
    pub(super) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
    pub element_size: usize,
}

#[cfg(target_os = "macos")]
impl GpuBuffer {
    pub fn upload<T: Copy>(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> Result<Self> {
        let buffer = make_buffer_from_slice(device, data, "gpu_buffer upload")?;
        Ok(GpuBuffer {
            buffer,
            len: data.len(),
            element_size: std::mem::size_of::<T>(),
        })
    }

    pub fn download_as<T: Copy>(&self) -> Vec<T> {
        assert_eq!(self.element_size, std::mem::size_of::<T>());
        let ptr = self.buffer.contents().as_ptr().cast::<T>();
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len) };
        slice.to_vec()
    }

    pub fn byte_len(&self) -> usize {
        self.len * self.element_size
    }
}

/// GPU-resident variable-length array. The count lives on GPU — never read
/// until after `end_batch()`. Downstream kernels read `count[0]` and
/// early-exit threads beyond it.
#[derive(Clone)]
pub struct GpuVector {
    pub data: GpuBuffer,
    pub count: GpuBuffer,         // single u32
    pub dispatch_args: GpuBuffer, // MTLDispatchThreadgroupsIndirectArguments
    pub max_len: usize,           // dispatch grid size
}

#[cfg(target_os = "macos")]
impl GpuVector {
    /// Wrap CPU-computed indices (e.g. from string filter) as a GpuVector.
    pub fn from_upload(device: &ProtocolObject<dyn MTLDevice>, indices: &[u32]) -> Result<Self> {
        let data = GpuBuffer::upload(device, indices)?;
        let count_val = [indices.len() as u32];
        let count = GpuBuffer::upload(device, &count_val)?;
        let groups = dispatch_group_count(indices.len());
        let dispatch_args = GpuBuffer::upload(device, &[groups, 1, 1])?;
        Ok(GpuVector {
            max_len: indices.len(),
            data,
            count,
            dispatch_args,
        })
    }

    /// Create a GpuVector that inherits count and dispatch metadata from another GpuVector.
    pub fn inherited_with_dispatch(
        data: GpuBuffer,
        count: &GpuBuffer,
        dispatch_args: &GpuBuffer,
        max_len: usize,
    ) -> Self {
        GpuVector {
            data,
            count: GpuBuffer {
                buffer: count.buffer.clone(),
                len: count.len,
                element_size: count.element_size,
            },
            dispatch_args: GpuBuffer {
                buffer: dispatch_args.buffer.clone(),
                len: dispatch_args.len,
                element_size: dispatch_args.element_size,
            },
            max_len,
        }
    }

    /// Read the count from GPU. Only valid after `end_batch()`.
    pub fn download_count(&self) -> usize {
        debug_assert!(self.count.len >= 1, "count buffer must have at least one slot");
        debug_assert_eq!(
            self.count.element_size,
            std::mem::size_of::<u32>(),
            "count buffer element size must be u32"
        );
        unsafe { *self.count.buffer.contents().as_ptr().cast::<u32>() as usize }
    }

    /// Download `count` elements as the given type.
    pub fn download_data_as<T: Copy>(&self, count: usize) -> Vec<T> {
        assert_eq!(self.data.element_size, std::mem::size_of::<T>());
        debug_assert!(
            count <= self.data.len,
            "download_data_as: requested {} elements, buffer has {}",
            count,
            self.data.len
        );
        let ptr = self.data.buffer.contents().as_ptr().cast::<T>();
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }
}

#[cfg(target_os = "macos")]
fn dispatch_group_count(len: usize) -> u32 {
    const GV_THREADS_PER_GROUP: usize = 256;
    if len == 0 {
        0
    } else {
        u32::try_from(len.div_ceil(GV_THREADS_PER_GROUP)).unwrap_or(u32::MAX)
    }
}

#[cfg(target_os = "macos")]
pub(super) fn allocate_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    byte_len: usize,
    label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    device
        .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            MettleError::MetalUnavailable(format!("failed to allocate Metal buffer for '{label}'"))
        })
}

#[cfg(target_os = "macos")]
pub(super) fn make_buffer_from_slice<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    values: &[T],
    label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let buffer = allocate_buffer(device, std::mem::size_of_val(values), label)?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().as_ptr().cast::<T>(),
            values.len(),
        );
    }
    Ok(buffer)
}
