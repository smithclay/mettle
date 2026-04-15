//! Metal compute kernel source and compilation helpers.
//!
//! The raw Metal Shading Language kernels live in `kernels.metal` and are
//! included verbatim at compile time. This module exposes the opcode encoders
//! for scalar ops plus the compile/dispatch helpers used by `runtime`.

mod compiler;

pub(in crate::metal) use compiler::*;

use super::{ScalarCompareOp, ScalarProjectOp};

#[cfg(target_os = "macos")]
pub(in crate::metal) const METAL_KERNEL_SOURCE: &str = include_str!("kernels.metal");

impl ScalarProjectOp {
    pub(in crate::metal) fn code(self) -> u32 {
        match self {
            Self::Add => 0,
            Self::Multiply => 1,
            Self::Subtract => 2,
            Self::SubtractFrom => 3,
        }
    }
}

impl ScalarCompareOp {
    pub(in crate::metal) fn code(self) -> u32 {
        match self {
            Self::Eq => 0,
            Self::Lt => 1,
            Self::Le => 2,
            Self::Gt => 3,
            Self::Ge => 4,
        }
    }
}
