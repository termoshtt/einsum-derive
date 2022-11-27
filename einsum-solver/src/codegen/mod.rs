//! Code generation for einsum for specific linear algebra libraries.

mod format;
pub use format::format_block;

pub mod ndarray;
