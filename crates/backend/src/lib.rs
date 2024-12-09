//! Select a backend

use cfg_if::cfg_if;
// Re-export the specified backend
pub use specified_backend::*;

// Select a Burn backend
cfg_if! {
    if #[cfg(feature = "wgpu")] {
        mod specified_backend {

            use burn::backend::{wgpu::WgpuDevice, Wgpu};

            /// The backend to use.
            pub type Backend = Wgpu<f32, i32>;

            /// Select a device
            pub fn get_device() -> WgpuDevice {
                WgpuDevice::default()
            }
        }
    } else if #[cfg(feature = "tch-gpu")] {
        mod specified_backend {

            use burn::backend::libtorch::{LibTorch, LibTorchDevice};

            /// The backend to use.
            pub type Backend = LibTorch<f32, i8>;

            /// Select a device
            pub fn get_device() -> LibTorchDevice {
                if cfg!(target_os = "macos") {
                    LibTorchDevice::Mps
                } else {
                    LibTorchDevice::Cuda(0)
                }
            }
        }
    } else if #[cfg(feature = "tch-cpu")] {
        mod specified_backend {

            use burn::backend::libtorch::{LibTorch, LibTorchDevice};

            /// The backend to use.
            pub type Backend = LibTorch;

            /// Select a device
            pub fn get_device() -> LibTorchDevice {
                LibTorchDevice::Cpu
            }
        }
    } else if #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))] {
        mod specified_backend {

            use burn::backend::ndarray::{NdArray, NdArrayDevice};

            /// The backend to use.
            pub type Backend = NdArray<f32, i32>;

            /// Select a device
            pub fn get_device() -> NdArrayDevice {
                NdArrayDevice::Cpu
            }
        }
    } else {
        compile_error!("None of the backend features were enabled. Exactly one must be enabled.");
    }
}
