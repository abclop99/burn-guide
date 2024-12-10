//! Select a backend

use cfg_if::cfg_if;
// Re-export the specified backend
pub use specified_backend::*;

// Select a Burn backend
cfg_if! {
    if #[cfg(any(
            feature = "wgpu",
            feature = "wgpu-spirv",
        ))] {
        mod specified_backend {

            use burn::backend::{wgpu::WgpuDevice, Wgpu};

            /// The backend ([`Wgpu`]) to use.
            pub type Backend = Wgpu<f32, i32>;

            /// Select a device for the [`Backend`]
            pub fn get_device() -> WgpuDevice {
                WgpuDevice::default()
            }
        }
    } else if #[cfg(feature = "candle-gpu")] {
        mod specified_backend {

            use burn::backend::candle::{Candle, CandleDevice};

            /// The backend ([`LibTorch`]) to use.
            pub type Backend = Candle<f32, i64>;

            /// Select a device for the [`Backend`] (GPU)
            pub fn get_device() -> CandleDevice {
                if cfg!(target_os = "macos") {
                    CandleDevice::metal(0)
                } else {
                    CandleDevice::cuda(0)
                }
            }
        }
    } else if #[cfg(feature = "tch-gpu")] {
        mod specified_backend {

            use burn::backend::libtorch::{LibTorch, LibTorchDevice};

            /// The backend ([`LibTorch`]) to use.
            pub type Backend = LibTorch<f32, i8>;

            /// Select a device for the [`Backend`] (GPU)
            pub fn get_device() -> LibTorchDevice {
                if cfg!(target_os = "macos") {
                    LibTorchDevice::Mps
                } else {
                    LibTorchDevice::Cuda(0)
                }
            }
        }
    } else if #[cfg(feature = "candle-cpu")] {
        mod specified_backend {

            use burn::backend::candle::{Candle, CandleDevice};

            /// The backend ([`LibTorch`]) to use.
            pub type Backend = Candle<f32, i64>;

            /// Select a device for the [`Backend`] (GPU)
            pub fn get_device() -> CandleDevice {
                CandleDevice::Cpu
            }
        }
    } else if #[cfg(feature = "tch-cpu")] {
        mod specified_backend {

            use burn::backend::libtorch::{LibTorch, LibTorchDevice};

            /// The backend ([`LibTorch`]) to use.
            pub type Backend = LibTorch;

            /// Select a device for the [`Backend`] (CPU)
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

            /// The backend ([`Ndarray`]) to use.
            pub type Backend = NdArray<f32, i32>;

            /// Select a device for the [`Backend`]
            pub fn get_device() -> NdArrayDevice {
                NdArrayDevice::Cpu
            }
        }
    } else {
        compile_error!("None of the backend features were enabled. Exactly one must be enabled.");
    }
}

/// Tests
#[cfg(test)]
mod tests {

    use super::*;

    /// Ensure that the device [`get_device`] returns is of the type
    /// associated with the [`Backend`].
    #[test]
    fn backend_and_device_types_compatible() {
        use burn::{backend::Autodiff, prelude::Backend};

        type MyAutoDiffBackend = Autodiff<super::Backend>;

        let _device: <MyAutoDiffBackend as Backend>::Device = get_device();
    }
}
