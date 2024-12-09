//! Select a backend

// Ensure a backend is enabled
#[cfg(not(any(
    feature = "ndarray",
    feature = "ndarray-blas-accelerate",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "tch-cpu",
    feature = "tch-gpu",
    feature = "wgpu",
)))]
compile_error!("None of the backend features were enabled. Exactly one must be enabled.");

// Re-export the specified backend
pub use specified_backend::*;

/// The NDArray backend
#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod specified_backend {

    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    /// The backend to use.
    pub type Backend = NdArray<f32, i32>;

    /// Select a device
    pub fn get_device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }
}

/// The Torch-CPU backend
#[cfg(feature = "tch-cpu")]
mod specified_backend {

    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    /// The backend to use.
    pub type Backend = LibTorch;

    /// Select a device
    pub fn get_device() -> LibTorchDevice {
        LibTorchDevice::Cpu
    }
}

/// The Torch-GPU backend
#[cfg(feature = "tch-gpu")]
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

/// The WGPU backend
#[cfg(feature = "wgpu")]
mod specified_backend {

    use burn::backend::{wgpu::WgpuDevice, Wgpu};

    /// The backend to use.
    pub type Backend = Wgpu<f32, i32>;

    /// Select a device
    pub fn get_device() -> WgpuDevice {
        WgpuDevice::default()
    }
}
