//! Inference

use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    data::dataset::{vision::MnistDataset, Dataset as _},
};

use guide::inference;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = WgpuDevice::default();
    let artifact_dir = "/tmp/guide";

    // Inference
    inference::infer::<MyBackend>(artifact_dir, device, MnistDataset::test().get(42).unwrap());
}
