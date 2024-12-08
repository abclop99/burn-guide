use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataset::{vision::MnistDataset, Dataset as _},
    optim::AdamConfig,
};

mod data;
mod inference;
mod model;
mod training;

use crate::{model::ModelConfig, training::TrainingConfig};

/// The main function.
///
/// Runs the training then runs inference on a MNIST test item.
fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let artifact_dir = "/tmp/guide";

    // Training
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    // Inference
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        MnistDataset::test().get(42).unwrap(),
    );
}
