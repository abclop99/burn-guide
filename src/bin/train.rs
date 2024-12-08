//! Training

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    optim::AdamConfig,
};

use guide::{
    model::ModelConfig,
    training::{self, TrainingConfig},
};

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
}
