use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    optim::AdamConfig,
};

mod data;
mod model;
mod training;

use crate::{model::ModelConfig, training::TrainingConfig};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
