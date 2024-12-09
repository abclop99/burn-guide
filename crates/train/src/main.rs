//! Training

use burn::{backend::Autodiff, optim::AdamConfig};

use model::{
    model::ModelConfig,
    training::{train, TrainingConfig},
};

fn main() {
    type MyAutodiffBackend = Autodiff<backend::Backend>;

    let device = backend::get_device();
    let artifact_dir = "/tmp/guide";

    // Training
    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
