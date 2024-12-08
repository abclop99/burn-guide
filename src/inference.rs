//! Inference

use burn::{
    data::{dataloader::batcher::Batcher as _, dataset::vision::MnistItem},
    prelude::*,
    record::{CompactRecorder, Recorder},
};

use crate::{data::MnistBatcher, training::TrainingConfig};

/// Runs inference with a trained model
///
/// # Parameters
///
/// - `artifact_dir`
///     The directory containing the trained model.
/// - `device`
///     The device to run the model on.
/// - `item`
///     A single MNIST item to run the inference on.
pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    // Set up training
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    // Create the model
    let model = config.model.init::<B>(&device).load_record(record);

    // Set up data
    let label = item.label;
    let batcher = MnistBatcher::new(device);
    let batch = batcher.batch(vec![item]);

    // Run model and predict
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    // Print results
    println!("Predicted {predicted} Expected {label}");
}
