//! Inference

use burn::data::dataset::{vision::MnistDataset, Dataset as _};

use model::inference;

fn main() {
    let device = backend::get_device();
    let artifact_dir = "/tmp/guide";

    // Inference
    inference::infer::<backend::Backend>(
        artifact_dir,
        device,
        MnistDataset::test().get(42).unwrap(),
    );
}
