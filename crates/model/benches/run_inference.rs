use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{vision::MnistDataset, Dataset},
};
use itertools::Itertools;
use model::{
    data::MnistBatcher,
    model::{Model, ModelConfig},
};

/// Run model on multiple batches
pub fn run_inference(c: &mut Criterion) {
    let device = backend::get_device();

    // Initialize a model
    let config = ModelConfig::new(10, 512);
    let model: Model<backend::Backend> = config.init(&device);

    // Batch the data
    let dataset = MnistDataset::test();
    c.bench_function("model.forward", |b| {
        b.iter(|| {
            dataset.iter().chunks(64).into_iter().for_each(|batch| {
                let items = batch.collect_vec();

                // Set up data
                let _labels: Vec<_> = items.iter().map(|item| item.label).collect();
                let batcher: MnistBatcher<backend::Backend> = MnistBatcher::new(device.clone());
                let batch = batcher.batch(items);

                // Run model
                let _prediction = model.forward(black_box(batch.images));
            })
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(100))
        .sample_size(60)
        .warm_up_time(Duration::from_secs(10));
    targets = run_inference
}

criterion_main!(benches);
