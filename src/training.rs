use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::vision::MnistDataset,
    },
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

use crate::data::{MnistBatch, MnistBatcher};
use crate::model::{Model, ModelConfig};

impl<B: Backend> Model<B> {
    /// Forward function for classification training
    ///
    /// # Parameters
    ///
    /// - `images`
    ///     A batch of images to run the model on.
    /// - `targets`
    ///     A batch of ground truths.
    ///
    /// # Returns
    ///
    /// A [`ClassificationOutput`]
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

/// Configuration for training
#[derive(Config)]
pub struct TrainingConfig {
    /// Configuration for the model
    pub model: ModelConfig,
    /// Configuration for the optimizer
    pub optimizer: AdamConfig,
    /// Number of epochs to train for
    #[config(default = 10)]
    pub num_epochs: usize,
    /// The batch size
    #[config(default = 64)]
    pub batch_size: usize,
    /// The number of workers for the [`DataLoader`]s
    #[config(default = 4)]
    pub num_workers: usize,
    /// The seed for random operations
    #[config(default = 42)]
    pub seed: u64,
    /// The learning rate
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

/// Create a clean artifact directory
///
/// # Parameters
///
/// - `artifact_dir`
///     The directory to store the trained model and other artifacts.
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// The training function
///
/// # Parameters
///
/// - `artifact_dir`
///     The directory to store the trained model and other artifacts.
/// - `config`
///     The training config
/// - `device`
///     The device to run the training on.
pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    // Save config into artifact dir
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // Set the seed
    B::seed(config.seed);

    // Create batchers for training and validation data
    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    // Create the training and validation dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    // Set up training
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    // Train model
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // Save model
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
