//! The dataset

use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

/// Batcher fo MNIST data
#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    /// The device to create the data on.
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    /// Create a new [`MnistBatcher`].
    ///
    /// # Parameters
    ///
    /// - The device to put the [`MnistBatch`] data on.
    ///
    /// # Returns
    ///
    /// A [`MnistBatcher`]
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

/// A batch of MNIST data
#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    /// The Mnist data images.
    ///
    /// Format: [B, H, W]
    pub images: Tensor<B, 3>,
    /// The ground truth labels for the MNIST data.
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Normalize: make vetween [0, 1] and make the mean=0 and std=1
            // Values mean=0.1307 and std=0.2081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [(item.label as i64).elem::<B::IntElem>()],
                    &self.device,
                )
            })
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MnistBatch { images, targets }
    }
}

/// Tests for data
#[cfg(test)]
pub(crate) mod test {
    use burn::data::dataloader::batcher::Batcher as _;
    use burn::data::dataset::vision::MnistItem;
    use burn::data::dataset::{vision::MnistDataset, Dataset as _};
    use burn::prelude::*;
    use pretty_assertions::assert_eq;

    use super::{MnistBatch, MnistBatcher};

    /// Get a batch of data for testing
    ///
    /// # Parameters
    ///
    /// - `device`: The device to create the batch on.
    ///
    /// # Returns
    ///
    /// - A batch of data.
    pub(crate) fn get_batch<B: Backend>(device: B::Device) -> MnistBatch<B> {
        // Get an item
        let dataset = MnistDataset::test();
        let items: Vec<MnistItem> = (0..50).map(|i| dataset.get(i).unwrap()).collect();

        // Set up data
        let _labels: Vec<_> = items.iter().map(|item| item.label).collect();
        let batcher: MnistBatcher<B> = MnistBatcher::new(device);
        let batch = batcher.batch(items);

        batch
    }

    /// Tests using [`MnistBatcher`] a bit.
    #[test]
    fn mnist_batcher() {
        let device = backend::get_device();

        let batch = get_batch::<backend::Backend>(device);

        // Same number of images and targets.
        assert_eq!(
            batch.images.dims()[0],
            batch.targets.dims()[0],
            "The batch size of the images and targets should be the same."
        );
    }
}
