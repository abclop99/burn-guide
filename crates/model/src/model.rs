use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};

/// The configuration for a MNIST type model
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// The number of classes to classify for. This is 10 for MNIST.
    num_classes: usize,
    /// The size of the hidden layer in the linear layers.
    hidden_size: usize,
    /// The probability of randomly zeroing each element of the data between
    /// the layers.
    #[config(default = 0.5)]
    dropout: f64,
}

impl ModelConfig {
    // Creates a model
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

/// The model
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    /// The first convolution layer
    conv1: Conv2d<B>,
    /// The second convolution layer
    conv2: Conv2d<B>,
    /// The pooling layer
    pool: AdaptiveAvgPool2d,
    /// The dropout layers
    dropout: Dropout,
    /// The first linear layer
    linear1: Linear<B>,
    /// The second linear layer
    linear2: Linear<B>,
    /// The activation layer
    activation: Relu,
}

impl<B: Backend> Model<B> {
    /// The forward function that runs the model.
    ///
    /// # Parameters
    ///
    /// - images
    ///     A batch of images.
    ///     
    ///     Format: [B, H, W]
    ///
    /// # Output
    ///
    /// A [`Tensor`] of the predicitons.
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}

/// Tests for the model
#[cfg(test)]
mod test {

    use pretty_assertions::assert_eq;

    /// Run the model
    #[test]
    fn run_model() {
        let device = backend::get_device();

        // Initialize a model
        let config = super::ModelConfig::new(10, 512);
        let model: super::Model<backend::Backend> = config.init(&device);

        // Get a batch of data
        let batch = crate::data::test::get_batch(device);

        // Run model
        let predictions = model.forward(batch.images.clone());

        // Test the batch size
        assert_eq!(
            batch.images.dims()[0],
            predictions.dims()[0],
            "The batch size of the images and predictions should be the same."
        );

        // Test the batch size
        assert_eq!(
            predictions.dims()[1],
            config.num_classes,
            "The number of classes should be {}",
            config.num_classes,
        );
    }
}
