mod activation;
mod layers;
mod loss;

pub use activation::{relu, relu_derivative, softmax};
pub use layers::NeuralNetwork;
pub use loss::cross_entropy_loss;