use ndarray::{Array, Array2, Axis};  // Tambahkan Array di sini
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use super::activation::{relu, relu_derivative, softmax};

pub struct NeuralNetwork {
    weights1: Array2<f64>,
    bias1: Array2<f64>,
    weights2: Array2<f64>,
    bias2: Array2<f64>,
    weights3: Array2<f64>,
    bias3: Array2<f64>,
    weights4: Array2<f64>,
    bias4: Array2<f64>,
}

impl NeuralNetwork {
    pub fn new(
        input_size: usize,
        hidden_size1: usize,
        hidden_size2: usize,
        hidden_size3: usize,
        output_size: usize,
    ) -> Self {
        let he_init = |size: usize| (2.0 / size as f64).sqrt();
        
        NeuralNetwork {
            weights1: Array::random((input_size, hidden_size1), Uniform::new(-he_init(input_size), he_init(input_size))),
            bias1: Array::zeros((1, hidden_size1)),
            weights2: Array::random((hidden_size1, hidden_size2), Uniform::new(-he_init(hidden_size1), he_init(hidden_size1))),
            bias2: Array::zeros((1, hidden_size2)),
            weights3: Array::random((hidden_size2, hidden_size3), Uniform::new(-he_init(hidden_size2), he_init(hidden_size2))),
            bias3: Array::zeros((1, hidden_size3)),
            weights4: Array::random((hidden_size3, output_size), Uniform::new(-he_init(hidden_size3), he_init(hidden_size3))),
            bias4: Array::zeros((1, output_size)),
        }
    }

    pub fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let hidden_input1 = x.dot(&self.weights1) + &self.bias1;
        let hidden_output1 = relu(&hidden_input1);

        let hidden_input2 = hidden_output1.dot(&self.weights2) + &self.bias2;
        let hidden_output2 = relu(&hidden_input2);

        let hidden_input3 = hidden_output2.dot(&self.weights3) + &self.bias3;
        let hidden_output3 = relu(&hidden_input3);

        let output_input = hidden_output3.dot(&self.weights4) + &self.bias4;
        let output_output = softmax(&output_input);

        (hidden_output1, hidden_output2, hidden_output3, output_output)
    }

    pub fn train(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64,
        lambda: f64,
    ) {
        let (hidden_output1, hidden_output2, hidden_output3, output_output) = self.forward(x);

        // Backpropagation
        let output_error = &output_output - y;
        let output_delta = output_error;

        let hidden_error3 = output_delta.dot(&self.weights4.t());
        let hidden_delta3 = hidden_error3 * relu_derivative(&hidden_output3);

        let hidden_error2 = hidden_delta3.dot(&self.weights3.t());
        let hidden_delta2 = hidden_error2 * relu_derivative(&hidden_output2);

        let hidden_error1 = hidden_delta2.dot(&self.weights2.t());
        let hidden_delta1 = hidden_error1 * relu_derivative(&hidden_output1);

        // Update weights and biases
        self.weights4 -= &(learning_rate * (hidden_output3.t().dot(&output_delta) + lambda * &self.weights4));
        self.bias4 -= &(learning_rate * output_delta.sum_axis(Axis(0)).insert_axis(Axis(0)));

        self.weights3 -= &(learning_rate * (hidden_output2.t().dot(&hidden_delta3) + lambda * &self.weights3));
        self.bias3 -= &(learning_rate * hidden_delta3.sum_axis(Axis(0)).insert_axis(Axis(0)));

        self.weights2 -= &(learning_rate * (hidden_output1.t().dot(&hidden_delta2) + lambda * &self.weights2));
        self.bias2 -= &(learning_rate * hidden_delta2.sum_axis(Axis(0)).insert_axis(Axis(0)));

        self.weights1 -= &(learning_rate * (x.t().dot(&hidden_delta1) + lambda * &self.weights1));
        self.bias1 -= &(learning_rate * hidden_delta1.sum_axis(Axis(0)).insert_axis(Axis(0)));
    }
}