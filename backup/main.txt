use airquality_nn::data::{load_dataset, normalize_data};
use airquality_nn::neural_network::{NeuralNetwork, cross_entropy_loss};
use airquality_nn::visualization::plot_training_progress;
use ndarray::Axis;

fn main() {
    // Load and prepare data
    let (x, y) = load_dataset("csv/airquality.csv");
    let x_normalized = normalize_data(&x);

    // Initialize neural network
    let mut nn = NeuralNetwork::new(5, 10, 10, 10, 3);

    // Training parameters
    let initial_learning_rate = 0.001;
    let lambda = 0.01;
    let epochs = 1000;
    let plot_interval = 10; // Plot setiap 10 epoch

    // Untuk menyimpan progress training
    let mut training_epochs = Vec::new();
    let mut training_losses = Vec::new();
    let mut training_accuracies = Vec::new();

    // Training loop
    for epoch in 0..epochs {
        let learning_rate = initial_learning_rate * (1.0 / (1.0 + 0.1 * (epoch as f64)));
        nn.train(&x_normalized, &y, learning_rate, lambda);
        
        if epoch % plot_interval == 0 {
            let (_, _, _, output) = nn.forward(&x_normalized);
            let loss = cross_entropy_loss(&y, &output);
            
            // Hitung akurasi
            let predictions = output.map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap()
            });
            let true_labels = y.map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap()
            });
            let accuracy = predictions.iter()
                .zip(true_labels.iter())
                .filter(|&(a, b)| a == b)
                .count() as f64 / predictions.len() as f64;

            println!("Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", 
                epoch, loss, accuracy * 100.0);

            // Simpan data untuk plotting
            training_epochs.push(epoch);
            training_losses.push(loss);
            training_accuracies.push(accuracy);
        }
    }

    // Plot training progress
    plot_training_progress(
        &training_epochs,
        &training_losses,
        &training_accuracies,
        "training_progress.png",
    ).expect("Gagal membuat plot training progress");

    println!("Grafik training progress disimpan sebagai training_progress.png");
}