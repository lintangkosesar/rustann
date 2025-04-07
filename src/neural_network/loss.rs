use ndarray::Array2;

pub fn cross_entropy_loss(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    -(y_true * y_pred.mapv(f64::ln)).sum()
}