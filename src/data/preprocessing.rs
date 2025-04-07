use csv::Reader;
use ndarray::{Array2, Axis};
use std::fs::File;

pub fn load_dataset(path: &str) -> (Array2<f64>, Array2<f64>) {
    let file = File::open(path).expect("Gagal membuka file CSV");
    let mut rdr = Reader::from_reader(file);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Gagal membaca baris CSV");

        let pm10: f64 = record[0].parse().unwrap_or_else(|_| {
            eprintln!("Peringatan: Nilai pm10 tidak valid pada baris {:?}", record);
            0.0
        });
        let so2: f64 = record[1].parse().unwrap_or_else(|_| {
            eprintln!("Peringatan: Nilai so2 tidak valid pada baris {:?}", record);
            0.0
        });
        let co: f64 = record[2].parse().unwrap_or_else(|_| {
            eprintln!("Peringatan: Nilai co tidak valid pada baris {:?}", record);
            0.0
        });
        let o3: f64 = record[3].parse().unwrap_or_else(|_| {
            eprintln!("Peringatan: Nilai o3 tidak valid pada baris {:?}", record);
            0.0
        });
        let no2: f64 = record[4].parse().unwrap_or_else(|_| {
            eprintln!("Peringatan: Nilai no2 tidak valid pada baris {:?}", record);
            0.0
        });

        let kategori = &record[5];
        inputs.push(vec![pm10, so2, co, o3, no2]);
        outputs.push(match kategori {
            "BAIK" => vec![1.0, 0.0, 0.0],
            "SEDANG" => vec![0.0, 1.0, 0.0],
            "TIDAK SEHAT" => vec![0.0, 0.0, 1.0],
            _ => {
                eprintln!("Peringatan: Kategori tidak valid pada baris {:?}", record);
                vec![0.0, 0.0, 0.0]
            }
        });
    }

    let x = Array2::from_shape_vec((inputs.len(), 5), inputs.concat())
        .expect("Gagal membuat array input");
    let y = Array2::from_shape_vec((outputs.len(), 3), outputs.concat())
        .expect("Gagal membuat array output");

    (x, y)
}

pub fn normalize_data(x: &Array2<f64>) -> Array2<f64> {
    let x_mean = x.mean_axis(Axis(0)).expect("Gagal menghitung mean");
    let x_std = x.std_axis(Axis(0), 1.0);
    (x - &x_mean) / &x_std
}