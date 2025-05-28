# Laporan Proyek Machine Learning - Lathif Nurma Huda

## Domain Proyek: Kesehatan Masyarakat

### Latar Belakang

Stunting adalah kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis, terutama pada 1000 hari pertama kehidupan. Menurut UNICEF (2021), stunting memengaruhi perkembangan otak, metabolisme tubuh, serta peningkatan risiko penyakit degeneratif di usia dewasa. Data dari Kementerian Kesehatan Republik Indonesia menunjukkan bahwa prevalensi stunting di Indonesia masih tergolong tinggi. Oleh karena itu, diperlukan pendekatan berbasis data untuk mendeteksi potensi stunting sejak dini.

### Referensi:

- UNICEF. (2021). "Improving Child Nutrition: The achievable imperative for global progress". [https://www.unicef.org/](https://www.unicef.org/)
- Kementerian Kesehatan RI. (2022). "Laporan Status Gizi Balita di Indonesia."

## Business Understanding

### Problem Statements

1. Bagaimana mengidentifikasi status gizi balita berdasarkan data usia, tinggi badan, dan jenis kelamin?
2. Bagaimana performa model machine learning dalam memprediksi stunting dibandingkan dengan baseline model?

### Goals

1. Membangun model klasifikasi untuk memprediksi status gizi (stunting atau tidak) berdasarkan atribut balita.
2. Membandingkan kinerja beberapa algoritma machine learning untuk mendapatkan model terbaik.

### Solution Statements

- Menggunakan algoritma K-Nearest Neighbors, Random Forest, dan Gradient Boosting untuk membangun model klasifikasi.
- Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa.

## Data Understanding

Dataset yang digunakan diperoleh dari Kaggle:
[https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows)

### Variabel:

- `Umur (bulan)`: usia balita dalam bulan (numerik)
- `Jenis Kelamin`: jenis kelamin balita (laki-laki/perempuan)
- `Tinggi Badan (cm)`: tinggi badan balita dalam sentimeter (numerik)
- `Status Gizi`: status gizi balita (stunting/tidak stunting)

### Visualisasi:

- Distribusi usia dan tinggi badan ditampilkan dalam histogram dan boxplot
- Korelasi antar fitur numerik dianalisis dengan scatter plot dan pairplot

## Data Preparation

1. **Mengatasi Missing Value**: Tidak ditemukan nilai yang hilang
2. **Menghapus Duplikat**: Data duplikat dihapus
3. **Mengatasi Outlier**: Menggunakan metode IQR untuk mendeteksi dan menghapus outlier pada kolom numerik
4. **Encoding Fitur Kategorikal**: Menggunakan LabelEncoder pada `Jenis Kelamin` dan `Status Gizi`
5. **Normalisasi Data**: Menggunakan StandardScaler untuk fitur numerik
6. **Split Dataset**: Data dibagi menjadi data latih dan data uji dengan rasio 80:20

## Modeling

Model yang digunakan:

1. **K-Nearest Neighbors (KNN)**

   - Kelebihan: Sederhana, tidak memerlukan asumsi distribusi data
   - Kekurangan: Sensitif terhadap skala data dan outlier

2. **Random Forest Classifier**

   - Kelebihan: Robust terhadap overfitting, interpretabilitas fitur
   - Kekurangan: Komputasi berat untuk dataset besar

3. **Gradient Boosting Classifier**

   - Kelebihan: Akurasi tinggi, performa unggul
   - Kekurangan: Training lebih lambat

### Hyperparameter Tuning

Dilakukan pada model terbaik (Gradient Boosting) menggunakan GridSearchCV untuk meningkatkan akurasi.

## Evaluation

Metrik evaluasi yang digunakan:

- **Accuracy**: Persentase prediksi yang benar
- **Precision, Recall, F1 Score**: Untuk mengevaluasi performa klasifikasi
- **Confusion Matrix**: Untuk menilai prediksi per kelas

### Hasil Evaluasi

Model Gradient Boosting memberikan akurasi tertinggi pada data uji dibandingkan dengan KNN dan Random Forest, dengan nilai F1-score di atas 90%.

## Penutup

Model prediksi status gizi balita berhasil dikembangkan dan memberikan hasil yang menjanjikan. Proyek ini dapat dikembangkan lebih lanjut dengan menambahkan fitur lain seperti berat badan, data ibu, serta kondisi lingkungan.

---

> 📌 **Catatan**: Proyek ini disusun untuk memenuhi submission proyek Machine Learning di Dicoding dan menggunakan dataset orisinal yang belum dipublikasikan sebelumnya. Seluruh proses dari data preparation hingga evaluasi dilakukan sendiri oleh Lathif Nurma Huda, Universitas Sahid Surakarta.
