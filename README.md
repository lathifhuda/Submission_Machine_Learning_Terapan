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
  ![Boxplot_umur](https://github.com/lathifhuda/Submission_Machine_Learning_Terapan/raw/main/boxplot%20umur.png)
  ![Boxplot](https://github.com/lathifhuda/Submission_Machine_Learning_Terapan/raw/main/boxplot.png)
- Korelasi antar fitur divisualisasikan dalam heatmap:

  ![Matriks Korelasi](https://github.com/lathifhuda/Submission_Machine_Learning_Terapan/raw/main/matrik%20korelasi.png)

---

### Ringkasan Dataset Sampel

- Dataset setelah proses sampling memiliki **5000 baris dan 4 kolom**
- Fitur: `Umur (bulan)`, `Jenis Kelamin`, `Tinggi Badan (cm)`, `Status Gizi`
- Tidak ditemukan **missing values**
- Terdapat **312 data duplikat**, yang kemudian dihapus

## Data Preparation

1. **Pengubahan Nama Kolom (Column Renaming)**
Untuk menjaga konsistensi dan kemudahan pemrosesan, semua nama kolom diubah ke format huruf kecil, tanpa spasi, dan diganti dengan underscore (`_`).  
Langkah ini dilakukan dengan kode berikut:
```python
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
```

2. **Menghapus Duplikat**  
   Sebanyak 312 baris duplikat dihapus dari dataset.
   ```
   df.drop_duplicates(inplace=True)
   ```
   Setelah proses ini:
    Jumlah data berkurang dari 5000 menjadi 4688
    Tidak ada lagi baris duplikat yang tersisa
   ```
   (df.duplicated().sum() == 0)
   ```

4. **Mengatasi Outlier**  
   Deteksi dan penghapusan outlier pada fitur numerik (`Umur`, `Tinggi Badan`) menggunakan metode IQR.
   ```
   def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
   ```
   Setelah menghapus outlier:
   Jumlah data menjadi 4688 baris
   Nilai ekstrim yang dapat mengganggu model telah dihilangkan
    

6. **Encoding Fitur Kategorikal**  
   Menggunakan `LabelEncoder` pada `Jenis Kelamin` dan `Status Gizi`.
   ```
   from sklearn.preprocessing import LabelEncoder
   le_jenis_kelamin = LabelEncoder()
   le_stunting = LabelEncoder()
   df['jenis_kelamin'] = le_jenis_kelamin.fit_transform(df['jenis_kelamin'])
   df['stunting'] = le_stunting.fit_transform(df['stunting'])
   ```
   Fitur kategorikal diubah ke bentuk numerik menggunakan LabelEncoder, agar dapat digunakan dalam model machine learning


8. **Split Dataset**  
   Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20.
   ```
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
   Dataset dibagi menjadi:
   Data Latih (Train): 3750 sampel
   Data Uji (Test): 938 sampel
   Menggunakan train_test_split dengan rasio 80:20.


10. **Normalisasi Data**  
   Menggunakan `StandardScaler` pada fitur numerik.  
   Normalisasi dilakukan setelah split, dan scaler **di-fit pada data latih saja** untuk mencegah data leakage.
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```




---


## Modeling

Model yang digunakan:

### 1. K-Nearest Neighbors (KNN)

- **Cara kerja**: Menentukan kelas berdasarkan mayoritas label dari tetangga terdekat (k) berdasarkan jarak Euclidean.
- **Parameter**: `n_neighbors=5` (default)

### 2. Random Forest Classifier

- **Cara kerja**: Membentuk banyak pohon keputusan dan menggabungkan hasilnya melalui voting.
- **Parameter**: `n_estimators=100`, `random_state=42`

### 3. Gradient Boosting Classifier

- **Cara kerja**: Membangun model secara iteratif dengan fokus pada kesalahan model sebelumnya.
- **Parameter**: `n_estimators=100`, `learning_rate=0.1`, `random_state=42`

---

## Evaluation

### Metrik yang Digunakan

- **Accuracy**: Persentase prediksi yang benar

### Hasil Evaluasi Model (Data Uji)

| Model                        | Akurasi Data Latih | Akurasi Data Uji |
| ---------------------------- | ------------------ | ---------------- |
| K-Nearest Neighbors (KNN)    | 97.36%             | 95.42%           |
| Random Forest Classifier     | 100%               | 95.84%           |
| Gradient Boosting Classifier | 96.19%             | 92.64%           |


- Model Random Forest menunjukkan performa terbaik pada data uji.

---


### Hasil Evaluasi

![Perbandingan hasil model](https://github.com/lathifhuda/Submission_Machine_Learning_Terapan/raw/main/evaluasi%20model.png)



> 📌 **Catatan**: Proyek ini disusun untuk memenuhi submission proyek Machine Learning di Dicoding. Seluruh proses dari data preparation hingga evaluasi dilakukan sendiri oleh Lathif Nurma Huda, Universitas Sahid Surakarta.
