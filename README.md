## Dataset

- **Nama:** California Housing Prices
- **Link dataset:** [Kaggle - California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **File training:** `data/housing.csv`
- **Sumber asli:** Data sensus penduduk California tahun 1990.
- **Lisensi:** CC0: Public Domain.
- **Ukuran:** 20.640 baris, 10 kolom, sekitar 1,42 MB.

Dataset berisi informasi karakteristik perumahan berdasarkan wilayah sensus di California. Variabel yang tersedia meliputi lokasi geografis, usia rumah, jumlah kamar, populasi, jumlah rumah tangga, pendapatan median, kedekatan dengan laut, dan nilai median rumah.

Target prediksi adalah `median_house_value`.

## Struktur Repository

**1.backend/**
- app.py: Menyediakan REST API menggunakan FastAPI untuk melakukan prediksi harga rumah.
- streamlit_app.py: Menyediakan antarmuka aplikasi interaktif menggunakan Streamlit.
- database.py: Mengelola penyimpanan dan riwayat prediksi menggunakan SQLite.
- preprocess.py: Menangani proses preprocessing yang digunakan saat training dan prediksi.
- train.py: Menjalankan proses pelatihan model MLP.
- requirements.txt: Berisi daftar library Python yang dibutuhkan.
- artifacts/: Menyimpan model final, parameter preprocessing, dan metrik evaluasi.

**2.dataset/**
- housing.csv: Dataset utama California Housing Prices.
- processed/: Menyimpan dataset hasil preprocessing, parameter deteksi OOD, konfigurasi Conformal Prediction, serta model dan metrik dari eksperimen sebelumnya.

**3.frontend/**
- index.html: Struktur halaman aplikasi.
- style.css: Mengatur tampilan dan desain antarmuka.
- script.js: Mengatur interaksi pengguna dengan aplikasi dan API.

**4.notebooks/**
- 01_eda.ipynb: Exploratory Data Analysis untuk memahami karakteristik dataset.
- 02_preprocessing.ipynb: Melakukan pembersihan dan transformasi data.
- 03_model_mlp.ipynb: Membangun dan mengevaluasi model MLP awal.
- 04_eksperimen_optimasi.ipynb: Melakukan eksperimen dan pemilihan konfigurasi model final.
- 05_fitur_unik_interval_prediksi.ipynb: Mengembangkan Conformal Prediction untuk menghasilkan interval prediksi.
- 06_fitur_unik_deteksi_ood.ipynb: Mengembangkan deteksi input di luar distribusi data training.
- 07_evaluasi_final.ipynb: berisi hasil evaluasi final

**5.reports/**
Menyimpan visualisasi hasil analisis, preprocessing, evaluasi model, eksperimen, interval prediksi, dan deteksi OOD.

**6.docs/**
Laporan Estimasi Harga Rumah dengan MLP Regresi.docx: Laporan proyek dalam format Word.
Laporan Estimasi Harga Rumah dengan MLP Regresi.pdf: Laporan proyek dalam format PDF.


## CARA MENJALANKAN
Disarankan memakai Python 3.12.

```bash
git clone https://github.com/lukito789/california-house-price-mlp.git
cd california-house-price-mlp

python -m venv .venv

.venv\Scripts\activate
# Windows

source .venv/bin/activate
# MacOS/Linux

pip install -r requirements.txt

```
Buka folder repository di VSCode, pilih kernel .venv, lalu jalankan notebook di folder notebooks/ secara berurutan.

## CARA MENJALANKAN APLIKASI streamlit_app.py
```
streamlit run backend/streamlit_app.py
```

**Link aplikasi**
Live Application:
[California House Price Prediction](https://california-house-price-mlp.streamlit.app/)

## RINGKASAN PREPROCESSING

Preprocessing dilakukan untuk mempersiapkan dataset sebelum digunakan dalam pelatihan model MLP.

Tahapan preprocessing meliputi:

1. **Pemeriksaan Data**
   - Memeriksa tipe data, distribusi fitur, dan nilai yang hilang.
   - Menghapus baris yang tidak memiliki nilai target `median_house_value` atau kategori `ocean_proximity`.

2. **Penanganan Missing Values**
   - Mengisi nilai yang hilang pada fitur numerik menggunakan median dari data training.

3. **Log Transformation**
   - Menggunakan transformasi `log1p` pada fitur:
     - `total_rooms`
     - `total_bedrooms`
     - `population`
     - `households`

4. **Encoding Data Kategorikal**
   - Menggunakan One-Hot Encoding pada fitur `ocean_proximity`.
   - Menghasilkan lima kategori fitur biner.

5. **Standardization**
   - Melakukan standardisasi fitur menggunakan nilai mean dan standard deviation dari data training.

6. **Pembagian Dataset**
   - Training: 70% (14.447 data)
   - Validation: 15% (3.096 data)
   - Testing: 15% (3.097 data)
   - Random seed: 42.

7. **Normalisasi Target**
   - Target `median_house_value` dibagi dengan 100.000 sebelum proses training.
   - Hasil prediksi dikembalikan ke satuan USD.

Hasil akhir preprocessing menghasilkan 13 fitur input yang digunakan oleh model MLP.

## MODEL FINAL

Model final yang digunakan adalah **Multi-Layer Perceptron (MLP) Regression** berbasis TensorFlow/Keras.

Model menggunakan tiga hidden layers dengan konfigurasi sebagai berikut:

| Komponen | Konfigurasi |
|---|---|
| Input Layer | 13 fitur |
| Hidden Layer 1 | 128 neuron, ReLU |
| Dropout | 0.2 |
| Hidden Layer 2 | 64 neuron, ReLU |
| Dropout | 0.2 |
| Hidden Layer 3 | 32 neuron, ReLU |
| Output Layer | 1 neuron, Linear |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Mean Squared Error (MSE) |
| Regularization | L2 = 0.0001 |
| Batch Size | 64 |
| Random Seed | 42 |


## HASIL EVALUASI FINAL

Evaluasi dilakukan untuk mengukur kemampuan model Multi-Layer Perceptron (MLP) dalam memprediksi harga median rumah di California. Pengujian menggunakan data testing yang tidak digunakan selama proses training maupun pemilihan model.

### 1. Metrik Evaluasi

Penelitian ini menggunakan tiga metrik evaluasi regresi, yaitu Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R-Squared (R²).

| Metrik | Hasil Testing | Keterangan |
|---|---:|---|
| MAE | $33,021.33 | Rata-rata kesalahan absolut prediksi |
| RMSE | $49,394.59 | Mengukur kesalahan dengan penalti lebih besar pada error tinggi |
| R² Score | 0.8212 | Model menjelaskan sekitar 82.12% variasi target |

**Interpretasi:**

- MAE sebesar $33,021.33 menunjukkan rata-rata selisih absolut antara harga aktual dan hasil prediksi model.
- RMSE sebesar $49,394.59 menunjukkan bahwa kesalahan prediksi yang besar memberikan pengaruh lebih tinggi terhadap nilai evaluasi.
- R² sebesar 0.8212 menunjukkan bahwa model mampu menjelaskan sekitar 82.12% variasi harga median rumah pada data testing.

Accuracy, Precision, Recall, dan F1-Score tidak digunakan karena proyek ini merupakan permasalahan regresi dengan target berupa nilai harga kontinu.

### 2. Pembagian Dataset

Dataset dibagi menjadi tiga bagian untuk mendukung proses training, validasi, dan pengujian model.

| Dataset | Jumlah Data | Persentase |
|---|---:|---:|
| Training | 14,447 | 70% |
| Validation | 3,096 | 15% |
| Testing | 3,097 | 15% |
| **Total** | **20,640** | **100%** |

Data training digunakan untuk melatih model. Data validation digunakan untuk memantau performa dan membantu pemilihan model. Data testing digunakan untuk mengevaluasi kemampuan generalisasi model pada data yang belum pernah digunakan sebelumnya.

### 3. Perbandingan Skenario Eksperimen

Eksperimen dilakukan pada Notebook 04 dengan pendekatan One-Factor-at-a-Time (OFAT). Setiap eksperimen mengubah satu faktor dari konfigurasi baseline.

| Skenario | Validation MAE |
|---|---:|
| Baseline | $32,371 |
| Dropout 0.1 | $32,034 |
| Arsitektur 256-128-64 | $31,535 |
| Target Log Transformation | $31,442 |
| Penambahan Fitur Rasio | $32,605 |
| Learning Rate 0.01 | $35,798 |
| Learning Rate 0.0003 | $34,975 |

Hasil screening menunjukkan bahwa beberapa konfigurasi menghasilkan MAE validation yang lebih rendah daripada baseline. Namun, pemilihan model final juga mempertimbangkan variasi hasil antar-seed dan ambang peningkatan performa.

Berdasarkan protokol eksperimen tersebut, konfigurasi baseline tetap dipilih sebagai model final karena kandidat lain tidak memenuhi kriteria peningkatan yang ditetapkan.

### 4. Grafik Loss dan Performa Training-Validation

Evaluasi proses pembelajaran menggunakan grafik Training Loss dan Validation Loss serta MAE Training dan Validation.

Grafik digunakan untuk:
- Mengamati perubahan kesalahan model selama proses training.
- Membandingkan performa data training dan validation.
- Mengidentifikasi indikasi overfitting atau underfitting.
- Memeriksa kemampuan model dalam melakukan generalisasi.

Karena model menggunakan regresi, MAE digunakan sebagai pendamping grafik loss, bukan accuracy.

### 5. Fitur Tambahan: Conformal Prediction

Model dilengkapi dengan Split Conformal Prediction untuk menghasilkan interval prediksi, bukan hanya satu estimasi harga.

| Parameter | Hasil |
|---|---:|
| Metode | Relative Conformal Prediction |
| Tingkat interval utama | 90% |
| Alpha | 0.10 |
| Data kalibrasi | 3,096 |
| Cakupan pada data testing | 89.80% |
| Lebar interval rata-rata | $145,836.25 |

Conformal Prediction menghasilkan rentang estimasi harga yang membantu pengguna memahami ketidakpastian prediksi.

Aplikasi menyediakan tiga tingkat interval, yaitu 80%, 90%, dan 95%.

### 6. Fitur Tambahan: Out-of-Distribution Detection

Aplikasi menggunakan metode Mahalanobis Distance untuk mendeteksi input yang memiliki karakteristik berbeda dari distribusi data training.

| Parameter | Hasil |
|---|---:|
| Metode | Mahalanobis Distance |
| Persentil threshold | 99 |
| Nilai threshold | 7.2640 |
| False Positive Rate Validation | 0.84% |

Deteksi OOD membantu mengidentifikasi input yang tidak lazim dibandingkan data yang digunakan untuk melatih model.

Jika nilai jarak Mahalanobis melebihi threshold, aplikasi memberikan peringatan bahwa input berada di luar domain data training. Hasil prediksi pada kondisi tersebut perlu ditafsirkan dengan hati-hati.

### 7. Kesimpulan Evaluasi

Berdasarkan hasil pengujian, model MLP final memperoleh MAE sebesar $33,021.33, RMSE sebesar $49,394.59, dan R² sebesar 0.8212 pada data testing.

Model juga dilengkapi dengan Conformal Prediction untuk memberikan interval estimasi harga dan Out-of-Distribution Detection untuk mengidentifikasi input yang tidak sesuai dengan distribusi data training.

Dengan demikian, aplikasi tidak hanya menghasilkan estimasi harga rumah, tetapi juga menyediakan informasi mengenai ketidakpastian prediksi dan karakteristik input yang diberikan pengguna.

**Catatan:** Model memprediksi median harga rumah pada blok sensus berdasarkan data California tahun 1990. Hasil prediksi bukan estimasi harga pasar properti terkini.

### Fitur Tambahan Model

Selain menghasilkan estimasi harga rumah, aplikasi menyediakan dua fitur tambahan:

**1. Conformal Prediction**

Menggunakan metode Split Conformal Prediction untuk menghasilkan interval prediksi pada tingkat 80%, 90%, dan 95%.

Pada tingkat 90%, hasil evaluasi data testing menunjukkan cakupan sebesar 89.8%.

**2. Out-of-Distribution (OOD) Detection**

Menggunakan jarak Mahalanobis untuk mendeteksi input yang memiliki karakteristik berbeda dari distribusi data training.

Ambang deteksi menggunakan persentil ke-99 dengan nilai threshold sekitar 7.264.

### Model Artifact

Model dan konfigurasi preprocessing disimpan pada:

- `backend/artifacts/model.keras`
- `backend/artifacts/prep.json`
- `backend/artifacts/metrics.json`

Konfigurasi Conformal Prediction dan OOD Detection disimpan pada:

- `data/processed/conformal.json`
- `data/processed/ood.json`
