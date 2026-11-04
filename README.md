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

**5.reports/**
Menyimpan visualisasi hasil analisis, preprocessing, evaluasi model, eksperimen, interval prediksi, dan deteksi OOD.

## CARA MENJALANKAN
Disarankan memakai Python 3.12.

git clone https://github.com/lukito789/california-house-price-mlp.git
cd california-house-price-mlp

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

streamlit run backend/streamlit_app.py

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

### Evaluasi Model

Evaluasi dilakukan menggunakan data testing yang tidak digunakan selama proses training maupun validasi.

| Metrik | Hasil |
|---|---:|
| MAE | $33,021.33 |
| RMSE | $49,394.59 |
| R² Score | 0.821 |

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
