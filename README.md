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
