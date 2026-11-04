"""Aplikasi Streamlit: Estimasi Harga Rumah California (MLP regresi) dengan dua fitur unik.

  Fitur unik 1  Interval prediksi (split conformal prediction)         -> notebook 05
  Fitur unik 2  Deteksi input di luar domain (jarak Mahalanobis)       -> notebook 06

Model dan preprocessing sama dengan API (backend/app.py), sehingga hasil prediksinya identik.
Lokal (dari folder utama proyek):   streamlit run backend/streamlit_app.py
Streamlit Community Cloud:          Main file path = backend/streamlit_app.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent          # folder backend/
ROOT = BASE.parent                              # folder utama proyek
sys.path.insert(0, str(BASE))
from preprocess import NUMERIC, TARGET, TARGET_SCALE, transform   # dipakai bersama API dan notebook

ART = BASE / "artifacts"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"

st.set_page_config(page_title="Estimasi Harga Rumah California", layout="wide")

# ---------------------------------------------------------------------------------------------------------------
# Aturan input (sama dengan backend/app.py::HouseInput) dan contoh cepat
# ---------------------------------------------------------------------------------------------------------------
OCEAN = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
FIELDS = {  # nama: (label, min, max, langkah, format)
    "latitude": ("Latitude", 32.0, 43.0, 0.01, "%.2f"),
    "longitude": ("Longitude", -125.0, -113.0, 0.01, "%.2f"),
    "housing_median_age": ("Umur median rumah (tahun)", 1.0, 100.0, 1.0, "%.0f"),
    "total_rooms": ("Total kamar", 1.0, 50000.0, 10.0, "%.0f"),
    "total_bedrooms": ("Total kamar tidur", 1.0, 10000.0, 10.0, "%.0f"),
    "population": ("Populasi", 1.0, 40000.0, 10.0, "%.0f"),
    "households": ("Jumlah rumah tangga", 1.0, 10000.0, 10.0, "%.0f"),
    "income_usd": ("Pendapatan median (USD/tahun)", 1000.0, 200000.0, 500.0, "%.0f"),
}
BLOK = dict(housing_median_age=30.0, total_rooms=2500.0, total_bedrooms=500.0, population=1200.0,
            households=450.0, income_usd=55000.0)          # nilai awal yang sama dengan frontend/index.html


def _kota(lat, lon, ocean):
    return dict(latitude=lat, longitude=lon, ocean_proximity=ocean, **BLOK)


PRESETS = {
    "Isi manual": None,
    "Los Angeles": _kota(34.05, -118.24, "<1H OCEAN"),
    "San Francisco": _kota(37.77, -122.42, "NEAR BAY"),
    "San Diego": _kota(32.72, -117.16, "NEAR OCEAN"),
    "Sacramento": _kota(38.58, -121.49, "INLAND"),
    "Fresno": _kota(36.74, -119.79, "INLAND"),
    "Uji fitur 2: titik di laut lepas": _kota(34.0, -124.0, "NEAR OCEAN"),
    "Uji fitur 2: rumah tangga 3x penduduk": {**_kota(34.05, -118.24, "<1H OCEAN"), "households": 3600.0},
    "Uji fitur 2: kamar tidur 2x total kamar": {**_kota(34.05, -118.24, "<1H OCEAN"), "total_bedrooms": 5000.0},
}
DEFAULT = PRESETS["Los Angeles"]


# ---------------------------------------------------------------------------------------------------------------
# Memuat model dan parameter fitur unik (sekali saja)
# ---------------------------------------------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def skor_mahalanobis(x, ood):
    d = x - np.array(ood["mean"])
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, np.array(ood["precision"]), d), 0.0))


@st.cache_resource(show_spinner="Memuat model dan parameter...")
def muat_sumber():
    wajib = [ART / "model.keras", ART / "prep.json", PROC / "conformal.json", PROC / "ood.json"]
    hilang = [str(p.relative_to(ROOT)) for p in wajib if not p.exists()]
    if hilang:
        return {"hilang": hilang}
    from tensorflow import keras
    model = keras.models.load_model(ART / "model.keras")
    prep = json.loads((ART / "prep.json").read_text())
    conf = json.loads((PROC / "conformal.json").read_text())
    ood = json.loads((PROC / "ood.json").read_text())
    metrics = json.loads((ART / "metrics.json").read_text()) if (ART / "metrics.json").exists() else {}

    peringatan = []
    if conf["model_sha256"] != sha256(ART / "model.keras"):
        peringatan.append("model.keras berbeda dari model yang dipakai saat kalibrasi interval (conformal.json). "
                          "Jalankan ulang notebook 05 agar jaminan cakupan interval tetap berlaku.")
    if ood["prep_sha256"] != sha256(ART / "prep.json"):
        peringatan.append("prep.json berbeda dari yang dipakai detektor (ood.json). Jalankan ulang notebook 06.")
    if conf["metode"] not in ("absolut", "relatif"):
        peringatan.append(f"Metode interval '{conf['metode']}' belum didukung aplikasi ini.")

    # Sebaran skor data nyata, hanya untuk grafik pembanding
    ref = None
    data_path = ROOT / "data" / "housing.csv"
    if data_path.exists():
        df = pd.read_csv(data_path).dropna(subset=[TARGET, "ocean_proximity"])
        ref = skor_mahalanobis(transform(df, prep)[:, : len(ood["fitur_numerik"])], ood)
    return {"model": model, "prep": prep, "conf": conf, "ood": ood, "metrics": metrics,
            "peringatan": peringatan, "ref": ref}


# ---------------------------------------------------------------------------------------------------------------
# Logika prediksi dan dua fitur unik
# ---------------------------------------------------------------------------------------------------------------
def prediksi_usd(S, df):
    x = transform(df, S["prep"])
    return max(float(S["model"](x, training=False).numpy().ravel()[0] * TARGET_SCALE), 0.0)


def interval_prediksi(y_hat, conf, alpha):
    """Fitur unik 1. Salinan dari interval_from_conf (notebook 05), tanpa metode MC-Dropout."""
    q = conf["q"][alpha]
    half = q if conf["metode"] == "absolut" else q * max(y_hat, conf["floor_usd"])
    return max(y_hat - half, 0.0), y_hat + half


def cek_domain(S, df):
    """Fitur unik 2. Sama dengan cek_domain_dari_params (notebook 06) ditambah uraian penyebab."""
    ood = S["ood"]
    x = transform(df, S["prep"])[:, : len(ood["fitur_numerik"])]
    skor = float(skor_mahalanobis(x, ood)[0])
    luar_rentang = bool(((x < np.array(ood["min"])) | (x > np.array(ood["max"]))).any())
    if ood["metode"] == "rentang":
        flag = luar_rentang
    else:
        flag = skor > ood["ambang"]
        if ood["metode"] == "gabungan":
            flag = flag or luar_rentang
    d = x - np.array(ood["mean"])
    top = np.argsort(-(d * (d @ np.array(ood["precision"])))[0])[:2]
    return {"skor": skor, "ambang": float(ood["ambang"]), "di_luar": bool(flag),
            "penyebab": [ood["fitur_numerik"][j] for j in top]}


def usd(v):
    return f"${v:,.0f}"


def tampil_grafik(chart):
    """Kompatibel dengan Streamlit lama (use_container_width) dan baru (width='stretch')."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def grafik_interval(lo, hi, pred):
    df = pd.DataFrame({"bawah": [lo], "atas": [hi], "prediksi": [pred]})
    sumbu = alt.X("bawah:Q", title="Harga (USD)", scale=alt.Scale(domain=[0, max(hi * 1.1, 100_000)]),
                  axis=alt.Axis(format="$,.0f"))
    batang = alt.Chart(df).mark_bar(height=22, color="#9ecae1").encode(x=sumbu, x2="atas:Q")
    titik = alt.Chart(df).mark_point(size=260, filled=True, color="#1f4e79").encode(x="prediksi:Q")
    return (batang + titik).properties(height=90)


def grafik_skor(ref, skor, ambang):
    xmax = max(15.0, min(max(skor * 1.15, ambang * 1.6), 40.0))
    bins = np.linspace(0, xmax, 41)
    jumlah, _ = np.histogram(np.clip(ref, 0, xmax), bins=bins)
    h = pd.DataFrame({"mulai": bins[:-1], "selesai": bins[1:], "jumlah": jumlah})
    hist = alt.Chart(h).mark_bar(color="#c6dbef").encode(
        x=alt.X("mulai:Q", title="skor jarak Mahalanobis"), x2="selesai:Q",
        y=alt.Y("jumlah:Q", title="jumlah blok data nyata"))
    garis = pd.DataFrame({"x": [ambang, min(skor, xmax)], "ket": ["ambang", "input Anda"]})
    rule = alt.Chart(garis).mark_rule(strokeWidth=3).encode(
        x="x:Q", color=alt.Color("ket:N", legend=alt.Legend(title=None, orient="top"),
                                 scale=alt.Scale(domain=["ambang", "input Anda"], range=["#444444", "#d62728"])))
    return (hist + rule).properties(height=230)


# ---------------------------------------------------------------------------------------------------------------
# Antarmuka
# ---------------------------------------------------------------------------------------------------------------
S = muat_sumber()
if "hilang" in S:
    st.title("Estimasi Harga Rumah California")
    st.error("Berkas berikut belum ada, sehingga aplikasi belum bisa dijalankan:")
    for h in S["hilang"]:
        st.code(h)
    st.markdown("- `backend/artifacts/*`: jalankan `python backend/train.py` (atau `git pull`).\n"
                "- `data/processed/ood.json`: jalankan notebook 06.\n"
                "- `data/processed/conformal.json`: jalankan notebook 05.")
    st.stop()

for kunci, nilai in DEFAULT.items():                       # nilai awal widget (sekali per sesi)
    st.session_state.setdefault(f"in_{kunci}", nilai)
st.session_state.setdefault("riwayat", [])


def terapkan_preset():
    p = PRESETS[st.session_state["preset"]]
    if p:
        for k, v in p.items():
            st.session_state[f"in_{k}"] = v


with st.sidebar:
    st.header("Tentang aplikasi")
    st.write("MLP regresi untuk memperkirakan **harga median rumah** pada sebuah blok sensus California, "
             "dengan dua fitur tambahan yang menilai seberapa pantas prediksinya dipercaya.")
    st.subheader("Status berkas")
    if S["peringatan"]:
        for w in S["peringatan"]:
            st.warning(w)
    else:
        st.success("Model, preprocessing, interval, dan detektor saling cocok (SHA-256 terverifikasi).")
    st.caption("Data: sensus California 1990. Hasil bukan penilaian harga rumah sungguhan.")

st.title("Estimasi Harga Rumah California")
tab_pred, tab_fitur, tab_model, tab_riwayat = st.tabs(["Prediksi", "Dua fitur unik", "Model dan data", "Riwayat sesi"])

# ------------------------------------------------------------------ Tab 1: prediksi
with tab_pred:
    st.selectbox("Contoh cepat", list(PRESETS), key="preset", on_change=terapkan_preset,
                 help="Pilih kota untuk mengisi lokasi, atau pilih salah satu 'Uji fitur 2' untuk melihat input tidak wajar.")
    with st.form("form_prediksi"):
        kiri, kanan = st.columns(2)
        for i, (nama, (label, lo, hi, step, fmt)) in enumerate(FIELDS.items()):
            with (kiri if i % 2 == 0 else kanan):
                st.number_input(label, min_value=lo, max_value=hi, step=step, format=fmt, key=f"in_{nama}")
        with kiri:
            st.selectbox("Jarak ke laut (ocean_proximity)", OCEAN, key="in_ocean_proximity")
        kirim = st.form_submit_button("Prediksi harga", type="primary")

    if kirim:
        v = {k: float(st.session_state[f"in_{k}"]) for k in FIELDS}
        baris = {**{k: v[k] for k in NUMERIC if k in v}, "median_income": v["income_usd"] / 10000,
                 "ocean_proximity": st.session_state["in_ocean_proximity"]}
        df = pd.DataFrame([baris])
        y_hat = prediksi_usd(S, df)
        dom = cek_domain(S, df)
        st.session_state["hasil"] = {"baris": baris, "y_hat": y_hat, "dom": dom}
        a90 = interval_prediksi(y_hat, S["conf"], "0.1")
        st.session_state["riwayat"].append({
            "waktu": datetime.now().strftime("%H:%M:%S"), "lokasi": f"{baris['latitude']:.2f}, {baris['longitude']:.2f}",
            "prediksi_usd": round(y_hat), "interval_90_bawah": round(a90[0]), "interval_90_atas": round(a90[1]),
            "status_domain": "DI LUAR DOMAIN" if dom["di_luar"] else "NORMAL"})

    h = st.session_state.get("hasil")
    if h:
        y_hat, dom, baris = h["y_hat"], h["dom"], h["baris"]
        st.divider()
        st.metric("Prediksi harga median blok", usd(y_hat))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fitur unik 1: interval prediksi")
            tingkat = st.radio("Tingkat keyakinan", ["80%", "90%", "95%"], index=1, horizontal=True, key="tingkat")
            kunci = {"80%": "0.2", "90%": "0.1", "95%": "0.05"}[tingkat]
            lo, hi = interval_prediksi(y_hat, S["conf"], kunci)
            st.write(f"Dengan keyakinan **{tingkat}**, harga sebenarnya diperkirakan antara **{usd(lo)}** dan **{usd(hi)}**.")
            tampil_grafik(grafik_interval(lo, hi, y_hat))
            st.caption(f"Lebar rentang {usd(hi - lo)}. Pada data uji, rentang 90% mencakup "
                       f"{S['conf']['cakupan_test']['cakupan'] * 100:.1f}% harga sebenarnya.")
        with col2:
            st.subheader("Fitur unik 2: deteksi di luar domain")
            if dom["di_luar"]:
                st.error(f"DI LUAR DOMAIN. Kombinasi nilai ini tidak lazim dibanding data latih "
                         f"(terutama pada {' dan '.join(dom['penyebab'])}). Angka prediksi hanya hasil ekstrapolasi.")
            else:
                st.success("NORMAL. Kombinasi nilai ini mirip dengan blok sensus yang dipelajari model.")
            m1, m2 = st.columns(2)
            m1.metric("Skor input", f"{dom['skor']:.2f}")
            m2.metric("Ambang", f"{dom['ambang']:.2f}")
            if S["ref"] is not None:
                tampil_grafik(grafik_skor(S["ref"], dom["skor"], dom["ambang"]))
                if dom["skor"] > 40:
                    st.caption("Skor input melampaui skala grafik.")

        if dom["di_luar"]:
            st.warning("Jaminan cakupan interval hanya berlaku untuk input yang mirip data latih. "
                       "Untuk input di luar domain, jangan memakai rentang di atas sebagai patokan.")

        peta, ringkas = st.columns([1, 1])
        with peta:
            st.caption("Lokasi")
            st.map(pd.DataFrame({"latitude": [baris["latitude"]], "longitude": [baris["longitude"]]}), zoom=5)
        with ringkas:
            st.caption("Input yang dipakai model")
            st.dataframe(pd.DataFrame({"nilai": baris}).astype(str))

# ------------------------------------------------------------------ Tab 2: penjelasan fitur unik
with tab_fitur:
    c = S["conf"]; o = S["ood"]
    st.subheader("Fitur unik 1: interval prediksi (split conformal prediction)")
    st.markdown(
        f"**Masalah.** Model hanya mengeluarkan satu angka, padahal galatnya berbeda-beda antar blok.\n\n"
        f"**Cara kerja.** Selisih relatif antara prediksi dan harga asli dihitung pada {c['n_kalibrasi']:,} blok **validation** "
        f"(data yang tidak dipakai melatih). Persentil tertentu dari selisih itu menjadi lebar rentang: "
        f"setengah-lebar = `q × max(prediksi, {c['floor_usd']:,.0f})`, dengan `q = {c['q']['0.1']:.4f}` untuk keyakinan 90%. "
        f"Metode: `{c['metode']}`.\n\n"
        f"**Bukti.** Pada data uji, rentang 90% mencakup **{c['cakupan_test']['cakupan'] * 100:.1f}%** harga sebenarnya, "
        f"dengan lebar rata-rata {usd(c['cakupan_test']['lebar_rata2_usd'])}.\n\n"
        f"**Batasan.** Jaminan berlaku untuk cakupan *rata-rata*, bukan tiap kelompok harga, dan hanya untuk input yang sebaran-nya mirip data latih. "
        f"Rentangnya lebar (sekitar ±{usd(c['cakupan_test']['lebar_rata2_usd'] / 2)}), sesuai besarnya galat model.")
    for gambar in sorted(FIG.glob("19_*.png"))[:1]:
        st.image(str(gambar), caption="Kalibrasi interval pada data uji (notebook 05)")
    st.divider()
    st.subheader("Fitur unik 2: deteksi input di luar domain (jarak Mahalanobis)")
    st.markdown(
        f"**Masalah.** Model tetap memberi harga yang terlihat wajar untuk input yang tidak mungkin ada, misalnya titik di laut lepas. "
        f"API hanya memeriksa rentang tiap fitur, bukan kombinasinya.\n\n"
        f"**Cara kerja.** Jarak Mahalanobis input terhadap sebaran 8 fitur numerik data latih (memperhitungkan korelasi antar-fitur). "
        f"Ditandai di luar domain bila skor > **{o['ambang']:.2f}** (persentil ke-{o['persentil']:g} data latih).\n\n"
        f"**Bukti (notebook 06).** Data nyata yang salah ditandai sekitar {o['fpr_validation_persen']:.1f}% pada validation. "
        f"Pada input tidak wajar yang lolos validasi API: titik di laut lepas, titik di Idaho, rumah tangga 3× penduduk, "
        f"dan kamar tidur 2× total kamar terdeteksi **100%**; kolom yang diacak independen sekitar 61%. "
        f"Coba semuanya lewat 'Contoh cepat' di tab Prediksi.\n\n"
        f"**Batasan.** Pergeseran halus (misalnya lintang +1,5 derajat) dan ketidakcocokan kategori dengan lokasi (misalnya `ISLAND` di daratan) "
        f"hampir tidak terdeteksi. Fitur ini memberi peringatan, tidak memblokir prediksi.")
    for gambar in sorted(FIG.glob("23_*.png"))[:1]:
        st.image(str(gambar), caption="Deteksi di luar domain per skenario (notebook 06)")

# ------------------------------------------------------------------ Tab 3: model dan data
with tab_model:
    m = S["metrics"]
    st.subheader("Model")
    st.write(f"MLP regresi dengan {S['model'].count_params():,} parameter: 13 fitur masuk, lapisan tersembunyi 128, 64, 32 (ReLU), "
             f"Dropout 0,2, regularisasi L2 1e-4, satu keluaran linear.")
    if m:
        a, b, c3 = st.columns(3)
        a.metric("MAE uji", usd(m["test_mae_usd"]))
        b.metric("RMSE uji", usd(m["test_rmse_usd"]))
        c3.metric("R² uji", f"{m['test_r2']:.3f}")
        st.caption(f"Pembagian data: {m['n_train']:,} latih, {m['n_val']:,} validation, {m['n_test']:,} uji.")
    with st.expander("Ringkasan arsitektur"):
        baris_ringkas = []
        S["model"].summary(print_fn=lambda s, **k: baris_ringkas.append(s))
        st.code("\n".join(baris_ringkas))
    st.subheader("Catatan penggunaan")
    st.markdown("- Target **terpotong di \\$500.001** pada data asli, sehingga rumah mahal cenderung ditaksir terlalu rendah.\n"
                "- Harga adalah **median blok sensus 1990**, bukan harga satu rumah.\n"
                "- Pendapatan dimasukkan dalam USD per tahun, lalu dikonversi ke satuan \\$10.000 seperti data latih.")

# ------------------------------------------------------------------ Tab 4: riwayat
with tab_riwayat:
    if st.session_state["riwayat"]:
        st.dataframe(pd.DataFrame(st.session_state["riwayat"]))
        if st.button("Hapus riwayat"):
            st.session_state["riwayat"] = []
            st.rerun()
    else:
        st.info("Belum ada prediksi pada sesi ini.")
