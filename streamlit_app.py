import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =========================
# KONFIGURASI LABEL OUTPUT
# =========================
# Kalau hasilnya kebalik, tuker aja:
# 0: "Hujan", 1: "Tidak Hujan"
CLASS_LABELS = {
    0: "Tidak Hujan",
    1: "Hujan",
}

# =========================
# DESKRIPSI FITUR (UNTUK UI)
# =========================
FEATURE_INFO = {
    "TANGGAL": "Tanggal pengamatan (format DD-MM-YYYY). Dipakai untuk ambil Month & Day.",
    "TN": "Suhu minimum harian (°C).",
    "TX": "Suhu maksimum harian (°C).",
    "TAVG": "Suhu rata-rata harian (°C).",
    "RH_AVG": "Kelembapan relatif rata-rata harian (%).",
    "SS": "Lama penyinaran matahari (jam).",
    "FF_X": "Kecepatan angin maksimum (m/s atau knot, sesuai dataset kamu).",
    "DDD_X": "Arah angin saat kecepatan maksimum (derajat 0–360).",
    "FF_AVG": "Kecepatan angin rata-rata (m/s atau knot, sesuai dataset kamu).",
    "DDD_CAR": "Arah angin dominan (kode mata angin: N, NE, E, SE, S, SW, W, NW).",
}

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

def parse_date(date_str: str):
    return datetime.strptime(date_str, "%d-%m-%Y")

def build_features(tanggal, TN, TX, TAVG, RH_AVG, SS, FF_X, DDD_X, FF_AVG, DDD_CAR):
    dt = parse_date(tanggal)
    return pd.DataFrame([{
        "TN": TN,
        "TX": TX,
        "TAVG": TAVG,
        "RH_AVG": RH_AVG,
        "SS": SS,
        "FF_X": FF_X,
        "DDD_X": DDD_X,
        "FF_AVG": FF_AVG,
        "Month": dt.month,
        "Day": dt.day,
        "DDD_CAR": DDD_CAR,
    }])

# =========================
# UI
# =========================
st.set_page_config(page_title="Prediksi Hujan (MLOps)", layout="centered")
st.title("🌧️ Prediksi Hujan / Tidak Hujan")
st.caption("Masukkan data cuaca, lalu klik **Prediksi**. Output akan ditampilkan sebagai label yang mudah dipahami.")

with st.sidebar:
    st.header("📌 Penjelasan Variabel")
    st.write("Berikut arti setiap input yang digunakan model:")
    for k, v in FEATURE_INFO.items():
        st.markdown(f"**{k}** — {v}")

model = load_model()

with st.form("form_prediksi", clear_on_submit=False):
    tanggal = st.text_input("TANGGAL (DD-MM-YYYY)", value="14-12-2025", help=FEATURE_INFO["TANGGAL"])

    c1, c2, c3 = st.columns(3)
    with c1:
        TN = st.number_input("TN", value=0.11, help=FEATURE_INFO["TN"])
        TX = st.number_input("TX", value=0.01, help=FEATURE_INFO["TX"])
        TAVG = st.number_input("TAVG", value=12.00, help=FEATURE_INFO["TAVG"])
    with c2:
        RH_AVG = st.number_input("RH_AVG", value=0.05, help=FEATURE_INFO["RH_AVG"])
        SS = st.number_input("SS", value=0.00, help=FEATURE_INFO["SS"])
        FF_X = st.number_input("FF_X", value=2.08, help=FEATURE_INFO["FF_X"])
    with c3:
        DDD_X = st.number_input("DDD_X", value=0.09, help=FEATURE_INFO["DDD_X"])
        FF_AVG = st.number_input("FF_AVG", value=0.05, help=FEATURE_INFO["FF_AVG"])
        DDD_CAR = st.text_input("DDD_CAR", value="N", help=FEATURE_INFO["DDD_CAR"])

    submitted = st.form_submit_button("🔮 Prediksi")

if submitted:
    try:
        df = build_features(tanggal, TN, TX, TAVG, RH_AVG, SS, FF_X, DDD_X, FF_AVG, DDD_CAR)

        pred = model.predict(df)[0]
        label = CLASS_LABELS.get(int(pred), f"Class {pred}")

        st.subheader("Hasil Prediksi")
        if label == "Hujan":
            st.warning(f"Prediksi: **{label}** ☔")
        else:
            st.success(f"Prediksi: **{label}** 🌤️")

        # Kalau model kamu punya predict_proba, tampilkan confidence biar keren:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            st.write("Probabilitas (confidence):")
            # asumsi kelas 0 & 1
            st.progress(float(max(proba)))
            st.caption(f"Class 0 ({CLASS_LABELS.get(0,'0')}): {proba[0]:.3f} | Class 1 ({CLASS_LABELS.get(1,'1')}): {proba[1]:.3f}")

        with st.expander("Lihat fitur yang dikirim ke model"):
            st.dataframe(df, use_container_width=True)

    except ValueError:
        st.error("Format tanggal salah. Gunakan **DD-MM-YYYY** (contoh: 14-12-2025).")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
