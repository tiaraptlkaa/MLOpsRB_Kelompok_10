from datetime import datetime
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Cuaca", page_icon="🌦️")

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

model = load_model()

st.title("Prediksi Class (MLOps)")
st.caption("Input data lalu klik Predict")

# ---- Form input ----
with st.form("predict_form"):
    tanggal = st.text_input("TANGGAL (DD-MM-YYYY)", value="14-12-2025")

    col1, col2, col3 = st.columns(3)
    with col1:
        TN = st.number_input("TN", value=0.0)
        TX = st.number_input("TX", value=0.0)
        TAVG = st.number_input("TAVG", value=0.0)
    with col2:
        RH_AVG = st.number_input("RH_AVG", value=0.0)
        SS = st.number_input("SS", value=0.0)
        FF_X = st.number_input("FF_X", value=0.0)
    with col3:
        DDD_X = st.number_input("DDD_X", value=0.0)
        FF_AVG = st.number_input("FF_AVG", value=0.0)
        DDD_CAR = st.text_input("DDD_CAR", value="N")

    submitted = st.form_submit_button("Predict")

if submitted:
    # validasi tanggal
    try:
        dt = datetime.strptime(tanggal, "%d-%m-%Y")
    except ValueError:
        st.error("TANGGAL harus format DD-MM-YYYY")
        st.stop()

    features = {
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
    }

    df = pd.DataFrame([features])

    try:
        pred = model.predict(df)
        st.success(f"Predicted class: {int(pred[0])}")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Gagal predict: {e}")



