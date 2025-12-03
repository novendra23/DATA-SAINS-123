import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn

# =========================
# CUSTOM STREAMLIT UI THEME
# =========================
st.markdown("""
    <style>
        .main {
            background-color: #0f1117;
            color: white;
        }
        .css-1d391kg {
            background-color: #0f1117;
        }
        .card {
            padding: 20px;
            border-radius: 12px;
            background: #1a1d29;
            border: 1px solid #2b2f44;
            margin-bottom: 20px;
        }
        .title {
            font-size: 30px;
            font-weight: bold;
            color: #4c8bf5;
            text-align: center;
        }
        .subtitle {
            font-size: 16px;
            color: #c9c9c9;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("gradient_boosting_regressor_model.pkl", "rb"))

# =========================
# HEADER
# =========================
st.markdown('<p class="title">📈 Aplikasi Prediksi Harga Penutupan Crypto</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Gunakan data untuk memprediksi nilai close price.</p>', unsafe_allow_html=True)

# =========================
# CARD INPUT
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔧 Input Data Harga Crypto")

open_val = st.number_input("Open price", value=0.0)
high_val = st.number_input("High price", value=0.0)
low_val = st.number_input("Low price", value=0.0)
volume_val = st.number_input("Volume", value=0.0)

ticker = st.selectbox("Pilih Crypto", ["BTC", "ADA"])

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HITUNG FITUR TURUNAN
# =========================
daily_return = high_val - low_val
range_val = high_val - low_val

# =========================
# DATAFRAME INPUT SESUAI TRAINING
# =========================
input_df = pd.DataFrame({
    'open': [open_val],
    'high': [high_val],
    'low': [low_val],
    'volume': [volume_val],
    'ticker_BTC': [1 if ticker=="BTC" else 0],
    'ticker_ETH': [1 if ticker=="ETH" else 0],
    'ticker_USDT': [1 if ticker=="USDT" else 0],
})

# =========================
# FUNGSI PREDIKSI
# =========================
def predict_price(input_df, model):
    """
    Menyamakan kolom input dengan kolom yang digunakan model saat training.
    Menghindari error mismatch fitur.
    """

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model tidak memiliki feature_names_in_. Pastikan training memakai DataFrame.")

    feature_names = list(model.feature_names_in_)

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]
    input_df = input_df.astype(float)

    prediction_value = model.predict(input_df)[0]
    return prediction_value

# =========================
# PREDIKSI
# =========================
prediction = predict_price(input_df, model)

# =========================
# CARD OUTPUT
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Hasil Prediksi Harga Penutupan")

st.markdown(
    f"""
    <h2 style='text-align:center; color:#4cdb7b; margin-top:10px'>
        {prediction:,.2f}
    </h2>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# DEBUG OPSIONAL
# =========================
with st.expander("🔍 Debug (Klik untuk melihat)"):
    st.write("Fitur model:", list(model.feature_names_in_))
    st.write("Kolom input_df:", list(input_df.columns))
    st.dataframe(input_df)
