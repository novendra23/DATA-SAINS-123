import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("gradient_boosting_regressor_model.pkl", "rb"))

# App title
st.title("Aplikasi Prediksi Harga Penutupan Crypto (Gradient Boosting Regressor)")
st.write("Masukkan data untuk memprediksi nilai **close price**.")

# Input dari user
open_val = st.number_input("Open price")
high_val = st.number_input("High price")
low_val = st.number_input("Low price")
volume_val = st.number_input("Volume")

ticker = st.selectbox("Pilih Crypto", ["BTC", "ADA"])   # hanya yang ada di model

# Hitung fitur turunan (HARUS sama seperti training!)
daily_return = high_val - low_val        # contoh saja, sesuaikan training kamu
range_val = high_val - low_val           # contoh saja, sesuaikan training kamu

# Buat dataframe input SAMA persis dengan kolom training
input_df = pd.DataFrame({
    'open': [open_val],
    'high': [high_val],
    'low': [low_val],
    'volume': [volume_val],
    'ticker_BTC': [1 if ticker=="BTC" else 0],
    'ticker_ETH': [1 if ticker=="ETH" else 0],
    'ticker_USDT': [1 if ticker=="USDT" else 0],
})

def predict_price(input_df, model):
    """
    Fungsi ini menyamakan kolom input dengan kolom yang digunakan model saat training.
    Menghindari error: ValueError: Number of features does not match.
    """

    # --- CEK APAKAH MODEL MEMILIKI NAMA FITUR ---
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model tidak memiliki attribute feature_names_in_. "
                         "Pastikan model dilatih menggunakan DataFrame pandas.")

    # Ambil fitur yang digunakan model saat training
    feature_names = list(model.feature_names_in_)

    # --- Tambahkan kolom yang tidak ada di input_df ---
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # --- Buang kolom input yang tidak dikenali model ---
    input_df = input_df[feature_names]

    # --- Pastikan nilai berbentuk float ---
    input_df = input_df.astype(float)

    # --- Prediksi ---
    prediction_value = model.predict(input_df)[0]

    return prediction_value

prediction = predict_price(input_df, model)

st.subheader("Hasil Prediksi")

prediction = predict_price(input_df, model)

st.write("Prediksi Close Price:", prediction)
