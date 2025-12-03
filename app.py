import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("gradient_boosting_regressor_model.pkl", "rb"))

# App title
st.title("Aplikasi Prediksi Harga Penutupan Crypto")
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
# Debug untuk melihat fitur model & input
st.write("Model expects:", getattr(model, "feature_names_in_", "unknown"))
st.write("Input columns:", list(input_df.columns))

# Prediction
prediction = model.predict(input_df)[0]

st.subheader("Hasil Prediksi")
st.write(prediction)
