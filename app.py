
import streamlit as st
import pandas as pd
import pickle
import sklearn
import json

# Load trained Random Forest model
model = pickle.load(open("gradient_boosting_regressor_model.pkl", "rb"))

# Streamlit app title
st.title("Aplikasi Prediksi Harga Penutupan Crypto")

st.write("Masukkan data untuk memprediksi nilai **close price**.")

# Sidebar inputs
st.sidebar.header("Input Fitur")

open_val = st.sidebar.number_input("Open", value=0.0)
high_val = st.sidebar.number_input("High", value=0.0)
low_val = st.sidebar.number_input("Low", value=0.0)

ticker = st.selectbox("Pilih Crypto", ["BTC", "ETH", "ADA"]))

# Define columns used during model training
columns = [
    "open", "high", "low", "volume",
    "daily_return", "range",
    "ticker_ADA", "ticker_BTC"
]

# Initialize empty DataFrame with correct columns
input_df = pd.DataFrame({col: [0] for col in columns})

# Populate numeric features
open_val = st.number_input("Open price")
high_val = st.number_input("High price")
low_val = st.number_input("Low price")
volume_val = st.number_input("Volume")

# Populate categorical one-hot columns
input_df = pd.DataFrame({
    "open": [open_val],
    "high": [high_val],
    "low": [low_val],
    "volume": [volume_val],
    'ticker_BTC': [1 if ticker == 'BTC' else 0],
    'ticker_ETH': [1 if ticker == 'ETH' else 0],
    'ticker_ADA': [1 if ticker == 'ADA' else 0],
})


# Prediction
prediction = model.predict(input_df)[0]

st.subheader("Hasil Prediksi Harga Penutupan Crypto ADA dan BTC")
st.write(prediction)

# Simpan nama fitur
json.dump(list(X_train.columns), open("feature_names.json", "w"))

# Pilihan crypto
ticker = st.selectbox("Pilih Crypto", ["BTC", "ETH", "ADA"])
