import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn


# Load trained model
model = pickle.load(open("gradient_boosting_regressor_model.pkl", "rb"))
ticker_USDT': ({[1 if ticker=="USDT" else 0],
})

# Hanya pakai kolom yang diperbolehkan model
input_df = input_df[feature_names]
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

# Prediksi
return model.predict(input_df)[0]
    # --- Buang kolom input yang tidak dikenali model ---
    input_df = input_df[feature_names]

    # --- Pastikan nilai berbentuk float ---
    input_df = input_df.astype(float)

    # --- Prediksi ---
    prediction_value = model.predict(input_df)[0]

    return prediction_value

st.subheader("Hasil Prediksi")

prediction = predict_price(input_df, model)

st.write("Prediksi Close Price:", prediction)
