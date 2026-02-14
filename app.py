import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Adult Income Predictor")

st.title("💰 Adult Income Prediction using ML Models")

# ================= TRAIN MODEL IF NOT EXISTS =================

if not os.path.exists("model/saved_models.pkl"):

    st.warning("Training models for first time... Please wait 1-2 minutes ⏳")

    from model.train_models import *

    st.success("Training Completed ✅")

# ================= LOAD MODELS =================

models = pickle.load(open("model/saved_models.pkl","rb"))
scaler = pickle.load(open("model/scaler.pkl","rb"))
columns = pickle.load(open("model/columns.pkl","rb"))

# ================= FILE UPLOAD =================

uploaded_file = st.file_uploader("Upload Adult CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # ---------- CLEAN DATA ----------
    data.replace(' ?', pd.NA, inplace=True)
    data.replace('?', pd.NA, inplace=True)
    data.fillna('Unknown', inplace=True)

    # ---------- ONE HOT ----------
    data = pd.get_dummies(data)

    # ---------- MATCH TRAINING COLUMNS ----------
    for col in columns:
        if col not in data.columns:
            data[col] = 0

    data = data[columns]

    # ---------- SCALE ----------
    X = scaler.transform(data)

    # ---------- PREDICT ----------
    results = {}

    for name, model in models.items():
        pred = model.predict(X)
        results[name] = pred

    result_df = pd.DataFrame(results)

    st.subheader("Predictions")
    st.write(result_df)

    # ---------- DOWNLOAD ----------
    csv = result_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )
