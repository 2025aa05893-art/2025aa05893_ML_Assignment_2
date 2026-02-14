import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Adult Income Prediction", layout="wide")

st.title("💰 Adult Income Prediction using ML Models")

# -------------------- TRAIN MODEL ONLY IF NOT EXISTS --------------------

if not os.path.exists("model/saved_models.pkl"):
    st.warning("Training models for first time... Please wait 1-2 minutes ⏳")
    import model.train_models

# -------------------- LOAD MODELS --------------------

models = pickle.load(open("model/saved_models.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# -------------------- UI --------------------

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

model_name = st.selectbox("Select Model", list(models.keys()))

# -------------------- PREDICTION --------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # -------------------- ENCODING --------------------

    data = pd.get_dummies(data)

    # Add missing columns
    for col in columns:
        if col not in data.columns:
            data[col] = 0

    # Remove extra columns
    for col in data.columns:
        if col not in columns:
            data.drop(col, axis=1, inplace=True)

    # Arrange in correct order
    data = data[columns]

    # -------------------- SCALE --------------------

    X = scaler.transform(data)

    # -------------------- PREDICT --------------------

    model = models[model_name]
    predictions = model.predict(X)

    data["Prediction"] = predictions

    st.subheader("Predictions")
    st.write(data)

    # -------------------- DOWNLOAD OPTION --------------------

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
