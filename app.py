import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Adult Income Prediction", layout="wide")

st.title("💰 Adult Income Prediction using ML Models")

# ================= TRAIN MODEL IF NOT EXISTS =================

if not os.path.exists("model/saved_models.pkl"):
    st.warning("Training models for first time... Please wait ⏳")
    import model.train_models
    st.success("Training Completed ✅")

# ================= LOAD MODELS =================

models = pickle.load(open("model/saved_models.pkl","rb"))
scaler = pickle.load(open("model/scaler.pkl","rb"))

# ================= UPLOAD CSV =================

uploaded_file = st.file_uploader("Upload Adult CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    # ================= HANDLE MISSING =================

    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # ================= SEPARATE TARGET =================

    if "income" in df.columns:
        y_true = df["income"].apply(lambda x: 1 if x=='>50K' else 0)
        df = df.drop("income", axis=1)
    else:
        y_true = None

    # ================= ONE HOT ENCODING =================

    df_encoded = pd.get_dummies(df)

    train_columns = scaler.feature_names_in_

    for col in train_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[train_columns]

    # ================= SCALE =================

    X = scaler.transform(df_encoded)

    # ================= MODEL DROPDOWN =================

    model_name = st.selectbox(
        "Choose a Model",
        list(models.keys())
    )

    model = models[model_name]
    preds = model.predict(X)

    st.subheader("Predictions")

    pred_df = pd.DataFrame()
    pred_df["Predicted Income Class"] = preds

    st.write(pred_df.head(10))

    # ================= METRICS =================

    if y_true is not None:

        st.subheader("Evaluation Metrics")

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds)
        rec = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        mcc = matthews_corrcoef(y_true, preds)

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(acc,3))
        col2.metric("Precision", round(prec,3))
        col3.metric("Recall", round(rec,3))

        col1.metric("F1 Score", round(f1,3))
        col2.metric("MCC", round(mcc,3))

        # ================= CONFUSION MATRIX =================

        st.subheader("Confusion Matrix Plot")

        cm = confusion_matrix(y_true, preds)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # ================= CLASSIFICATION REPORT =================

        st.subheader("Classification Report")
        st.text(classification_report(y_true, preds))

    # ================= DOWNLOAD =================

    csv = pred_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Predictions CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )
