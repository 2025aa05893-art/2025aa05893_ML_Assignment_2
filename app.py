import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="Adult Income Prediction", layout="wide")

st.title("💰 Adult Income Prediction using ML Models")

# ================= LOAD MODELS DIRECTLY =================

models = pickle.load(open("model/saved_models.pkl","rb"))
scaler = pickle.load(open("model/scaler.pkl","rb"))

# ================= FILE UPLOAD =================

uploaded_file = st.file_uploader("Upload Adult CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # ================= HANDLE MISSING =================

    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # ================= TARGET =================

    if "income" in df.columns:
        y_true = df["income"].apply(lambda x: 1 if x=='>50K' else 0)
        df = df.drop("income", axis=1)
    else:
        y_true = None

    # ================= ENCODING =================

    df_encoded = pd.get_dummies(df)

    train_columns = scaler.feature_names_in_

    for col in train_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[train_columns]

    # ================= SCALE =================

    X = scaler.transform(df_encoded)

    # ================= MODEL DROPDOWN =================

    st.markdown("---")
    model_name = st.selectbox("Select Model", list(models.keys()))

    model = models[model_name]
    preds = model.predict(X)

    # ================= PREDICTIONS =================

    st.markdown("---")
    st.subheader("Predictions")

    pred_df = pd.DataFrame()
    pred_df["Predicted Income Class"] = preds

    st.dataframe(pred_df.head(10), use_container_width=True)

    # ================= METRICS =================

    if y_true is not None:

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds)
        rec = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        mcc = matthews_corrcoef(y_true, preds)

        st.markdown("---")
        st.subheader("📈 Model Performance")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Accuracy", round(acc,3))
        col2.metric("Precision", round(prec,3))
        col3.metric("Recall", round(rec,3))
        col4.metric("F1 Score", round(f1,3))
        col5.metric("MCC", round(mcc,3))

        # ================= CONFUSION MATRIX =================

        st.markdown("---")
        st.subheader("🧩 Confusion Matrix")

        cm = confusion_matrix(y_true, preds)

        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                    xticklabels=['<=50K','>50K'],
                    yticklabels=
