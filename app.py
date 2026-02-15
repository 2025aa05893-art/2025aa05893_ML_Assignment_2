import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------- LOAD FILES ---------------- #

models = pickle.load(open("model/saved_models.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
feature_columns = pickle.load(open("model/feature_columns.pkl", "rb"))

# ---------------- UI ---------------- #

st.set_page_config(page_title="Adult Income Prediction", layout="wide")

st.title("💰 Adult Income Prediction using ML Models")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    if 'income' in data.columns:

        y_true = data['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
        data = data.drop('income', axis=1)
    else:
        y_true = None

    data = pd.get_dummies(data)

    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_columns]

    X = scaler.transform(data)

    model_choice = st.selectbox("Select Model", list(models.keys()))

    model = models[model_choice]
    preds = model.predict(X)

        st.subheader("Predictions")
        st.write(preds)

        if y_true is not None:

            st.subheader("Evaluation Metrics")

            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds)
            rec = recall_score(y_true, preds)
            f1 = f1_score(y_true, preds)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", round(acc, 2))
            col2.metric("Precision", round(prec, 2))
            col3.metric("Recall", round(rec, 2))
            col4.metric("F1 Score", round(f1, 2))

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_true, preds)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("Classification Report")
            report = classification_report(y_true, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

        result_df = pd.DataFrame(preds, columns=["Prediction"])

        csv = result_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
