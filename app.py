import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess

if not os.path.exists("model/saved_models.pkl"):

    import subprocess
    subprocess.run(["python", "model/train_models.py"])

models = pickle.load(open("model/saved_models.pkl","rb"))
scaler = pickle.load(open("model/scaler.pkl","rb"))


st.set_page_config(page_title="Income Classification App", layout="wide")

st.title("💼 Adult Income Classification using ML Models")

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a Model",
    list(models.keys())
)

st.sidebar.markdown("### Prediction Meaning")
st.sidebar.write("0 → Income ≤ 50K")
st.sidebar.write("1 → Income > 50K")

uploaded_file = st.file_uploader("📂 Upload Test CSV (without income column)")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(data.head())

    data = pd.get_dummies(data)

    model_features = scaler.feature_names_in_

    for col in model_features:
        if col not in data.columns:
            data[col] = 0

    data = data[model_features]

    X = scaler.transform(data)

    model = models[model_name]
    prediction = model.predict(X)

    st.subheader("🔮 Predictions")

    result_df = pd.DataFrame(prediction, columns=["Predicted Income Class"])
    st.write(result_df)

    csv = result_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )

    if st.checkbox("Show Confusion Matrix"):
        y_true = st.number_input("Enter Actual Class (0 or 1)",0,1)
        cm = confusion_matrix([y_true]*len(prediction), prediction)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    if st.checkbox("Show Classification Report"):
        y_true = st.number_input("Enter Actual Class ",0,1)
        report = classification_report([y_true]*len(prediction), prediction, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
