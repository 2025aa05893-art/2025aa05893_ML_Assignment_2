import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Adult Income Prediction", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:32px !important;
    font-weight: bold;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">💰 Adult Income Prediction using ML Models</p>', unsafe_allow_html=True)

# ---------- LOAD MODELS ----------
models = pickle.load(open("model/saved_models.pkl","rb"))
scaler = pickle.load(open("model/scaler.pkl","rb"))
feature_columns = pickle.load(open("model/feature_columns.pkl","rb"))

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload Adult CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    y_true = None

    # If target present
    if 'income' in data.columns:
        y_true = data['income'].map({'>50K':1,'<=50K':0})
        data = data.drop('income',axis=1)

    data = pd.get_dummies(data)

    # Match training columns
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_columns]

    X = scaler.transform(data)

    # ---------- MODEL SELECTION ----------
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    preds = model.predict(X)

    st.subheader("Predictions")
    st.write(preds[:10])

    # ---------- METRICS ----------
    if y_true is not None:

        st.subheader("Evaluation Metrics")

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("Accuracy", round(accuracy_score(y_true,preds),3))
        col2.metric("Precision", round(precision_score(y_true,preds),3))
        col3.metric("Recall", round(recall_score(y_true,preds),3))
        col4.metric("F1 Score", round(f1_score(y_true,preds),3))

        # ---------- CONFUSION MATRIX ----------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true,preds)

        fig,ax = plt.subplots()
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['<=50K','>50K'],
                    yticklabels=['<=50K','>50K'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)

        # ---------- CLASSIFICATION REPORT ----------
        st.subheader("Classification Report")
        report = classification_report(y_true,preds,output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    else:
        st.warning("Upload test dataset WITH income column to view metrics.")

    # ---------- DOWNLOAD ----------
    pred_df = pd.DataFrame(preds,columns=["Prediction"])
    csv = pred_df.to_csv(index=False).encode('utf-8')

    st.download_button("Download Predictions CSV",
                       data=csv,
                       file_name='predictions.csv',
                       mime='text/csv')
