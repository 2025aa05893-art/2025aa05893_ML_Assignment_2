# 🧠 Machine Learning Assignment 2

## 💰 Adult Income Classification using Multiple ML Models

---

## 📌 Problem Statement

The objective of this project is to build a Machine Learning classification system that predicts whether an individual earns:

* **<=50K**
* **>50K**

based on demographic and employment-related attributes from the Adult Income Dataset.

---

## 📂 Dataset Used

Adult Income Dataset from the UCI Machine Learning Repository.

This dataset contains features such as:

* Age
* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Gender
* Hours-per-week
* Native Country

These features are used to predict the income class of an individual.

---

## ⚙️ Machine Learning Models Implemented

The following classification algorithms were implemented:

* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Random Forest
* XGBoost

---

## 📊 Evaluation Metrics Used

To evaluate model performance, the following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)
* Confusion Matrix
* Classification Report

---

## 🚀 Streamlit Web Application Features

The deployed Streamlit application includes:

✔ CSV Dataset Upload Option (Test Data Only)
✔ Model Selection Dropdown
✔ Evaluation Metrics Display
✔ Confusion Matrix Visualization
✔ Classification Report
✔ Predictions Table
✔ Download Predictions as CSV Option

---

## ❗ Important Note Regarding Model Files (.pkl)

The trained model files such as:

* `saved_models.pkl`
* `scaler.pkl`

are **NOT uploaded to GitHub** due to the file size limitations of the Streamlit Free Tier and GitHub.

Instead:

✔ The Streamlit application automatically trains the models during the **first deployment** in the cloud environment.
✔ The trained `.pkl` files are then generated dynamically within the Streamlit Cloud runtime environment.
✔ These files are reused in subsequent runs for prediction.

This approach ensures:

* Successful deployment on Streamlit Cloud
* Compliance with GitHub file size limits
* Smooth functioning of the application

---

## 🖥️ How to Run the Application Locally

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
2025aa05893_ML_Assignment_2
│
├── model
│   └── train_models.py
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🌐 Deployment

The Streamlit application is deployed using Streamlit Cloud.

🔗 Streamlit App Link:
https://2025aa05893mlassignment2-jqokmz27sn5afuwruvrd8c.streamlit.app/

🔗 GitHub Repository Link:
https://github.com/2025aa05893-art/2025aa05893_ML_Assignment_2

---

## 📌 Conclusion

Among all the implemented models, **Random Forest** and **XGBoost** provided the best performance for predicting income levels in the Adult Income dataset.

---
