# 🧠 Machine Learning Assignment 2

## 💰 Adult Income Prediction using Multiple ML Models

---

## 📌 Problem Statement

The objective of this project is to build a Machine Learning classification system that predicts whether an individual earns:

* **<=50K**
* **>50K**

based on demographic and employment-related attributes from the **Adult Income Dataset**.

---

## 📂 Dataset Used

**Adult Income Dataset** from the UCI Machine Learning Repository.

This dataset contains various features such as:

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
✔ Display of Evaluation Metrics
✔ Confusion Matrix Visualization
✔ Classification Report
✔ Predictions Table
✔ Download Predictions as CSV Option

---

## 🖥️ How to Run the Application Locally

Follow the steps below:

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
│   ├── train_models.py
│   ├── saved_models.pkl
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📈 Sample Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | MCC  |
| ------------------- | -------- | --------- | ------ | -------- | ---- |
| Logistic Regression | 0.85     | 0.84      | 0.85   | 0.84     | 0.55 |
| Decision Tree       | 0.81     | 0.81      | 0.81   | 0.80     | 0.47 |
| KNN                 | 0.82     | 0.81      | 0.82   | 0.81     | 0.47 |
| Naive Bayes         | 0.37     | 0.79      | 0.37   | 0.33     | 0.18 |
| Random Forest       | 0.85     | 0.84      | 0.85   | 0.84     | 0.56 |
| XGBoost             | 0.87     | 0.86      | 0.87   | 0.86     | 0.62 |

---

## 🌐 Deployment

The Streamlit application is deployed using **Streamlit Cloud Platform**.

---

## 📌 Conclusion

Among all the implemented models, **Random Forest** and **XGBoost** provided the best classification performance for predicting income levels in the Adult Income dataset.



Machine Learning Assignment Submission
Adult Income Classification using Streamlit Deployment
