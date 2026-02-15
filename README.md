# Machine Learning Assignment 2

## a. Problem Statement

The objective of this assignment is to build and evaluate multiple Machine Learning classification models to predict whether an individual's income exceeds $50K per year based on demographic and employment-related attributes from the Adult Income Dataset.


## b. Dataset Description

The dataset used for this assignment is the **Adult Income Dataset** obtained from the UCI Machine Learning Repository.

This dataset contains demographic and employment information such as:

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Gender
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

### Target Variable:
Income

- <=50K
- >50K

This is a binary classification problem where the goal is to predict whether an individual's income is above or below $50K annually.

---

## c. Models Used

Six Machine Learning classification models were implemented and evaluated using the following performance metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.85 | 0.91 | 0.84 | 0.85 | 0.84 | 0.55 |
| Decision Tree | 0.81 | 0.83 | 0.81 | 0.81 | 0.80 | 0.47 |
| kNN | 0.82 | 0.86 | 0.81 | 0.82 | 0.81 | 0.47 |
| Naive Bayes | 0.37 | 0.78 | 0.79 | 0.37 | 0.33 | 0.18 |
| Random Forest (Ensemble) | 0.85 | 0.92 | 0.84 | 0.85 | 0.84 | 0.56 |
| XGBoost (Ensemble) | 0.87 | 0.94 | 0.86 | 0.87 | 0.86 | 0.62 |

---

### Model Performance Observations

| ML Model | Observation about model performance |
|----------|-------------------------------------|
| Logistic Regression | Provided good performance for binary classification with balanced precision and recall values. |
| Decision Tree | Simple and interpretable model but slightly lower performance due to overfitting tendencies. |
| kNN | Showed moderate performance but sensitive to feature scaling and noisy data. |
| Naive Bayes | Performed poorly due to independence assumption among features. |
| Random Forest (Ensemble) | Performed better than individual models by reducing variance through ensemble learning. |
| XGBoost (Ensemble) | Achieved the best performance with highest accuracy and MCC score due to gradient boosting technique. |

---

## Streamlit Application Deployment

A Streamlit Web Application has been developed to deploy the trained models with the following features:

- Dataset Upload Option (CSV)
- Model Selection Dropdown
- Display of Evaluation Metrics
- Confusion Matrix Visualization
- Classification Report
- Download Predictions as CSV


### Streamlit App Link
https://2025aa05893mlassignment2-jsvyezsxgisy8gpf2jzuvk.streamlit.app/

### GitHub Repository Link
https://github.com/2025aa05893-art/2025aa05893_ML_Assignment_2



## Conclusion

Among all implemented models, **Random Forest** and **XGBoost** provided the best classification performance for predicting income levels in the Adult Income dataset due to their ensemble learning capabilities.

---

