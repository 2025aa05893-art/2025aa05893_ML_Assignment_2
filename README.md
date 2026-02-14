# Machine Learning Assignment 2

## Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual's income exceeds $50K per year based on demographic and employment related attributes.

---

## Dataset Description
The dataset used in this assignment is the Adult Income Dataset obtained from the UCI Machine Learning Repository.

It contains:
- 48842 instances
- 14 input features
- 1 target variable (income)

The target variable classifies whether an individual's annual income is:
<=50K or >50K

Categorical features were converted into numerical format using One-Hot Encoding for model compatibility.

---

## Models Used

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8478 | 0.8408 | 0.8478 | 0.8478 | 0.8422 | 0.5539 |
| Decision Tree | 0.8085 | 0.8103 | 0.8085 | 0.8085 | 0.8094 | 0.4739 |
| KNN | 0.8183 | 0.8116 | 0.8183 | 0.8183 | 0.8142 | 0.4760 |
| Naive Bayes | 0.3707 | 0.7957 | 0.3707 | 0.3707 | 0.3360 | 0.1899 |
| Random Forest | 0.8498 | 0.8440 | 0.8498 | 0.8498 | 0.8456 | 0.5646 |
| XGBoost | 0.8714 | 0.8666 | 0.8714 | 0.8714 | 0.8670 | 0.6254 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|------------------------------------|
| Logistic Regression | Logistic Regression showed good balanced performance across all metrics indicating its effectiveness for binary classification tasks. |
| Decision Tree | Decision Tree classifier showed moderate accuracy but may suffer from overfitting on complex datasets. |
| KNN | KNN performed reasonably well but is sensitive to feature scaling and computationally expensive for larger datasets. |
| Naive Bayes | Naive Bayes showed poor accuracy due to its strong independence assumption among features which may not hold true in real-world datasets. |
| Random Forest | Random Forest improved performance compared to Decision Tree due to ensemble learning which reduces variance. |
| XGBoost | XGBoost showed the highest accuracy and MCC score making it the best performing model among all the implemented classification models. |

---

## Project Structure

project-folder/
│-- app.py  
│-- requirements.txt  
│-- README.md  
│-- model/  
  │-- train_models.py  
  │-- saved_models.pkl  
  │-- scaler.pkl  

---

## Deployment

The Streamlit application has been deployed using Streamlit Community Cloud and provides the following features:

- Dataset Upload Option (CSV)
- Model Selection Dropdown
- Display of Evaluation Metrics
- Confusion Matrix
- Classification Report
