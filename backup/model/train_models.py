import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ---------------- LOAD DATA ---------------- #

df = pd.read_csv("data.csv")

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# ---------------- TARGET ---------------- #

y = df['income']
X = df.drop('income', axis=1)

# convert target
y = y.apply(lambda x: 1 if x == '>50K' else 0)

# ---------------- ONE HOT ENCODING ---------------- #

X = pd.get_dummies(X)

# ---------------- SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ---------------- #

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ---------------- MODELS ---------------- #

models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# ---------------- SAVE FILES ---------------- #

pickle.dump(models, open("saved_models.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# ⭐ VERY IMPORTANT FOR STREAMLIT
pickle.dump(X_train.columns.tolist(), open("feature_columns.pkl", "wb"))

print("All models and preprocessing files saved successfully!")
