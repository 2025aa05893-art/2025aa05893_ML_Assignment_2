import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ================= LOAD DATASET =================

df = pd.read_csv("model/data.csv")

df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

df["income"] = df["income"].apply(lambda x: 1 if x=='>50K' else 0)

X = df.drop("income", axis=1)
y = df["income"]

# ================= ONE HOT ENCODING =================

X = pd.get_dummies(X)

# ================= TRAIN TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= SCALING =================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= MODELS =================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train, y_train)

# ================= SAVE MODELS =================

# Create model folder if not exists
if not os.path.exists("model"):
    os.makedirs("model")

pickle.dump(models, open("model/saved_models.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Models Trained & Saved Successfully")
