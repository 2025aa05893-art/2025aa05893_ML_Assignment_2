import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

df = pd.read_csv("adult.csv")

df.replace(' ?', pd.NA, inplace=True)
df.replace('?', pd.NA, inplace=True)
df.fillna('Unknown', inplace=True)

df['income'] = df['income'].map({'<=50K':0, '>50K':1})

X = df.drop('income', axis=1)
y = df['income']

# 🔥 ONE HOT ENCODE HERE
X = pd.get_dummies(X)

# 🔥 SAVE COLUMN NAMES
columns = X.columns

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

rf.fit(X_train,y_train)
dt.fit(X_train,y_train)
xgb.fit(X_train,y_train)

models = {
    "Random Forest":rf,
    "Decision Tree":dt,
    "XGBoost":xgb
}

if not os.path.exists("model"):
    os.makedirs("model")

pickle.dump(models, open("model/saved_models.pkl","wb"))
pickle.dump(scaler, open("model/scaler.pkl","wb"))
pickle.dump(columns, open("model/columns.pkl","wb"))

print("Training Done Successfully")
