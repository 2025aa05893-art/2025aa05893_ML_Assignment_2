import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

df = pd.read_csv("model/data.csv")
df.columns = df.columns.str.strip()

X = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]

y = y.str.strip()
y = y.map({'<=50K':0,'>50K':1})

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
"Logistic Regression":LogisticRegression(),
"Decision Tree":DecisionTreeClassifier(),
"KNN":KNeighborsClassifier(),
"Naive Bayes":GaussianNB(),
"Random Forest":RandomForestClassifier(),
"XGBoost":XGBClassifier(eval_metric='logloss')
}

results={}

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    results[name]=[
        accuracy_score(y_test,y_pred),
        roc_auc_score(y_test,y_pred),
        precision_score(y_test,y_pred),
        recall_score(y_test,y_pred),
        f1_score(y_test,y_pred),
        matthews_corrcoef(y_test,y_pred)
    ]

pickle.dump(models, open("model/saved_models.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(X.columns, open("model/columns.pkl", "wb"))


print(results)
