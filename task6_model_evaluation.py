import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

model = joblib.load("churn_model.pkl")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

y_pred = model.predict(X_test)

print("🔥 CLASSIFICATION REPORT 🔥")
print(classification_report(y_test, y_pred))

print("🔥 CONFUSION MATRIX 🔥")
print(confusion_matrix(y_test, y_pred))
