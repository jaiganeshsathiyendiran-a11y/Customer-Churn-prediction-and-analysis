import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SAVE model AND feature names
joblib.dump(model, "churn_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_features.pkl")

print("✅ Model and feature names saved")
