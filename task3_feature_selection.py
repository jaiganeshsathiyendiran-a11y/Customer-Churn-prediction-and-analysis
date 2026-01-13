import pandas as pd
from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("🔥 TOP IMPORTANT FEATURES 🔥")
print(feature_importance.head(10))
