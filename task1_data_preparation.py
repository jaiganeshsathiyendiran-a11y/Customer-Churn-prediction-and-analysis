import pandas as pd

print("🔥 DATA PREPARATION STARTED 🔥")

# Load dataset
df = pd.read_csv("data/Telco_Customer_Churn.csv")

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convert target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)

print("✅ Data cleaned and saved as cleaned_data.csv")

