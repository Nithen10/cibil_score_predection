import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from xgboost import XGBRegressor

# Load the dataset
file_path = "Processed_Credit_Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Selecting numerical features
selected_features = [
    "Annual_Revenue", "Annual_Profit", "Existing_Loan_Amount", "Existing_Loan_Interest_Rate",
    "GST_Compliance(%)", "Past_Defaults", "Bank_Transactions", "Ecommerce_Sales",
    "Interest_Coverage_Ratio", "Loan_Amount_Required", "Principal_Investment", "Annual_Investment"
]

# Encode categorical variables
label_encoders = {}
for col in ["Market_Trend", "Sector"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize numerical features
scaler = MinMaxScaler(feature_range=(0, 1))
df[selected_features] = scaler.fit_transform(df[selected_features])

# Compute risk score
weights = {
    "Annual_Revenue": 0.2, "Annual_Profit": 0.2, "Existing_Loan_Amount": -0.15,
    "Existing_Loan_Interest_Rate": -0.1, "GST_Compliance(%)": 0.1, "Past_Defaults": -0.2,
    "Bank_Transactions": 0.15, "Ecommerce_Sales": 0.1, "Interest_Coverage_Ratio": 0.1,
    "Loan_Amount_Required": -0.1, "Principal_Investment": 0.05, "Annual_Investment": 0.05
}
df["Risk_Score"] = df[selected_features].mul(weights).sum(axis=1)
df["Risk_Score"] = np.interp(df["Risk_Score"], (df["Risk_Score"].min(), df["Risk_Score"].max()), (300, 900))

# Prepare data for training
X = df[selected_features]
y = df["Risk_Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, R2 Score: {r2}")

# Save model and preprocessing objects
save_dir = "./model_files"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, "risk_score_model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(save_dir, "label_encoders.pkl"))
print("Model and preprocessing objects saved successfully!")
