import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_excel(r"C:\Users\nithe\OneDrive\Desktop\citi\citi_cibil\Processed_Credit_Data.xlsx", engine="openpyxl")

# Select numerical columns that need scaling
numerical_features = ["Age", "Income", "Loan Amount", "Credit Utilization", "Credit History Length"]

# Check if columns exist
missing_cols = [col for col in numerical_features if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# Initialize and fit the scaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numerical_features])

# Save the trained scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")
