
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model and preprocessing objects
model_path = "./model_files/risk_score_model.pkl"
scaler_path = "./model_files/scaler.pkl"
encoders_path = "./model_files/label_encoders.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)

# Feature columns used in training
selected_features = [
    "Annual_Revenue", "Annual_Profit", "Existing_Loan_Amount", "Existing_Loan_Interest_Rate",
    "GST_Compliance(%)", "Past_Defaults", "Bank_Transactions", "Ecommerce_Sales",
    "Interest_Coverage_Ratio", "Loan_Amount_Required", "Principal_Investment", "Annual_Investment"
]

@app.route('/')
def home():
    return render_template("citi.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract numerical input features
        features = [
            float(data.get("Annual_Revenue", 0)),
            float(data.get("Annual_Profit", 0)),
            float(data.get("Existing_Loan_Amount", 0)),
            float(data.get("Existing_Loan_Interest_Rate", 0)),
            float(data.get("GST_Compliance(%)", 0)),
            float(data.get("Past_Defaults", 0)),
            float(data.get("Bank_Transactions", 0)),
            float(data.get("Ecommerce_Sales", 0)),
            float(data.get("Interest_Coverage_Ratio", 0)),
            float(data.get("Loan_Amount_Required", 0)),
            float(data.get("Principal_Investment", 0)),
            float(data.get("Annual_Investment", 0))
        ]
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale input features
        features_scaled = scaler.transform(features)
        
        # Predict risk score
        risk_score = model.predict(features_scaled)[0]
        risk_score = int(risk_score)
        
        return jsonify({"risk_score": risk_score})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)