# cibil_score_predection
CIBIL score prediction web application
This is a Flask-based web application that predicts the CIBIL credit score based on user input features like income, loan amount, and number of credit cards. It demonstrates a simple machine learning model deployment with a web interface.

## Features
 User input form with 12 financial/business features:
  - Annual Revenue
  - Annual Profit
  - Existing Loan Amount
  - Existing Loan Interest Rate
  - GST Compliance (%)
  - Past Defaults
  - Bank Transactions
  - Ecommerce Sales
  - Interest Coverage Ratio
  - Loan Amount Required
  - Principal Investment
  - Annual Investment

- Loads a pre-trained ML model to predict the CIBIL score.
- Simple web UI for entering data and showing predictions.

## Project Structure
cibil/
├── app.py # Flask backend app
├── src/
│ └── data.py
│ └── model.py
├── model_files
│ └── label_encoders.pkl
│ └── risk_score-model.pkl
│ └── scaler.pkl
├── templates/
│ └── index.html # HTML form and results page
├── requirements.txt # Python dependencies
└── README.md # This file

## Create and activate a Python virtual environment
python -m venv venv
# Windows
venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Run the Flask app
python app.py

## Open your browser and go to
http://127.0.0.1:5000/
