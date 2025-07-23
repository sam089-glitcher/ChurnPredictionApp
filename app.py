import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = pickle.load(open("logistic_model.pkl", "rb"))

# Load label encoders if used (optional)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📱 Customer Churn Prediction")

st.markdown("Enter customer details below to predict if they are likely to churn.")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 2500.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Create a DataFrame from input
input_dict = {
    'gender': [gender],
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method]
}
input_df = pd.DataFrame(input_dict)

# Encode categorical features
for col in input_df.select_dtypes(include='object').columns:
    if col in label_encoders:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])
    else:
        st.warning(f"⚠️ No encoder found for column '{col}'. Please retrain or check.")

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to **churn**. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is likely to **stay**. (Probability: {1 - probability:.2f})")

# Optional: Model performance metrics
with st.expander("📊 Show model performance (optional)"):
    st.markdown("Model: Logistic Regression")
    st.write("Accuracy: 85%")
    st.write("Precision: 0.80")
    st.write("Recall: 0.74")
    st.write("F1 Score: 0.76")
