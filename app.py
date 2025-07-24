
import streamlit as st
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load metrics from JSON file
with open("model_metrics.json", "r") as f:
    metrics = json.load(f)

# Load model and pre-processing tools
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict if they are likely to churn.")

# UI inputs
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Build input dict for prediction
input_dict = {
    'SeniorCitizen': senior,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender_Male': 1 if gender == "Male" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'PhoneService_Yes': 1 if phone_service == "Yes" else 0,
    'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
    'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
    'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,
    'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
    'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
    'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
    'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
    'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
    'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
    'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
    'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
    'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
    'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
    'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
    'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'PaperlessBilling_Yes': 1 if paperless == "Yes" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0,
}

# Fill missing feature values with 0
for f in features:
    if f not in input_dict:
        input_dict[f] = 0

# Prepare input for prediction
input_array = np.array([input_dict[feature] for feature in features]).reshape(1, -1)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    if prediction == 1:
        st.warning("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
with st.expander("📊 Show Model Performance Metrics"):
    st.write("### 🧪 Classification Metrics")
    st.write(f"**Accuracy**: {metrics['Accuracy']:.2f}")
    st.write(f"**Precision**: {metrics['Precision']:.2f}")
    st.write(f"**Recall**: {metrics['Recall']:.2f}")
    st.write(f"**F1 Score**: {metrics['F1 Score']:.2f}")

    st.write("### 🌀 Confusion Matrix")
    cm = metrics["Confusion Matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

