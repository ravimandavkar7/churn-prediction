import streamlit as st
import pandas as pd
import joblib

# Load everything
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Customer Churn Prediction")

# Inputs
tenure = st.number_input("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
gender = st.selectbox("Gender", ["Male", " Female"])

# Build raw input
raw_input = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract,
    "gender": gender
}

# Convert to DataFrame
input_df = pd.DataFrame([raw_input])

# One-hot encode like training
input_encoded = pd.get_dummies(input_df)

# Reindex to match training features
input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

# Scale
input_scaled = scaler.transform(input_encoded)

# Predict
if st.button("Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Churn Probability: **{prob:.2f}**")

    if prob >= 0.6:
        st.error("⚠️ High Risk Customer")
    elif prob >= 0.35:
        st.warning("⚠️ Medium Risk Customer")
    else:
        st.success("✅ Low Risk Customer")