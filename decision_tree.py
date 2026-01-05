import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("decision_tree_model.pkl")

st.title("🏠 House Price Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = pd.get_dummies(data)

    # Align columns with training data
    expected_cols = model.feature_names_in_  # works if model saved sklearn >= 1.0
    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0
    data = data[expected_cols]

    # Make prediction
    prediction = model.predict(data)
    st.write("Predicted Prices:", prediction)
