import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("decision_tree_model.pkl")

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction App")
st.write("Upload a CSV file with house features to predict prices.")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(data.head())

    # Convert categorical columns to dummies
    data = pd.get_dummies(data)

    # Align columns with training model
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0
    data = data[expected_cols]

    # Predict
    try:
        prediction = model.predict(data)
        st.write("### Predicted Prices")
        st.dataframe(pd.DataFrame(prediction, columns=["Predicted Price"]))
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Optional: Manual input for single prediction
st.write("---")
st.write("### Or Predict for a Single House")

# Example manual inputs (replace with your actual features)
col1, col2, col3 = st.columns(3)
with col1:
    borough_x = st.number_input("Borough X", min_value=0)
with col2:
    block = st.number_input("Block", min_value=0)
with col3:
    lot = st.number_input("Lot", min_value=0)

# Button to predict
if st.button("Predict Single House"):
    input_df = pd.DataFrame([[borough_x, block, lot]], columns=['borough_x','block','lot'])
    input_df = pd.get_dummies(input_df)

    # Align columns
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    price = model.predict(input_df)
    st.success(f"Predicted Price: ${price[0]:,.2f}")
