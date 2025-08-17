import streamlit as st
import pandas as pd
import joblib
import json

# ----------------- Load Model & Metadata -----------------
model = joblib.load("model.joblib")

with open("metadata.json", "r") as f:
    metadata = json.load(f)

target = metadata["target"]
numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
categorical_options = metadata["categorical_options"]

st.title("🏡 House Price Prediction App")

st.markdown("Fill in the details below to predict the house price:")

# ----------------- Collect User Inputs -----------------
user_input = {}

# Numeric inputs
for col in numeric_features:
    user_input[col] = st.number_input(f"Enter {col}:", value=0.0)

# Handle CITY separately (only once)
#if "city" in categorical_features:
    #options = categorical_options.get("city", ["Unknown"])
    #user_input["city"] = st.selectbox("Select City:", options)

# Handle OTHER categorical features
for col in categorical_features:
    if col == "city":  # skip city, already handled
        continue
    options = categorical_options.get(col, ["Unknown"])
    default_val = options[0] if options else "Unknown"
    user_input[col] = st.selectbox(f"Select {col}:", options, index=0)

# ----------------- Prediction -----------------
if st.button("Predict Price"):
    df_input = pd.DataFrame([user_input])

    # ✅ Ensure all training columns exist in input
    for col in numeric_features + categorical_features:
        if col not in df_input.columns:
            df_input[col] = "Unknown" if col in categorical_features else 0.0

    prediction = model.predict(df_input)[0]
    st.success(f"💰 Predicted {target}: {prediction:,.2f}")
