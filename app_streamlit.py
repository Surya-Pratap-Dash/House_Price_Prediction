import streamlit as st
import pandas as pd
import joblib
import json

# Load model & metadata
model = joblib.load("model.joblib")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

st.title("🏠 House Price Prediction")

mode = st.radio("Choose Mode:", ["Single Prediction", "Batch Prediction"])

if mode == "Single Prediction":
    st.header("Enter House Details")

    input_data = {}
    # Numeric features
    for col in metadata["numeric_features"]:
        input_data[col] = st.number_input(f"{col}", step=1.0)

    # Categorical features
    for col in metadata["categorical_features"]:
        input_data[col] = st.selectbox(f"{col}", metadata["categorical_options"][col])

    if st.button("Predict"):
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        st.success(f"🏷 Predicted Price: ₹ {prediction:,.0f}")

else:
    st.header("Upload CSV for Batch Prediction")
    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        preds = model.predict(df)
        df["PredictedPrice"] = preds
        st.write(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
