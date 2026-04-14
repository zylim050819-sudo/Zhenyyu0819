import streamlit as st
import joblib
import json
import numpy as np

# Load model
model = joblib.load("xgboost_model.joblib")

# Load metadata
with open("xgboost_metadata.json") as f:
    metadata = json.load(f)

st.title("Obesity Prediction System")

st.write("Click predict to test model")

if st.button("Predict"):
    # Dummy input (must match number of features)
    input_data = np.zeros(len(metadata["feature_columns"]))

    prediction = model.predict([input_data])[0]

    st.success(f"Prediction: {metadata['class_labels'][prediction]}")
