# ==========================================
# CELL 2: RUN THIS SECOND (Create UI File)
# ==========================================
%%writefile app.py
import numpy as np
import pandas as pd
import streamlit as st
import json
from joblib import load

MODEL_PATH = "xgboost_model.joblib"
METADATA_PATH = "xgboost_metadata.json"

@st.cache_resource
def load_model():
    model = load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata["feature_columns"], metadata["class_labels"]

def main():
    st.title("Obesity Level Prediction Prototype")
    st.write("Enter the participant's lifestyle behaviors below to predict their classification.")

    try:
        model, feature_cols, class_labels = load_model()
    except Exception as e:
        st.error(f"❌ Could not load model files. Please run the training cell first. Error: {e}")
        return

    st.subheader("Participant Information")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 12, 65, 25)
        family = st.selectbox("Family History of Obesity", ["Yes", "No"])
        favc = st.selectbox("Frequent High Calorie Food", ["Yes", "No"])
        fcvc = st.selectbox("Vegetable Intake Scale", [1.0, 2.0, 3.0])
        ncp = st.selectbox("Meals per Day", [1.0, 2.0, 3.0])
        caec = st.selectbox("Snacking Between Meals", ["No", "Sometimes", "Frequently", "Always"])

    with col2:
        smoke = st.selectbox("Smoking Status", ["Yes", "No"])
        ch2o = st.selectbox("Water Intake Scale", [1.0, 2.0, 3.0])
        scc = st.selectbox("Track Calories", ["Yes", "No"])
        faf = st.selectbox("Exercise Frequency Scale", [0.0, 1.0, 2.0, 3.0])
        tue = st.selectbox("Screen Time Scale", [0.0, 1.0, 2.0])
        calc = st.selectbox("Alcohol Consumption", ["Don't drink", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportation", ["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"])

    if st.button("Predict Obesity Level", type="primary"):
        caec_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        calc_map = {"Don't drink": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

        # Create dictionary pre-filled with zeros based on training columns
        input_data = {col: 0 for col in feature_cols}

        # Map numerical features
        if 'Age' in input_data: input_data['Age'] = float(age)
        if 'Vegetable Consumption' in input_data: input_data['Vegetable Consumption'] = float(fcvc)
        if 'Number of Main Meals' in input_data: input_data['Number of Main Meals'] = float(ncp)
        if 'Eats Between Meals' in input_data: input_data['Eats Between Meals'] = caec_map[caec]
        if 'Water Consumption' in input_data: input_data['Water Consumption'] = float(ch2o)
        if 'Physical Activity' in input_data: input_data['Physical Activity'] = float(faf)
        if 'Time on Devices' in input_data: input_data['Time on Devices'] = float(tue)
        if 'Alcohol Consumption' in input_data: input_data['Alcohol Consumption'] = calc_map[calc]

        # Map boolean/categorical features
        if 'Gender' in input_data: input_data['Gender'] = 1 if gender == "Male" else 0
        if 'Family History of Overweight' in input_data: input_data['Family History of Overweight'] = 1 if family == "Yes" else 0
        if 'Eats High Caloric Food' in input_data: input_data['Eats High Caloric Food'] = 1 if favc == "Yes" else 0
        if 'Smoker' in input_data: input_data['Smoker'] = 1 if smoke == "Yes" else 0
        if 'Monitors Calories' in input_data: input_data['Monitors Calories'] = 1 if scc == "Yes" else 0

        # Map Transport
        if mtrans == "Bike" and 'Transport: Bike' in input_data: input_data['Transport: Bike'] = 1
        elif mtrans == "Motorbike" and 'Transport: Motorbike' in input_data: input_data['Transport: Motorbike'] = 1
        elif mtrans == "Public Transportation" and 'Transport: Public Transit' in input_data: input_data['Transport: Public Transit'] = 1
        elif mtrans == "Walking" and 'Transport: Walking' in input_data: input_data['Transport: Walking'] = 1

        # Convert to DataFrame and Predict
        input_aligned = pd.DataFrame([input_data])
        pred = int(model.predict(input_aligned)[0])
        proba = model.predict_proba(input_aligned)[0]

        st.success(f"### **Prediction:** {class_labels[pred]}")
        st.progress(float(np.max(proba)))
        st.write(f"**Confidence:** {np.max(proba)*100:.2f}%")

        # Visual Chart of all probabilities
        st.write("---")
        st.write("**Probability Breakdown:**")
        prob_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": proba
        }).sort_values(by="Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Class"))

if __name__ == "__main__":
    main()
