import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Model & Encoders
# ==============================

model = joblib.load("dev_recommender_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Developer Recommender", layout="centered")

st.title("🚀 Developer Recommendation System")

st.write("Fill the details below to check if developer is recommended.")

# ==============================
# User Inputs
# ==============================

project_type = st.selectbox(
    "Project Type",
    label_encoders["project_type"].classes_
)

required_seniority = st.selectbox(
    "Required Seniority",
    label_encoders["required_seniority"].classes_
)

dev_specialty = st.selectbox(
    "Developer Specialty",
    label_encoders["dev_specialty"].classes_
)

dev_seniority = st.selectbox(
    "Developer Seniority",
    label_encoders["dev_seniority"].classes_
)

dev_workload = st.selectbox(
    "Developer Workload",
    label_encoders["dev_workload"].classes_
)

dev_on_leave = st.checkbox("Developer is on Leave")

dev_tasks_this_week = st.number_input(
    "Tasks This Week",
    min_value=0,
    max_value=20,
    value=0
)

# ==============================
# Prediction Button
# ==============================

if st.button("Predict Recommendation"):

    # HARD RULE
    if dev_on_leave:
        st.error("❌ Not Recommended (Developer is on leave - Hard Rule Applied)")
    else:
        # Create dataframe
        input_data = pd.DataFrame([{
            "project_type": project_type,
            "required_seniority": required_seniority,
            "dev_specialty": dev_specialty,
            "dev_seniority": dev_seniority,
            "dev_workload": dev_workload,
            "dev_on_leave": dev_on_leave,
            "dev_tasks_this_week": dev_tasks_this_week
        }])

        # Encode
        for col in input_data.columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        # Predict
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Recommended Developer")
        else:
            st.error("❌ Not Recommended")
