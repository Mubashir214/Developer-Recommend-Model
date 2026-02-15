import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("dev_recommender_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("👨‍💻 Developer Recommendation System")

st.write("Fill the project & developer details below:")

# Inputs
project_type = st.selectbox("Project Type", ["web", "app", "game"])
required_seniority = st.selectbox("Required Seniority", ["junior", "mid", "senior"])
dev_specialty = st.selectbox("Developer Specialty", ["web", "app", "game"])
dev_seniority = st.selectbox("Developer Seniority", ["junior", "mid", "senior"])
dev_workload = st.selectbox("Developer Workload", ["free", "light", "heavy"])
dev_on_leave = st.checkbox("Developer On Leave?")
dev_tasks_this_week = st.number_input("Tasks This Week", min_value=0, step=1)


def predict_with_rules(sample):
    # HARD RULE
    if sample["dev_on_leave"] == True:
        return 0

    temp = pd.DataFrame([sample])

    for col in temp.columns:
        if col in label_encoders:
            temp[col] = label_encoders[col].transform(temp[col])

    return int(model.predict(temp)[0])


if st.button("Predict"):

    sample = {
        "project_type": project_type,
        "required_seniority": required_seniority,
        "dev_specialty": dev_specialty,
        "dev_seniority": dev_seniority,
        "dev_workload": dev_workload,
        "dev_on_leave": dev_on_leave,
        "dev_tasks_this_week": dev_tasks_this_week
    }

    prediction = predict_with_rules(sample)

    if prediction == 1:
        st.success("✅ Recommended Developer")
    else:
        st.error("❌ Not Recommended")
