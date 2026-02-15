import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load("dev_recommender_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, label_encoders

model, label_encoders = load_artifacts()

st.title("👨‍💻 Developer Recommendation System")
st.write("Fill the project & developer details below:")

project_type = st.selectbox("Project Type", ["web", "app", "game"])
required_seniority = st.selectbox("Required Seniority", ["junior", "mid", "senior"])
dev_specialty = st.selectbox("Developer Specialty", ["web", "app", "game"])
dev_seniority = st.selectbox("Developer Seniority", ["junior", "mid", "senior"])
dev_workload = st.selectbox("Developer Workload", ["free", "light", "heavy"])
dev_on_leave = st.checkbox("Developer On Leave?")
dev_tasks_this_week = st.number_input("Tasks This Week", min_value=0, step=1)

def predict_with_rules(sample):

    if sample["dev_on_leave"]:
        return 0

    temp = pd.DataFrame([sample])

    for col in temp.columns:
        if col in label_encoders:
            temp[col] = label_encoders[col].transform(temp[col])

    temp = temp[model.feature_names_in_]

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
