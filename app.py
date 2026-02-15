import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Developer Recommendation System",
    page_icon="👨‍💻",
    layout="wide"
)

# Title and description
st.title("👨‍💻 Developer Recommendation System")
st.markdown("---")

# Hard rule warning
st.warning("⚠️ **HARD RULE**: Developer on leave can NEVER be recommended")

# Function to load model and encoders
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders"""
    try:
        # Load model
        model = joblib.load("dev_recommender_model.pkl")
        
        # Load label encoders
        label_encoders = joblib.load("label_encoders.pkl")
        
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
with st.spinner("Loading model..."):
    model, label_encoders = load_model_and_encoders()

if model is not None:
    st.success("✅ Model loaded successfully!")

# Prediction function with hard rule
def predict_with_rules(sample_dict):
    """
    Make prediction with hard rules
    Returns: 1 for recommended, 0 for not recommended
    """
    # HARD RULE: on leave can NEVER be recommended
    if sample_dict["dev_on_leave"] == True or sample_dict["dev_on_leave"] == "Yes":
        return 0, "Hard Rule: Developer is on leave"
    
    try:
        # Convert to DataFrame
        temp_df = pd.DataFrame([sample_dict])
        
        # Encode categorical columns
        for col in temp_df.columns:
            if col in label_encoders and col != 'dev_on_leave':
                temp_df[col] = label_encoders[col].transform(temp_df[col])
        
        # Make prediction
        prediction = model.predict(temp_df)[0]
        
        # Add reasoning
        if prediction == 1:
            reason = "ML model recommends this developer"
        else:
            # Try to determine why
            if sample_dict["project_type"] != sample_dict["dev_specialty"]:
                reason = f"Specialty mismatch: Need {sample_dict['project_type']}, Have {sample_dict['dev_specialty']}"
            elif (sample_dict["required_seniority"] == "senior" and 
                  sample_dict["dev_seniority"] == "mid"):
                reason = "Under-qualified: Need senior, have mid"
            elif (sample_dict["dev_workload"] == "heavy" or 
                  sample_dict["dev_tasks_this_week"] >= 4):
                reason = "Overloaded: Too many tasks or heavy workload"
            else:
                reason = "ML model does not recommend this developer"
        
        return int(prediction), reason
    
    except Exception as e:
        return 0, f"Error in prediction: {str(e)}"

# Test cases
st.header("📊 Test Cases Results")

test_cases = [
    # 1. Perfect match
    {
        "name": "Perfect Match",
        "data": {
            "project_type": "web",
            "required_seniority": "mid",
            "dev_specialty": "web",
            "dev_seniority": "senior",
            "dev_workload": "light",
            "dev_on_leave": False,
            "dev_tasks_this_week": 1
        }
    },
    # 2. Wrong specialty
    {
        "name": "Wrong Specialty",
        "data": {
            "project_type": "game",
            "required_seniority": "mid",
            "dev_specialty": "web",
            "dev_seniority": "senior",
            "dev_workload": "free",
            "dev_on_leave": False,
            "dev_tasks_this_week": 0
        }
    },
    # 3. On leave (hard rule)
    {
        "name": "On Leave (Hard Rule)",
        "data": {
            "project_type": "app",
            "required_seniority": "junior",
            "dev_specialty": "app",
            "dev_seniority": "junior",
            "dev_workload": "free",
            "dev_on_leave": True,
            "dev_tasks_this_week": 0
        }
    },
    # 4. Under-qualified
    {
        "name": "Under-qualified",
        "data": {
            "project_type": "web",
            "required_seniority": "senior",
            "dev_specialty": "web",
            "dev_seniority": "mid",
            "dev_workload": "free",
            "dev_on_leave": False,
            "dev_tasks_this_week": 1
        }
    },
    # 5. Overloaded
    {
        "name": "Overloaded",
        "data": {
            "project_type": "game",
            "required_seniority": "mid",
            "dev_specialty": "game",
            "dev_seniority": "senior",
            "dev_workload": "heavy",
            "dev_on_leave": False,
            "dev_tasks_this_week": 5
        }
    }
]

# Display test cases in a grid
for i in range(0, len(test_cases), 2):
    cols = st.columns(2)
    
    for j in range(2):
        if i + j < len(test_cases):
            test_case = test_cases[i + j]
            with cols[j]:
                with st.container():
                    st.subheader(f"Case {i+j+1}: {test_case['name']}")
                    
                    # Display input data
                    with st.expander("View Input Data"):
                        st.json(test_case['data'])
                    
                    # Make prediction
                    if model is not None:
                        result, reason = predict_with_rules(test_case['data'])
                        
                        # Display result with color
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if result == 1:
                                st.markdown("# ✅")
                            else:
                                st.markdown("# ❌")
                        with col2:
                            if result == 1:
                                st.success("**RECOMMENDED**")
                            else:
                                st.error("**NOT RECOMMENDED**")
                            st.caption(f"*{reason}*")
                    else:
                        st.warning("Model not loaded")
                    
                    st.markdown("---")

# Summary Table
st.header("📈 Summary Report")

if model is not None:
    # Create summary dataframe
    summary_data = []
    for i, test_case in enumerate(test_cases, 1):
        result, reason = predict_with_rules(test_case['data'])
        summary_data.append({
            "Case": f"Case {i}",
            "Description": test_case['name'],
            "Result": "✅ Recommended" if result == 1 else "❌ Not Recommended",
            "Reason": reason
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Statistics
    st.header("📊 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_cases = len(summary_data)
    recommended = sum(1 for row in summary_data if "✅" in row["Result"])
    not_recommended = total_cases - recommended
    
    with col1:
        st.metric("Total Cases", total_cases)
    with col2:
        st.metric("✅ Recommended", recommended)
    with col3:
        st.metric("❌ Not Recommended", not_recommended)
    with col4:
        rate = (recommended/total_cases)*100
        st.metric("Success Rate", f"{rate:.1f}%")
    
    # Progress bar
    st.progress(recommended/total_cases)
    st.caption(f"Recommendation Rate: {rate:.1f}%")

# Interactive Prediction Section
st.header("🎯 Try Your Own Prediction")
st.mark("Test with custom values:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        project_type = st.selectbox("Project Type", ["web", "app", "game"])
        required_seniority = st.selectbox("Required Seniority", ["junior", "mid", "senior"])
        dev_specialty = st.selectbox("Developer Specialty", ["web", "app", "game"])
        
    with col2:
        dev_seniority = st.selectbox("Developer Seniority", ["junior", "mid", "senior"])
        dev_workload = st.selectbox("Developer Workload", ["free", "light", "medium", "heavy"])
        dev_on_leave = st.selectbox("Developer on Leave?", ["No", "Yes"])
        dev_tasks_this_week = st.number_input("Tasks This Week", min_value=0, max_value=10, value=0)
    
    submitted = st.form_submit_button("Get Recommendation")
    
    if submitted and model is not None:
        custom_case = {
            "project_type": project_type,
            "required_seniority": required_seniority,
            "dev_specialty": dev_specialty,
            "dev_seniority": dev_seniority,
            "dev_workload": dev_workload,
            "dev_on_leave": dev_on_leave == "Yes",
            "dev_tasks_this_week": dev_tasks_this_week
        }
        
        result, reason = predict_with_rules(custom_case)
        
        st.markdown("---")
        st.subheader("Prediction Result:")
        
        if result == 1:
            st.success(f"✅ **RECOMMENDED** - {reason}")
        else:
            st.error(f"❌ **NOT RECOMMENDED** - {reason}")

# Footer
st.markdown("---")
st.markdown("### 📝 About This System")
st.markdown("""
- **Model**: Random Forest Classifier (trained on historical data)
- **Hard Rule**: Developers on leave are automatically not recommended
- **Features Used**:
  - Project Type & Required Seniority
  - Developer Specialty & Seniority
  - Current Workload & Tasks
  - Leave Status
""")
