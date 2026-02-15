import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import subprocess

# Page configuration
st.set_page_config(
    page_title="Developer Recommendation System",
    page_icon="👨‍💻",
    layout="wide"
)

# Title
st.title("👨‍💻 Developer Recommendation System")
st.markdown("---")

# Debug section - Check installed packages
with st.expander("🔧 Debug Information", expanded=False):
    st.write("### System Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Python executable: {sys.executable}")
    
    st.write("### Installed Packages")
    packages = ['streamlit', 'pandas', 'numpy', 'sklearn', 'joblib']
    for package in packages:
        try:
            if package == 'sklearn':
                import sklearn
                st.write(f"scikit-learn: {sklearn.__version__}")
            else:
                module = __import__(package)
                st.write(f"{package}: {module.__version__}")
        except Exception as e:
            st.write(f"{package}: Not installed - {str(e)}")

# Hard rule warning
st.warning("⚠️ **HARD RULE**: Developer on leave can NEVER be recommended")

# Function to load model and encoders
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders"""
    try:
        # Check if files exist
        if not os.path.exists("dev_recommender_model.pkl"):
            st.error("❌ Model file not found!")
            st.info("Please upload dev_recommender_model.pkl to the app directory")
            return None, None
        
        if not os.path.exists("label_encoders.pkl"):
            st.error("❌ Label encoders file not found!")
            st.info("Please upload label_encoders.pkl to the app directory")
            return None, None
        
        # Get file sizes
        model_size = os.path.getsize("dev_recommender_model.pkl") / (1024*1024)  # MB
        encoder_size = os.path.getsize("label_encoders.pkl") / (1024*1024)  # MB
        
        st.write(f"Model file size: {model_size:.2f} MB")
        st.write(f"Encoders file size: {encoder_size:.2f} MB")
        
        # Load model
        with st.spinner("Loading model..."):
            model = joblib.load("dev_recommender_model.pkl")
            label_encoders = joblib.load("label_encoders.pkl")
        
        st.success("✅ Model loaded successfully!")
        return model, label_encoders
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.exception(e)
        return None, None

# Load model
model, label_encoders = load_model_and_encoders()

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
        
        # Debug info
        with st.expander("🔍 Prediction Debug", expanded=False):
            st.write("Input data:", sample_dict)
            
            # Encode categorical columns
            for col in temp_df.columns:
                if col in label_encoders and col != 'dev_on_leave':
                    try:
                        original_value = temp_df[col].iloc[0]
                        temp_df[col] = label_encoders[col].transform(temp_df[col])
                        encoded_value = temp_df[col].iloc[0]
                        st.write(f"{col}: '{original_value}' → {encoded_value}")
                    except Exception as e:
                        st.write(f"Error encoding {col}: {str(e)}")
                        # Use default value
                        temp_df[col] = 0
            
            st.write("Encoded data:", temp_df.to_dict('records')[0])
        
        # Make prediction
        prediction = model.predict(temp_df)[0]
        probability = model.predict_proba(temp_df)[0]
        
        with st.expander("📊 Model Output", expanded=False):
            st.write(f"Prediction: {prediction}")
            st.write(f"Probability: {probability}")
            st.write(f"Not Recommended: {probability[0]:.3f}, Recommended: {probability[1]:.3f}")
        
        # Add reasoning
        if prediction == 1:
            reason = f"ML model recommends (confidence: {probability[1]:.1%})"
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
                reason = f"ML model does not recommend (confidence: {probability[0]:.1%})"
        
        return int(prediction), reason
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.exception(e)
        return 0, f"Error: {str(e)}"

# Main content
if model is None:
    st.error("❌ Model could not be loaded. Please check the debug information above.")
    st.info("""
    **Troubleshooting steps:**
    1. Make sure both files are uploaded:
       - dev_recommender_model.pkl
       - label_encoders.pkl
    2. Check file permissions
    3. Verify the files are not corrupted
    """)
else:
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

    # Display test cases
    for i, test_case in enumerate(test_cases, 1):
        with st.container():
            st.subheader(f"Case {i}: {test_case['name']}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Input:**")
                st.json(test_case['data'])
            
            with col2:
                st.write("**Result:**")
                result, reason = predict_with_rules(test_case['data'])
                
                if result == 1:
                    st.success(f"✅ **RECOMMENDED**")
                else:
                    st.error(f"❌ **NOT RECOMMENDED**")
                st.caption(f"*{reason}*")
            
            st.markdown("---")

    # Summary Table
    st.header("📈 Summary")

    summary_data = []
    for i, test_case in enumerate(test_cases, 1):
        result, reason = predict_with_rules(test_case['data'])
        summary_data.append({
            "Case": f"Case {i}",
            "Description": test_case['name'],
            "Result": "✅ Recommended" if result == 1 else "❌ Not Recommended",
            "Reason": reason.split('(')[0].strip()  # Clean reason
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Statistics
    col1, col2, col3 = st.columns(3)
    
    total = len(summary_data)
    recommended = sum(1 for row in summary_data if "✅" in row["Result"])
    
    with col1:
        st.metric("Total Cases", total)
    with col2:
        st.metric("✅ Recommended", recommended)
    with col3:
        st.metric("❌ Not Recommended", total - recommended)

# Expected output note
st.info("""
### Expected Output:
- **Case 1**: ✅ Recommended (Perfect match)
- **Case 2**: ❌ Not Recommended (Wrong specialty)
- **Case 3**: ❌ Not Recommended (On leave - Hard Rule)
- **Case 4**: ❌ Not Recommended (Under-qualified)
- **Case 5**: ❌ Not Recommended (Overloaded)
""")
