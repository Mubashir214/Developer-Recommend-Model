import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import traceback

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

# Initialize session state for model and encoders
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.label_encoders = None

# Function to load model with error handling
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders"""
    try:
        # Check if model file exists
        if not os.path.exists("dev_recommender_model.pkl"):
            st.error("❌ Model file 'dev_recommender_model.pkl' not found!")
            return None, None
        
        # Load model
        model = joblib.load("dev_recommender_model.pkl")
        
        # Load label encoders (create dummy if not exists)
        if os.path.exists("label_encoders.pkl"):
            label_encoders = joblib.load("label_encoders.pkl")
        else:
            # Create dummy encoders if not available
            st.warning("⚠️ Label encoders not found. Using default encoding.")
            label_encoders = create_dummy_encoders()
        
        return model, label_encoders
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

def create_dummy_encoders():
    """Create dummy label encoders if real ones aren't available"""
    encoders = {}
    
    # Define possible values for each categorical column
    categories = {
        "project_type": ["web", "app", "game"],
        "required_seniority": ["junior", "mid", "senior"],
        "dev_specialty": ["web", "app", "game"],
        "dev_seniority": ["junior", "mid", "senior"],
        "dev_workload": ["free", "light", "medium", "heavy"]
    }
    
    for col, values in categories.items():
        le = LabelEncoder()
        le.fit(values)
        encoders[col] = le
    
    return encoders

def predict_with_rules(sample_dict, model, label_encoders):
    """
    Make prediction with hard rules
    Returns: 1 for recommended, 0 for not recommended, and reason
    """
    # HARD RULE: on leave can NEVER be recommended
    if sample_dict.get("dev_on_leave") == True or sample_dict.get("dev_on_leave") == "True" or sample_dict.get("dev_on_leave") == "Yes":
        return 0, "Hard Rule: Developer is on leave"
    
    # Check if model is loaded
    if model is None:
        return 0, "Model not loaded"
    
    try:
        # Convert to DataFrame
        temp_df = pd.DataFrame([sample_dict])
        
        # Encode categorical columns
        for col in temp_df.columns:
            if col in label_encoders and col != 'dev_on_leave' and col in temp_df.columns:
                try:
                    temp_df[col] = label_encoders[col].transform(temp_df[col])
                except Exception as e:
                    # If encoding fails, use a default value
                    st.warning(f"Encoding issue with column {col}: {str(e)}")
                    temp_df[col] = 0  # Default value
        
        # Make prediction
        prediction = model.predict(temp_df)[0]
        
        # Add reasoning
        if prediction == 1:
            reason = "ML model recommends this developer"
        else:
            # Try to determine why
            if sample_dict["project_type"] != sample_dict["dev_specialty"]:
                reason = f"Specialty mismatch: Need {sample_dict['project_type']}, Have {sample_dict['dev_specialty']}"
            elif sample_dict["required_seniority"] == "senior" and sample_dict["dev_seniority"] == "mid":
                reason = "Under-qualified: Need senior, have mid"
            elif sample_dict["dev_workload"] == "heavy" or sample_dict["dev_tasks_this_week"] >= 4:
                reason = "Overloaded: Too many tasks or heavy workload"
            else:
                reason = "ML model does not recommend this developer"
        
        return int(prediction), reason
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0, f"Error in prediction: {str(e)}"

# Load model
with st.spinner("Loading model..."):
    model, label_encoders = load_model_and_encoders()

if model is not None:
    st.success("✅ Model loaded successfully!")
    st.session_state.model_loaded = True
    st.session_state.model = model
    st.session_state.label_encoders = label_encoders

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

# Create columns for grid display
cols = st.columns(2)

for idx, test_case in enumerate(test_cases):
    with cols[idx % 2]:
        with st.container():
            st.subheader(f"Case {idx+1}: {test_case['name']}")
            
            # Display input data
            with st.expander("View Input Data"):
                st.json(test_case['data'])
            
            # Make prediction
            if st.session_state.model_loaded:
                result, reason = predict_with_rules(
                    test_case['data'], 
                    st.session_state.model, 
                    st.session_state.label_encoders
                )
                
                # Display result with color
                if result == 1:
                    st.success(f"✅ **RECOMMENDED**")
                    st.caption(f"Reason: {reason}")
                else:
                    st.error(f"❌ **NOT RECOMMENDED**")
                    st.caption(f"Reason: {reason}")
            else:
                st.warning("Model not loaded. Cannot make prediction.")
            
            st.markdown("---")

# Summary Table
st.header("📈 Summary")

if st.session_state.model_loaded:
    results_data = []
    for idx, test_case in enumerate(test_cases):
        result, reason = predict_with_rules(
            test_case['data'], 
            st.session_state.model, 
            st.session_state.label_encoders
        )
        results_data.append({
            "Case": f"Case {idx+1}",
            "Description": test_case['name'],
            "Result": "✅ Recommended" if result == 1 else "❌ Not Recommended",
            "Reason": reason
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Statistics
    st.header("📊 Statistics")
    
    total = len(results_data)
    recommended = sum(1 for r in results_data if "✅" in r["Result"])
    not_recommended = total - recommended
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cases", total)
    with col2:
        st.metric("✅ Recommended", recommended)
    with col3:
        st.metric("❌ Not Recommended", not_recommended)
    with col4:
        rate = (recommended/total)*100
        st.metric("Recommendation Rate", f"{rate:.1f}%")
    
    # Progress bar
    st.progress(recommended/total)
else:
    st.warning("⚠️ Model not loaded. Please check the model files.")

# Footer
st.markdown("---")
st.markdown("### 📝 Notes")
st.markdown("""
- **Hard Rule**: Developers on leave are automatically not recommended
- **Model**: Random Forest Classifier
- **Features**: Project type, seniority requirements, developer skills, workload, etc.
""")
