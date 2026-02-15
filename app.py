from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Global variables for model and encoders
model = None
label_encoders = {}
feature_columns = [
    "project_type", 
    "required_seniority", 
    "dev_specialty", 
    "dev_seniority", 
    "dev_workload", 
    "dev_on_leave", 
    "dev_tasks_this_week"
]

def load_model():
    """Load the trained model and label encoders"""
    global model, label_encoders
    
    try:
        # Load the trained model
        model = joblib.load("dev_recommender_model.pkl")
        print("✅ Model loaded successfully")
        
        # Load label encoders (if saved separately)
        # For now, we'll recreate them based on the training data
        # You might want to save and load the encoders as well
        if os.path.exists("label_encoders.pkl"):
            label_encoders = joblib.load("label_encoders.pkl")
            print("✅ Label encoders loaded successfully")
        else:
            print("⚠️ Label encoders file not found. You may need to fit encoders with training data.")
            
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        model = None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model = None

def encode_input_data(input_df):
    """Encode categorical columns using label encoders"""
    encoded_df = input_df.copy()
    
    for col in encoded_df.columns:
        if col in label_encoders and col != 'dev_on_leave':
            try:
                encoded_df[col] = label_encoders[col].transform(encoded_df[col])
            except (ValueError, KeyError) as e:
                # Handle unseen labels by assigning a default value
                print(f"Warning: Unseen label in column {col}: {encoded_df[col].iloc[0]}")
                # You might want to handle this differently based on your requirements
                encoded_df[col] = -1  # or some default value
    
    return encoded_df

def predict_with_rules(sample_dict):
    """
    Make prediction with hard rules
    Returns: 1 for recommended, 0 for not recommended
    """
    # HARD RULE: on leave can NEVER be recommended
    if sample_dict.get("dev_on_leave") == True or sample_dict.get("dev_on_leave") == "True":
        return 0
    
    # If model is not loaded, return None or handle appropriately
    if model is None:
        raise Exception("Model not loaded")
    
    # Convert to DataFrame
    temp_df = pd.DataFrame([sample_dict])
    
    # Encode categorical columns
    temp_df_encoded = encode_input_data(temp_df)
    
    # Make prediction
    prediction = model.predict(temp_df_encoded)[0]
    
    return int(prediction)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Convert boolean string to actual boolean
        if 'dev_on_leave' in data:
            if isinstance(data['dev_on_leave'], str):
                data['dev_on_leave'] = data['dev_on_leave'].lower() == 'true'
        
        # Convert tasks to integer
        if 'dev_tasks_this_week' in data:
            data['dev_tasks_this_week'] = int(data['dev_tasks_this_week'])
        
        # Make prediction with hard rules
        result = predict_with_rules(data)
        
        # Prepare response
        response = {
            'success': True,
            'recommendation': result,
            'message': '✅ Recommended Developer' if result == 1 else '❌ Not Recommended',
            'hard_rule_applied': data.get('dev_on_leave', False)
        }
        
        if request.is_json:
            return jsonify(response)
        else:
            return render_template('result.html', result=response)
            
    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e)
        }
        if request.is_json:
            return jsonify(error_response), 400
        else:
            return render_template('error.html', error=str(e))

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({'error': 'Expected a list of samples'}), 400
        
        results = []
        for sample in data:
            # Convert boolean string to actual boolean
            if 'dev_on_leave' in sample:
                if isinstance(sample['dev_on_leave'], str):
                    sample['dev_on_leave'] = sample['dev_on_leave'].lower() == 'true'
            
            # Convert tasks to integer
            if 'dev_tasks_this_week' in sample:
                sample['dev_tasks_this_week'] = int(sample['dev_tasks_this_week'])
            
            # Make prediction
            result = predict_with_rules(sample)
            results.append({
                'input': sample,
                'recommendation': result,
                'message': '✅ Recommended' if result == 1 else '❌ Not Recommended'
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# Create templates directory and HTML files if they don't exist
def create_templates():
    """Create basic HTML templates for the web interface"""
    import os
    
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Create index.html
    index_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Developer Recommendation System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select, input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .note { color: #666; font-size: 0.9em; margin-top: 20px; }
        .hard-rule { color: #ff0000; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Developer Recommendation System</h1>
    <p class="hard-rule">⚠️ Hard Rule: Developer on leave can NEVER be recommended</p>
    
    <form action="/predict" method="post">
        <div class="form-group">
            <label for="project_type">Project Type:</label>
            <select name="project_type" id="project_type" required>
                <option value="web">Web</option>
                <option value="app">App</option>
                <option value="game">Game</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="required_seniority">Required Seniority:</label>
            <select name="required_seniority" id="required_seniority" required>
                <option value="junior">Junior</option>
                <option value="mid">Mid</option>
                <option value="senior">Senior</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="dev_specialty">Developer Specialty:</label>
            <select name="dev_specialty" id="dev_specialty" required>
                <option value="web">Web</option>
                <option value="app">App</option>
                <option value="game">Game</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="dev_seniority">Developer Seniority:</label>
            <select name="dev_seniority" id="dev_seniority" required>
                <option value="junior">Junior</option>
                <option value="mid">Mid</option>
                <option value="senior">Senior</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="dev_workload">Developer Workload:</label>
            <select name="dev_workload" id="dev_workload" required>
                <option value="free">Free</option>
                <option value="light">Light</option>
                <option value="medium">Medium</option>
                <option value="heavy">Heavy</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="dev_on_leave">Developer on Leave:</label>
            <select name="dev_on_leave" id="dev_on_leave" required>
                <option value="false">No</option>
                <option value="true">Yes</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="dev_tasks_this_week">Tasks This Week:</label>
            <input type="number" name="dev_tasks_this_week" id="dev_tasks_this_week" min="0" max="10" value="0" required>
        </div>
        
        <button type="submit">Get Recommendation</button>
    </form>
    
    <div class="note">
        <p><strong>Note:</strong> This system uses a trained Random Forest model with hard rules.</p>
    </div>
</body>
</html>
    '''
    
    # Create result.html
    result_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Recommendation Result</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; text-align: center; }
        .result-box { padding: 20px; margin: 20px 0; border-radius: 5px; }
        .recommended { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .not-recommended { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .hard-rule { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 10px; margin-top: 20px; }
        .button { display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; margin-top: 20px; }
        .button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <h1>Recommendation Result</h1>
    
    <div class="result-box {{ 'recommended' if result.recommendation == 1 else 'not-recommended' }}">
        <h2>{{ result.message }}</h2>
    </div>
    
    {% if result.hard_rule_applied %}
    <div class="hard-rule">
        <strong>⚠️ Hard Rule Applied:</strong> Developer on leave cannot be recommended.
    </div>
    {% endif %}
    
    <a href="/" class="button">Make Another Prediction</a>
</body>
</html>
    '''
    
    # Create error.html
    error_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; text-align: center; }
        .error-box { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .button { display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Error</h1>
    
    <div class="error-box">
        <h2>Something went wrong!</h2>
        <p>{{ error }}</p>
    </div>
    
    <a href="/" class="button">Go Back</a>
</body>
</html>
    '''
    
    # Write HTML files
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    with open(os.path.join(templates_dir, 'result.html'), 'w') as f:
        f.write(result_html)
    
    with open(os.path.join(templates_dir, 'error.html'), 'w') as f:
        f.write(error_html)
    
    print("✅ Templates created successfully")

if __name__ == '__main__':
    # Create templates
    create_templates()
    
    # Load the model
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
