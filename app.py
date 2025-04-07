from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import random

app = Flask(__name__)

# Context processor to make datetime available in templates
@app.context_processor
def inject_now():
    return {'now': datetime.now}

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'models/best_model.pkl')
preprocessing_artifacts_path = os.path.join(os.path.dirname(__file__), 'models/preprocessing_artifacts.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(preprocessing_artifacts_path, 'rb') as f:
        preprocessing_artifacts = pickle.load(f)
    
    model_loaded = True
    print("Model loaded successfully!")
except (FileNotFoundError, EOFError) as e:
    model_loaded = False
    print(f"Model or preprocessing artifacts not found or corrupted. Error: {e}")
    print(f"Model path: {model_path}")
    print(f"Preprocessing artifacts path: {preprocessing_artifacts_path}")

# Create or load the synthetic dataset for comparisons and analysis
def create_sample_credit_data(n_samples=5000):
    np.random.seed(42)
    
    # Create features that would typically be used in credit risk assessment
    data = {
        'age': np.random.normal(40, 10, n_samples).clip(18, 75).astype(int),
        'income': np.random.lognormal(10, 1, n_samples).clip(10000, 500000),
        'employment_years': np.random.gamma(5, 1, n_samples).clip(0, 40).astype(int),
        'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 0.6,
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850).astype(int),
        'loan_amount': np.random.lognormal(10, 0.5, n_samples).clip(1000, 100000),
        'loan_term': np.random.choice([12, 24, 36, 48, 60, 72, 84, 96, 120], n_samples),
        'loan_purpose': np.random.choice(['home', 'car', 'education', 'medical', 'business', 'other'], n_samples),
        'has_mortgage': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'has_credit_card': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'previous_defaults': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.7, 0.1, 0.08, 0.05, 0.04, 0.03]),
        'credit_inquiries': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_samples, p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate default probability for the synthetic dataset
    default_prob = (
        (850 - df['credit_score']) / 850 * 0.5 +
        df['debt_to_income_ratio'] * 0.2 +
        (np.log(500000) - np.log(df['income'].clip(10000))) / np.log(500000) * 0.1 +
        df['previous_defaults'] / 5 * 0.1 +
        df['credit_inquiries'] / 10 * 0.1
    ) * 0.7
    
    # Add some randomness
    default_prob += np.random.normal(0, 0.1, n_samples)
    default_prob = default_prob.clip(0, 1)
    df['default_prob'] = default_prob
    
    # Create binary target
    df['default'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Add derived features
    df['payment_to_income'] = (df['loan_amount'] / df['loan_term']) / (df['income'] / 12)
    
    return df

# Load or create sample data for comparisons and analysis
comparison_data_path = os.path.join(os.path.dirname(__file__), 'comparison_data.csv')
if os.path.exists(comparison_data_path):
    comparison_data = pd.read_csv(comparison_data_path)
    print("Comparison data loaded from file")
else:
    comparison_data = create_sample_credit_data(5000)
    comparison_data.to_csv(comparison_data_path, index=False)
    print("Generated new comparison data and saved to file")

# Helper function to get risk category based on probability
def get_risk_category(probability):
    if probability < 0.2:
        return "Very Low Risk", "green"
    elif probability < 0.4:
        return "Low Risk", "blue"
    elif probability < 0.6:
        return "Moderate Risk", "orange" 
    elif probability < 0.8:
        return "High Risk", "red"
    else:
        return "Very High Risk", "darkred"

# Helper function to add derived features required by the model
def add_derived_features(data):
    """Add any derived features that the model expects but aren't in the input data"""
    # Create a copy to avoid modifying the original
    data_copy = data.copy()
    
    # Apply the same transformations that were used during model training
    if 'preprocessing_artifacts' in globals() and preprocessing_artifacts:
        # Check if payment_to_income is required but not present
        if 'payment_to_income' not in data_copy and 'payment_to_income' in preprocessing_artifacts.get('feature_names', []):
            data_copy['payment_to_income'] = (data_copy['loan_amount'] / data_copy['loan_term']) / (data_copy['income'] / 12)
    else:
        # Fallback to default derived features if artifacts not available
        if 'payment_to_income' not in data_copy:
            data_copy['payment_to_income'] = (data_copy['loan_amount'] / data_copy['loan_term']) / (data_copy['income'] / 12)
    
    return data_copy

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Dashboard page
@app.route('/dashboard')
def dashboard():
    # In a real application, this data would come from a database
    metrics = {
        'total_assessments': 110,
        'low_risk_percent': 75,
        'high_risk_percent': 25,
        'avg_loan_amount': '24,750'
    }
    
    # Sample recent assessments data
    recent_assessments = [
        {
            'id': 'CR12845',
            'date': '2025-02-15',
            'loan_amount': '320,000',
            'term': 360,
            'grade': 'A',
            'risk_score': 15,
            'status': 'Approved'
        },
        {
            'id': 'CR12844',
            'date': '2025-02-16',
            'loan_amount': '45,000',
            'term': 60,
            'grade': 'C',
            'risk_score': 55,
            'status': 'Review'
        },
        {
            'id': 'CR12843',
            'date': '2025-01-11',
            'loan_amount': '32,500',
            'term': 48,
            'grade': 'B',
            'risk_score': 25,
            'status': 'Approved'
        },
        {
            'id': 'CR12842',
            'date': '2025-01-09',
            'loan_amount': '18,750',
            'term': 36,
            'grade': 'F',
            'risk_score': 75,
            'status': 'Denied'
        },
        {
            'id': 'CR12841',
            'date': '2024-012-27',
            'loan_amount': '125,000',
            'term': 84,
            'grade': 'D',
            'risk_score': 45,
            'status': 'Review'
        }
    ]
    
    return render_template('dashboard.html', metrics=metrics, recent_assessments=recent_assessments)

# Model metrics page
@app.route('/model_metrics')
def model_metrics():
    # Check if model metadata exists
    metadata_path = os.path.join(os.path.dirname(__file__), 'models/model_metrics.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metrics = json.load(f)
    else:
        # Create default metrics if file doesn't exist
        metrics = {
            'model_type': 'Random Forest',
            'roc_auc': 0.85,
            'accuracy': 0.86,
            'precision': 0.75,
            'recall': 0.68,
            'f1': 0.71,
            'training_size': 1600,
            'feature_count': 13,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    # Check if visualization files exist
    feature_importance_path = os.path.join(os.path.dirname(__file__), 'models/feature_importance.png')
    roc_path = os.path.join(os.path.dirname(__file__), 'static/images/roc_curves.png')
    
    # Use the feature importance from the models folder if available
    if os.path.exists(feature_importance_path):
        # Copy to static directory if needed
        static_feature_path = os.path.join(os.path.dirname(__file__), 'static/images/feature_importance.png')
        if not os.path.exists(os.path.dirname(static_feature_path)):
            os.makedirs(os.path.dirname(static_feature_path), exist_ok=True)
        
        if not os.path.exists(static_feature_path):
            import shutil
            shutil.copy2(feature_importance_path, static_feature_path)
    
    roc_curve_exists = os.path.exists(roc_path)
    shap_exists = os.path.exists(feature_importance_path)
    
    return render_template('model_metrics.html', 
                          metrics=metrics, 
                          roc_curve_exists=roc_curve_exists,
                          shap_exists=shap_exists)

# Feature Importance visualization helper
def create_feature_importance_chart(input_data):
    # Check if we have a feature importance file from the model training
    feature_importance_path = os.path.join(os.path.dirname(__file__), 'models/feature_importance.csv')
    if os.path.exists(feature_importance_path):
        # Use pre-computed feature importances
        features_df = pd.read_csv(feature_importance_path)
        features_df = features_df.sort_values('Importance', ascending=False).head(10)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
        plt.title('Top 10 Feature Importances for Credit Risk Prediction', fontsize=14)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
    
    # If we don't have a pre-computed file but model has feature_importances_    
    elif hasattr(model, 'feature_importances_'):
        # Get feature importances directly from model
        importances = model.feature_importances_
        
        # Get feature names from preprocessing artifacts if available
        if 'preprocessing_artifacts' in globals() and preprocessing_artifacts:
            feature_names = preprocessing_artifacts.get('feature_names', [])
        else:
            # Fallback to generic feature names
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Ensure feature_names and importances have the same length
        if len(feature_names) != len(importances):
            # Truncate the longer array to match the shorter one
            min_length = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_length]
            importances = importances[:min_length]
            print(f"Warning: Truncated feature arrays to length {min_length} to match")
        
        # Create a DataFrame with feature names and importances
        features_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        features_df = features_df.sort_values('Importance', ascending=False).head(10)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
        plt.title('Top 10 Feature Importances for Credit Risk Prediction', fontsize=14)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
    
    # Fallback method if importances are not directly available
    else:
        # Create a bar chart with synthetic feature importances based on domain knowledge
        features = [
            'credit_score', 
            'debt_to_income_ratio', 
            'previous_defaults', 
            'income', 
            'loan_amount',
            'payment_to_income', 
            'employment_years', 
            'credit_inquiries', 
            'loan_term', 
            'has_mortgage'
        ]
        
        importances = [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01]
        
        features_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
        plt.title('Feature Importances for Credit Risk Prediction (Estimated)', fontsize=14)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
    
    # Save to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Encode as base64 for embedding in HTML
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    return graphic

# Risk Comparison helper function
def create_risk_comparison_chart(input_data):
    # Filter comparison data based on demographics
    age = input_data['age']
    income = input_data['income']
    loan_purpose = input_data['loan_purpose']
    
    # Define age and income ranges for comparison
    age_min = max(18, age - 10)
    age_max = min(75, age + 10)
    income_min = max(10000, income * 0.5)
    income_max = income * 1.5
    
    # Filter comparison data
    similar_profiles = comparison_data[
        (comparison_data['age'] >= age_min) & 
        (comparison_data['age'] <= age_max) &
        (comparison_data['income'] >= income_min) & 
        (comparison_data['income'] <= income_max) &
        (comparison_data['loan_purpose'] == loan_purpose)
    ]
    
    # If not enough similar profiles, expand the criteria
    if len(similar_profiles) < 50:
        similar_profiles = comparison_data[
            (comparison_data['age'] >= age_min) & 
            (comparison_data['age'] <= age_max)
        ]
    
    # Use the model to predict probabilities for the subset
    input_df = pd.DataFrame([input_data])
    
    # Add derived features
    input_df = add_derived_features(input_df)
    
    input_prob = model.predict_proba(input_df)[0, 1]
    
    # Create histogram of default probabilities
    plt.figure(figsize=(10, 6))
    
    # Plot distribution of default probabilities from comparison group
    sns.histplot(similar_profiles['default_prob'], bins=20, kde=True, color='blue', alpha=0.6)
    
    # Add vertical line for the user's probability
    plt.axvline(x=input_prob, color='red', linestyle='--', linewidth=2, 
                label=f'Your Risk Score: {input_prob:.2f}')
    
    # Add percentile annotation
    percentile = (similar_profiles['default_prob'] <= input_prob).mean() * 100
    plt.text(input_prob + 0.02, plt.gca().get_ylim()[1] * 0.9, 
             f'You are at the {percentile:.1f}th percentile', 
             color='red', fontsize=12, verticalalignment='top')
    
    plt.title('Your Risk Score Compared to Similar Applicants', fontsize=14)
    plt.xlabel('Default Probability', fontsize=12)
    plt.ylabel('Number of Similar Applicants', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    # Save to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Encode as base64 for embedding in HTML
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    return graphic, percentile

# Loan Term Calculator helper function
def recommend_loan_terms(input_data, probability):
    # Extract key variables
    loan_amount = input_data['loan_amount']
    credit_score = input_data['credit_score']
    
    # Base interest rates based on risk probability
    if probability < 0.2:  # Very Low Risk
        base_rate = 0.035  # 3.5%
    elif probability < 0.4:  # Low Risk
        base_rate = 0.045  # 4.5%
    elif probability < 0.6:  # Moderate Risk
        base_rate = 0.06   # 6%
    elif probability < 0.8:  # High Risk
        base_rate = 0.08   # 8%
    else:  # Very High Risk
        base_rate = 0.11   # 11%
    
    # Adjust based on credit score
    if credit_score >= 750:
        rate_adjustment = -0.005  # -0.5%
    elif credit_score >= 700:
        rate_adjustment = -0.0025  # -0.25%
    elif credit_score >= 650:
        rate_adjustment = 0
    elif credit_score >= 600:
        rate_adjustment = 0.005  # +0.5%
    else:
        rate_adjustment = 0.01  # +1%
    
    # Adjust based on loan amount (higher loans get slight discount)
    if loan_amount >= 100000:
        amount_adjustment = -0.0025  # -0.25%
    elif loan_amount >= 50000:
        amount_adjustment = -0.001  # -0.1%
    elif loan_amount >= 10000:
        amount_adjustment = 0
    else:
        amount_adjustment = 0.002  # +0.2%
    
    # Calculate final rate
    final_rate = base_rate + rate_adjustment + amount_adjustment
    
    # Calculate monthly payments for different terms
    terms = [12, 24, 36, 48, 60, 72, 84, 96, 120]
    recommendations = []
    
    for term in terms:
        # Adjustment for term length
        term_adjustment = (term / 12 - 1) * 0.001  # Longer terms have slightly higher rates
        term_rate = final_rate + term_adjustment
        
        # Calculate monthly payment using loan formula
        monthly_rate = term_rate / 12
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** term) / ((1 + monthly_rate) ** term - 1)
        
        # Calculate total cost
        total_cost = monthly_payment * term
        total_interest = total_cost - loan_amount
        
        # Check if this term should be recommended based on risk
        recommended = False
        if probability < 0.2:
            # Very low risk - recommend longer terms for lower monthly payments
            recommended = term >= 60
        elif probability < 0.4:
            # Low risk - recommend medium to long terms
            recommended = 36 <= term <= 84
        elif probability < 0.6:
            # Moderate risk - recommend medium terms
            recommended = 24 <= term <= 60
        elif probability < 0.8:
            # High risk - recommend shorter terms
            recommended = 12 <= term <= 36
        else:
            # Very high risk - recommend only shortest terms if any
            recommended = term <= 24
        
        recommendations.append({
            'term': term,
            'monthly_payment': round(monthly_payment, 2),
            'total_cost': round(total_cost, 2),
            'total_interest': round(total_interest, 2),
            'interest_rate': round(term_rate * 100, 2),
            'recommended': recommended
        })
    
    return recommendations

# Prediction API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get the data from the request
        input_data = request.get_json()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Add derived features if needed
        input_df = add_derived_features(input_df)
        
        # Process the input according to preprocessing artifacts if available
        if 'preprocessing_artifacts' in globals() and preprocessing_artifacts:
            # Handle categorical features using encoders if available
            encoders = preprocessing_artifacts.get('encoders', {})
            for col in input_df.columns:
                if col in encoders:
                    try:
                        input_df[col] = encoders[col].transform(input_df[col])
                    except:
                        print(f"Warning: Could not transform {col} with encoder")
            
            # Scale numerical features if scaler available
            if 'scaler' in preprocessing_artifacts:
                num_cols = [col for col in preprocessing_artifacts.get('numerical_cols', []) if col in input_df.columns]
                if num_cols:
                    try:
                        input_df[num_cols] = preprocessing_artifacts['scaler'].transform(input_df[num_cols])
                    except:
                        print(f"Warning: Could not scale numerical features")
        
        # Make prediction
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0].tolist()
        
        # Return the result
        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities,
            'prediction_text': 'Default' if prediction == 1 else 'Non-default'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

# Web form prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return render_template('index.html', 
                              error="Model not loaded. Please train the model first.",
                              model_loaded=model_loaded)
    
    try:
        # Get form data
        form_data = {k: float(v) if k not in ['loan_purpose'] else v 
                     for k, v in request.form.items()}
        
        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])
        
        # Save raw form data for display
        raw_data = form_data.copy()
        
        # Add derived features
        input_df = add_derived_features(input_df)
        
        # Process the input according to preprocessing artifacts if available
        if 'preprocessing_artifacts' in globals() and preprocessing_artifacts:
            # Make a copy for prediction that will be transformed
            prediction_df = input_df.copy()
            
            # Handle categorical features using encoders if available
            encoders = preprocessing_artifacts.get('encoders', {})
            for col in prediction_df.columns:
                if col in encoders:
                    try:
                        prediction_df[col] = encoders[col].transform(prediction_df[col])
                    except:
                        print(f"Warning: Could not transform {col} with encoder")
            
            # Scale numerical features if scaler available
            if 'scaler' in preprocessing_artifacts:
                num_cols = [col for col in preprocessing_artifacts.get('numerical_cols', []) if col in prediction_df.columns]
                if num_cols:
                    try:
                        prediction_df[num_cols] = preprocessing_artifacts['scaler'].transform(prediction_df[num_cols])
                    except:
                        print(f"Warning: Could not scale numerical features")
            
            # Make prediction with transformed data
            prediction = int(model.predict(prediction_df)[0])
            probabilities = model.predict_proba(prediction_df)[0].tolist()
        else:
            # Fallback to direct prediction if no artifacts
            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0].tolist()
        
        # Get probabilities for default (usually index 1, but could be 0 depending on model)
        # For binary classification with classes [0, 1], index 1 is probability of class 1 (default)
        default_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Get risk category
        risk_category, risk_color = get_risk_category(default_probability)
        
        # Create feature importance chart
        feature_importance = create_feature_importance_chart(input_df)
        
        # Create risk comparison chart 
        risk_comparison, percentile = create_risk_comparison_chart(raw_data)
        
        # Get loan recommendations
        recommendations = recommend_loan_terms(raw_data, default_probability)
        
        return render_template('result.html',
                              prediction=prediction,
                              probability=default_probability * 100,  # Convert to percentage
                              form_data=raw_data,
                              risk_category=risk_category,
                              risk_color=risk_color,
                              feature_importance=feature_importance,
                              risk_comparison=risk_comparison,
                              recommendations=recommendations,
                              model_loaded=model_loaded)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', 
                              error=f"Error making prediction: {str(e)}",
                              model_loaded=model_loaded,
                              form_data=request.form)

# What-if analysis API
@app.route('/api/what_if', methods=['POST'])
def what_if_analysis():
    # Get data from request
    data = request.json
    
    # In a real application, this would use the risk model to calculate both original 
    # and adjusted risk scores based on the parameters
    
    # For demo purposes, generate sample response
    original = {
        'grade': 'C',
        'riskLevel': 'Moderate Credit Risk',
        'riskPercent': 55,
        'circleColor': '#cddc39',
        'score': 75
    }
    
    adjusted = {
        'grade': 'B',
        'riskLevel': 'Low Credit Risk',
        'riskPercent': 35,
        'circleColor': '#8bc34a',
        'score': 85
    }
    
    # Generate impact factors
    impacts = []
    
    # Compare parameters and generate impact statements
    fico_score = int(data['ficoScore'])
    if fico_score > 700:
        impacts.append({
            'effect': 'positive',
            'text': f'Higher FICO score reduced risk by {original["riskPercent"] - adjusted["riskPercent"]}%'
        })
    
    loan_amount = int(data['loanAmount'])
    if loan_amount < 15000:
        impacts.append({
            'effect': 'positive',
            'text': f'Lower loan amount reduced risk by 10%'
        })
    
    debt_to_income = float(data['debtToIncome'])
    if debt_to_income > 30:
        impacts.append({
            'effect': 'negative',
            'text': f'Higher debt-to-income ratio increased risk by 15%'
        })
    else:
        impacts.append({
            'effect': 'positive',
            'text': f'Lower debt-to-income ratio reduced risk by 8%'
        })
    
    employment_length = int(data['employmentLength'])
    if employment_length < 3:
        impacts.append({
            'effect': 'negative',
            'text': f'Short employment history increased risk by 12%'
        })
    
    delinquencies = int(data['delinquencies'])
    if delinquencies > 0:
        impacts.append({
            'effect': 'negative',
            'text': f'{delinquencies} delinquencies increased risk by {delinquencies * 5}%'
        })
    else:
        impacts.append({
            'effect': 'positive',
            'text': 'No delinquencies reduced risk by 10%'
        })
    
    # Generate recommendations
    recommendations = [
        f'Reducing your debt-to-income ratio below 20% would significantly improve your credit risk grade.',
        f'Improving your FICO score by 50 points could raise your grade from {original["grade"]} to {adjusted["grade"]}.',
        f'Reducing the loan amount by $5,000 would lower your risk assessment.'
    ]
    
    # Generate sensitivity data for chart
    sensitivity = {
        'factors': ['FICO Score', 'Debt-to-Income', 'Loan Amount', 'Employment Length', 'Interest Rate', 'Delinquencies', 'Loan Term'],
        'impacts': [15, 12, 8, 6, 5, 4, 2]
    }
    
    return jsonify({
        'original': original,
        'adjusted': adjusted,
        'impacts': impacts,
        'recommendations': recommendations,
        'sensitivity': sensitivity
    })

# API endpoint for credit risk calculation
@app.route('/api/calculate-risk', methods=['POST'])
def calculate_risk():
    data = request.json
    
    # Extract the form data
    loan_amount = float(data.get('loanAmount', 0))
    loan_term = int(data.get('loanTerm', 0))
    interest_rate = float(data.get('interestRate', 0))
    annual_income = float(data.get('annualIncome', 0))
    employment_length = int(data.get('employmentLength', 0))
    home_ownership = data.get('homeOwnership', '')
    loan_purpose = data.get('loanPurpose', '')
    debt_to_income = float(data.get('debtToIncome', 0))
    fico_low = int(data.get('ficoLow', 0))
    fico_high = int(data.get('ficoHigh', 0))
    delinquencies = int(data.get('delinquencies', 0))
    model_type = data.get('modelType', 'default')
    
    # Calculate credit risk (simplified scoring algorithm)
    score = 0
    
    # FICO score impact (0-40 points)
    fico_avg = (fico_low + fico_high) / 2
    if fico_avg >= 750:
        score += 40
    elif fico_avg >= 700:
        score += 30
    elif fico_avg >= 650:
        score += 20
    elif fico_avg >= 600:
        score += 10
    else:
        score += 5
    
    # Debt-to-income impact (0-25 points)
    if debt_to_income <= 20:
        score += 25
    elif debt_to_income <= 30:
        score += 20
    elif debt_to_income <= 40:
        score += 15
    elif debt_to_income <= 50:
        score += 5
    
    # Employment length impact (0-15 points)
    if employment_length >= 5:
        score += 15
    elif employment_length >= 3:
        score += 10
    else:
        score += 5
    
    # Delinquencies impact (0-20 points)
    if delinquencies == 0:
        score += 20
    elif delinquencies == 1:
        score += 10
    elif delinquencies <= 3:
        score += 5
    
    # Calculate risk grade
    max_score = 100
    normalized_score = min(max(score, 0), max_score)
    
    if normalized_score >= 90:
        grade = 'A'
        risk_level = 'Very Low Credit Risk'
        risk_percent = 10
        circle_color = '#4caf50'  # Green
    elif normalized_score >= 80:
        grade = 'B'
        risk_level = 'Low Credit Risk'
        risk_percent = 20
        circle_color = '#8bc34a'  # Light Green
    elif normalized_score >= 70:
        grade = 'C'
        risk_level = 'Moderate Credit Risk'
        risk_percent = 30
        circle_color = '#cddc39'  # Lime
    elif normalized_score >= 60:
        grade = 'D'
        risk_level = 'Average Credit Risk'
        risk_percent = 50
        circle_color = '#ffeb3b'  # Yellow
    elif normalized_score >= 50:
        grade = 'E'
        risk_level = 'Elevated Credit Risk'
        risk_percent = 70
        circle_color = '#ffc107'  # Amber
    elif normalized_score >= 40:
        grade = 'F'
        risk_level = 'High Credit Risk'
        risk_percent = 85
        circle_color = '#ff9800'  # Orange
    else:
        grade = 'G'
        risk_level = 'Very High Credit Risk'
        risk_percent = 95
        circle_color = '#f44336'  # Red
    
    # Generate risk factors
    factors = []
    
    # FICO score factors
    if fico_avg >= 700:
        factors.append({
            'positive': True,
            'text': 'Strong credit score'
        })
    elif fico_avg < 600:
        factors.append({
            'positive': False,
            'text': 'Low credit score'
        })
    
    # Debt-to-income factors
    if debt_to_income > 40:
        factors.append({
            'positive': False,
            'text': 'High debt-to-income ratio'
        })
    elif debt_to_income <= 20:
        factors.append({
            'positive': True,
            'text': 'Low debt-to-income ratio'
        })
    
    # Employment length factors
    if employment_length < 3:
        factors.append({
            'positive': False,
            'text': 'Short employment history'
        })
    elif employment_length >= 5:
        factors.append({
            'positive': True,
            'text': 'Stable employment history'
        })
    
    # Delinquencies factors
    if delinquencies == 0:
        factors.append({
            'positive': True,
            'text': 'No delinquencies in past 2 years'
        })
    elif delinquencies > 1:
        factors.append({
            'positive': False,
            'text': f'Multiple delinquencies ({delinquencies}) in past 2 years'
        })
    
    # Loan amount to income ratio
    loan_to_income_ratio = loan_amount / annual_income
    if loan_to_income_ratio > 0.5:
        factors.append({
            'positive': False,
            'text': 'Loan amount is high relative to annual income'
        })
    elif loan_to_income_ratio <= 0.2:
        factors.append({
            'positive': True,
            'text': 'Loan amount is appropriate for income level'
        })
    
    # Return the risk assessment results
    result = {
        'grade': grade,
        'riskLevel': risk_level,
        'riskPercent': risk_percent,
        'circleColor': circle_color,
        'score': normalized_score,
        'factors': factors,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Log the result to a file for analysis
    log_assessment(data, result)
    
    return jsonify(result)

# Function to log assessment data
def log_assessment(input_data, result):
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'assessments.jsonl')
    
    log_entry = {
        'input': input_data,
        'result': result,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# Model Performance page
@app.route('/model-performance')
def model_performance():
    # Sample model metrics - in a real app, this would come from a database or model file
    metrics = {
        'default_model': {
            'name': 'Random Forest Model',
            'accuracy': 94,
            'auc': 0.89,
            'f1_score': 0.93,
            'training_size': 10000,
            'last_updated': '2025-02-23'
        },
        'alternative_model': {
            'name': 'XGBoost Model',
            'accuracy': 92,
            'auc': 0.91,
            'f1_score': 0.92,
            'training_size': 10000,
            'last_updated': '2025-02-23'
        }
    }
    
    return render_template('model_performance.html', metrics=metrics)

@app.route('/what-if')
def what_if():
    return render_template('what_if.html')

if __name__ == '__main__':
    app.run(debug=True) 