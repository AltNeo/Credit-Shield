import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib
import os
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42

def load_and_explore_data(file_path):
    """
    Load data and perform initial exploration
    """
    print("Loading and exploring data...")
    df = pd.read_csv(file_path)
    
    # Basic information
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nSummary statistics:")
    print(df.describe())
    
    # Identify target variable
    print("\nUnique values in potential target variables:")
    for col in df.columns:
        if 'risk' in col.lower() or 'grade' in col.lower() or 'rating' in col.lower():
            print(f"{col}: {df[col].unique()}")
    
    return df

def preprocess_data(df, target_column):
    """
    Preprocess data by handling missing values, encoding categorical features,
    and splitting into training and testing sets
    """
    print("\nPreprocessing data...")
    

    data = df.copy()
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Handle missing values
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Fill missing numerical values with median
    for col in numerical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Fill missing categorical values with mode
    for col in categorical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])
    
    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    preprocessing_artifacts = {
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'categorical_cols': categorical_cols.tolist(),
        'numerical_cols': numerical_cols.tolist(),
    }
    
    return X_train, X_test, y_train, y_test, preprocessing_artifacts

def train_model(X_train, y_train):
    """
    Train Random Forest model
    """
    print("\nTraining model...")
    
    # Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    # Parameter grid for training with larger dataset
    rf_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_grid = GridSearchCV(
        rf_model, rf_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
    )
    
    print(f"Starting grid search at {datetime.now().strftime('%H:%M:%S')}...")
    rf_grid.fit(X_train, y_train)
    print(f"Completed grid search at {datetime.now().strftime('%H:%M:%S')}")
    
    print(f"Best Random Forest parameters: {rf_grid.best_params_}")
    best_model = rf_grid.best_estimator_
    
    return best_model

def evaluate_model(model, X_test, y_test, model_name="RandomForest"):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'models/confusion_matrix_{model_name}.png')
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Try calculating ROC AUC (works for binary classification)
    try:
        if len(set(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            print(f"ROC AUC: {roc_auc:.4f}")
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.savefig(f'models/roc_curve_{model_name}.png')
    except:
        print("ROC AUC calculation skipped (not applicable for multiclass)")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def export_model(model_results, artifacts, target_column, model_name="RandomForest"):
    """
    Export the model
    """
    print("\nExporting model...")
    
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Remove any existing models
    for file in os.listdir('models'):
        file_path = os.path.join('models', file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Removed old model file: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    # Save model
    model = model_results['model']
    
    # Ensure model has predict and predict_proba methods before saving
    if not hasattr(model, 'predict'):
        raise ValueError(f"Model does not have predict method: {type(model)}")
    if not hasattr(model, 'predict_proba'):
        print(f"WARNING: Model does not have predict_proba method: {type(model)}")
    
    # Save RandomForest model using pickle for consistency with app.py's load mechanism
    model_path = f'models/randomforest_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle  # Use pickle here for consistency
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Create a copy as best_model for compatibility
    best_model_path = f'models/best_model.pkl'
    with open(best_model_path, 'wb') as f:
        import pickle
        pickle.dump(model, f)
    print(f"Model saved to {best_model_path}")
    
    # Save preprocessing artifacts using pickle for consistency
    artifacts_path = 'models/preprocessing_artifacts.pkl'
    artifacts['target_column'] = target_column
    with open(artifacts_path, 'wb') as f:
        import pickle
        pickle.dump(artifacts, f)
    print(f"Preprocessing artifacts saved to {artifacts_path}")
    
    # Test load to verify the model can be loaded
    print("Verifying model load works...")
    try:
        with open(model_path, 'rb') as f:
            import pickle
            test_model = pickle.load(f)
        # Verify it has required methods
        if hasattr(test_model, 'predict'):
            print("✅ Verified model can be loaded and has predict method")
        else:
            print("❌ Loaded model does not have predict method!")
    except Exception as e:
        print(f"❌ Error verifying model load: {e}")
    
    # Create model metrics JSON file
    metrics = {
        'models': {
            'RandomForest': {
                'accuracy': float(model_results['accuracy']),
                'precision': float(model_results['precision']),
                'recall': float(model_results['recall']),
                'f1': float(model_results['f1'])
            }
        },
        'best_model': model_name,
        'input_features': artifacts['feature_names'],
        'target_column': target_column,
        'categorical_features': artifacts['categorical_cols'],
        'numerical_features': artifacts['numerical_cols'],
        'dataset_size': 10000,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metrics_path = 'models/model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Model metrics saved to {metrics_path}")
    
    # Create a feature importance plot
    try:
        if hasattr(model, 'feature_importances_'):
            feature_names = artifacts['feature_names']
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Show only top 30 features for readability
            top_n = min(30, len(indices))
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.bar(range(top_n), importances[indices[:top_n]], align='center')
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
            plt.tight_layout()
            plt.savefig(f'models/feature_importance.png')
            print(f"Feature importance plot saved to models/feature_importance.png")
            
            # Also save feature importances as CSV
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            importance_df.to_csv('models/feature_importance.csv', index=False)
            print(f"Feature importance data saved to models/feature_importance.csv")
    except Exception as e:
        print(f"Could not generate feature importance plot: {str(e)}")
    
    return model

def main():
    """
    Main function to orchestrate the credit risk model development
    """
    # Load and explore data
    file_path = 'medium_data.csv'
    df = load_and_explore_data(file_path)
    
    # Identify target column - adjust based on actual data exploration
    target_columns = [col for col in df.columns if 'risk' in col.lower() or 
                     'grade' in col.lower() or 'rating' in col.lower()]
    
    if not target_columns:
        print("Could not automatically identify target column. Using 'grade'")
        target_column = 'grade'  # Default target column
    else:
        target_column = target_columns[0]
        print(f"Identified target column: {target_column}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessing_artifacts = preprocess_data(df, target_column)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    model_results = evaluate_model(model, X_test, y_test)
    
    # Export model
    export_model(model_results, preprocessing_artifacts, target_column)
    
    print("\nCredit risk model development completed successfully!")
    print(f"Random Forest model F1 score: {model_results['f1']:.4f}")

if __name__ == "__main__":
    main()
