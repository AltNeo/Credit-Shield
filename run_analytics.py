import os
import sys
import pandas as pd
import pickle
from model_analytics import run_analytics

def main():
    """
    Run analytics on credit risk model and dataset
    """
    print("=== Credit Risk Model Analytics ===")
    
    # Check if model exists
    model_path = 'models/best_model.pkl'
    artifacts_path = 'models/preprocessing_artifacts.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        print("Model or preprocessing artifacts not found. Please train the model first.")
        print("Run 'python latestmodel.py' to train the model.")
        return
    
    # Get original data path
    data_path = 'medium_data.csv'
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        alt_data_path = input("Please enter the path to your data file: ")
        if os.path.exists(alt_data_path):
            data_path = alt_data_path
        else:
            print(f"Data file {alt_data_path} not found. Exiting.")
            return
    
    # Run analytics
    print(f"Running analytics on model and dataset...")
    output_files = run_analytics(data_path, model_path, artifacts_path)
    
    print("\nAnalytics completed!")
    print(f"Dashboard: {output_files['dashboard']}")
    print(f"Report: {output_files['report']}")
    print(f"Model Metrics: {output_files['metrics']}")
    
    # Try to open the report in the default browser
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_files['report'])}")
        print("Report opened in your default web browser.")
    except:
        print("Could not open report automatically.")

if __name__ == "__main__":
    main() 