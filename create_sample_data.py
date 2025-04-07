import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def create_sample_credit_data():
    """
    Create a sample credit risk dataset for testing
    """
    print("Generating sample credit risk dataset...")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Create a DataFrame with features
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Add some specific credit risk related features
    np.random.seed(42)
    
    # Numeric features
    df['age'] = np.random.randint(18, 85, size=5000)
    df['income'] = np.random.normal(50000, 20000, size=5000)
    df['debt_to_income'] = np.random.uniform(0, 1, size=5000)
    df['credit_score'] = np.random.randint(300, 850, size=5000)
    df['loan_amount'] = np.random.normal(15000, 7000, size=5000)
    df['loan_term_months'] = np.random.choice([12, 24, 36, 48, 60], size=5000)
    df['interest_rate'] = np.random.uniform(0.01, 0.15, size=5000)
    df['num_credit_lines'] = np.random.randint(0, 20, size=5000)
    df['utilization_rate'] = np.random.uniform(0, 1, size=5000)
    
    # Categorical features
    df['employment_status'] = np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], 
                                              p=[0.7, 0.1, 0.1, 0.1], size=5000)
    df['home_ownership'] = np.random.choice(['Own', 'Mortgage', 'Rent', 'Other'], 
                                           p=[0.3, 0.4, 0.25, 0.05], size=5000)
    df['purpose'] = np.random.choice(['Debt consolidation', 'Home improvement', 'Medical', 'Education', 'Other'], 
                                     p=[0.4, 0.2, 0.1, 0.1, 0.2], size=5000)
    
    # Add a grade (target) column
    df['grade'] = np.array(['A', 'B', 'C'])[y]
    
    # Add some missing values
    for col in df.columns:
        if np.random.random() < 0.3:  # 30% chance of column having missing values
            missing_mask = np.random.random(size=5000) < 0.05  # 5% missing values
            df.loc[missing_mask, col] = np.nan
    
    # Save dataset
    df.to_csv('medium_data.csv', index=False)
    print(f"Sample dataset created with {df.shape[0]} rows and {df.shape[1]} columns")
    print("Saved as 'medium_data.csv'")

if __name__ == "__main__":
    create_sample_credit_data() 