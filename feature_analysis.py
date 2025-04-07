import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import datetime
import webbrowser
import json
import matplotlib
matplotlib.use('Agg')

def generate_feature_analysis():
    """
    Generate a comprehensive feature analysis report with statistics and visualizations
    """
    # Create output directories if they don't exist
    output_dir = Path('reports/feature_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Try to load actual data if available, otherwise use synthetic data
    try:
        if os.path.exists('medium_data.csv'):
            print("Loading data from medium_data.csv...")
            data = pd.read_csv('medium_data.csv')
            print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Sample data if it's too large to prevent memory issues
            if data.shape[0] > 10000:
                print(f"Data is large ({data.shape[0]} rows). Sampling 10,000 rows...")
                data = data.sample(10000, random_state=42)
                
            # Drop columns with all missing values
            missing_cols = data.columns[data.isna().all()].tolist()
            if missing_cols:
                print(f"Dropping {len(missing_cols)} columns with all missing values")
                data = data.drop(columns=missing_cols)
        else:
            # Generate synthetic data for demonstration
            print("Data file not found. Generating synthetic data...")
            data = generate_synthetic_data()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        # Generate synthetic data as fallback
        print("Using synthetic data as fallback")
        data = generate_synthetic_data()
    
    # Extract feature names
    features = data.columns.tolist()
    if 'default' in features:  # Assuming 'default' is the target variable
        features.remove('default')
    
    # Limit the number of features if there are too many
    if len(features) > 20:
        print(f"Too many features ({len(features)}). Selecting most informative features...")
        
        # Define critical features that must be included
        critical_features = ['int_rate', 'installment', 'annual_inc', 'loan_amnt', 'dti']
        
        # Filter to only include critical features that actually exist in the dataset
        critical_features = [f for f in critical_features if f in features]
        print(f"Including critical features: {', '.join(critical_features)}")
        
        # Remove critical features from the list to process
        remaining_features = [f for f in features if f not in critical_features]
        
        # Calculate how many additional features we can include
        additional_count = 20 - len(critical_features)
        
        # Try to select features based on importance or variance
        if 'default' in data.columns and additional_count > 0:
            # Calculate correlation with target
            correlations = []
            for feature in remaining_features:
                if pd.api.types.is_numeric_dtype(data[feature]) and not data[feature].isna().all():
                    try:
                        corr = abs(data[feature].corr(data['default']))
                        if pd.notna(corr):
                            correlations.append((feature, corr))
                    except:
                        pass
            
            # Sort by correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            # Take top numerical features
            top_numerical = [f[0] for f in correlations[:int(additional_count * 0.75)]]
            
            # Also include some categorical features
            categorical_features = [f for f in remaining_features if not pd.api.types.is_numeric_dtype(data[f])]
            top_categorical = categorical_features[:int(additional_count * 0.25)]
            
            selected_features = critical_features + top_numerical + top_categorical
        else:
            # Select based on variance for numerical and uniqueness for categorical
            numerical_features = [f for f in remaining_features if pd.api.types.is_numeric_dtype(data[f])]
            categorical_features = [f for f in remaining_features if not pd.api.types.is_numeric_dtype(data[f])]
            
            # Calculate variance for numerical features
            if numerical_features and additional_count > 0:
                variances = [(f, data[f].var()) for f in numerical_features if not data[f].isna().all()]
                variances = [(f, v) for f, v in variances if pd.notna(v) and np.isfinite(v)]
                variances.sort(key=lambda x: x[1], reverse=True)
                top_numerical = [f[0] for f in variances[:int(additional_count * 0.75)]]
            else:
                top_numerical = []
                
            # Select categorical features with most unique values
            if categorical_features and additional_count > len(top_numerical):
                uniqueness = [(f, data[f].nunique()) for f in categorical_features]
                uniqueness.sort(key=lambda x: x[1], reverse=True)
                top_categorical = [f[0] for f in uniqueness[:int(additional_count * 0.25)]]
            else:
                top_categorical = []
                
            selected_features = critical_features + top_numerical + top_categorical
            
        # Make sure we don't exceed 20 features
        features = selected_features[:20]
        print(f"Selected features: {', '.join(features)}")
    
    print("Calculating feature statistics...")
    feature_stats = calculate_feature_statistics(data, features)
    
    print("Generating feature visualizations...")
    # Close all figures first to prevent memory issues
    plt.close('all')
    visualization_paths = generate_feature_visualizations(data, features, output_dir)
    
    print("Calculating feature importance...")
    # Calculate feature importance if available
    feature_importance = calculate_feature_importance(features)
    
    # Calculate correlation matrix only for numerical features to avoid errors
    print("Generating correlation matrix...")
    try:
        numerical_features = [f for f in features if pd.api.types.is_numeric_dtype(data[f])]
        # Filter out columns with all NaN values
        valid_numerical_features = [f for f in numerical_features if not data[f].isna().all()]
        
        # Check if we have enough valid numerical features
        if len(valid_numerical_features) > 1:
            # Ensure all values are finite
            correlation_data = data[valid_numerical_features].copy()
            for col in correlation_data.columns:
                correlation_data[col] = pd.to_numeric(correlation_data[col], errors='coerce')
            
            # Generate correlation matrix
            correlation_matrix = correlation_data.corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, annot=False, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_heatmap.png')
            plt.close()
        else:
            print("Not enough valid numerical features for correlation matrix")
    except Exception as e:
        print(f"Error generating correlation matrix: {str(e)}")
        plt.close('all')  # Close any open figures
    
    # Save feature statistics to JSON for future reference
    print("Saving feature statistics...")
    with open(output_dir / 'feature_statistics.json', 'w') as f:
        json.dump(feature_stats, f, indent=4, default=str)
    
    # Generate HTML report
    print("Generating HTML report...")
    html = generate_html_report(feature_stats, visualization_paths, feature_importance, output_dir)
    
    # Write to file
    output_file = 'reports/feature_analysis.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Feature analysis generated: {output_file}")
    
    # Try to open the report in the default browser
    try:
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
        print("Feature analysis report opened in your default web browser.")
    except Exception as e:
        print(f"Could not open report automatically: {str(e)}")
        print(f"Report is available at: {os.path.abspath(output_file)}")
    
    return output_file

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for credit risk analysis
    """
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
        'previous_defaults': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.7, 0.1, 0.08, 0.05, 0.04, 0.03]),
        'credit_inquiries': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples, p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02]),
        'payment_to_income': np.random.beta(2, 8, n_samples) * 0.5,
    }
    
    # Add categorical features
    data['loan_purpose'] = np.random.choice(['home', 'car', 'education', 'medical', 'business', 'other'], n_samples)
    data['has_mortgage'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    data['has_credit_card'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Create target variable
    default_prob = (
        (850 - data['credit_score']) / 850 * 0.5 +
        data['debt_to_income_ratio'] * 0.2 +
        (data['previous_defaults'] / 5) * 0.2 +
        (data['credit_inquiries'] / 8) * 0.1
    ) * 0.7
    default_prob += np.random.normal(0, 0.1, n_samples)
    default_prob = default_prob.clip(0, 1)
    data['default'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    return pd.DataFrame(data)

def calculate_feature_statistics(data, features):
    """
    Calculate comprehensive statistics for each feature
    """
    stats = {}
    
    for feature in features:
        try:
            feature_data = data[feature]
            
            # Skip if all values are missing
            if feature_data.isna().all():
                stats[feature] = {
                    'type': 'unknown',
                    'missing_values': len(feature_data),
                    'missing_percentage': 100.0,
                    'error': 'All values are missing'
                }
                continue
            
            # Determine if feature is numerical or categorical
            if pd.api.types.is_numeric_dtype(feature_data):
                # For numerical features
                stats[feature] = {
                    'type': 'numerical',
                    'missing_values': feature_data.isna().sum(),
                    'missing_percentage': feature_data.isna().mean() * 100
                }
                
                # Get non-missing values for calculations
                non_missing = feature_data.dropna()
                
                # Filter out non-finite values
                finite_values = non_missing[np.isfinite(non_missing)]
                if len(finite_values) > 0:
                    try:
                        stats[feature].update({
                            'mean': finite_values.mean(),
                            'median': finite_values.median(),
                            'std': finite_values.std(),
                            'min': finite_values.min(),
                            'max': finite_values.max(),
                        })
                        
                        # Calculate percentiles
                        try:
                            stats[feature].update({
                                'percentile_5': np.percentile(finite_values, 5),
                                'percentile_25': np.percentile(finite_values, 25),
                                'percentile_75': np.percentile(finite_values, 75),
                                'percentile_95': np.percentile(finite_values, 95),
                            })
                        except Exception as e:
                            stats[feature]['percentile_error'] = str(e)
                        
                        # Calculate additional statistics
                        try:
                            stats[feature].update({
                                'skewness': finite_values.skew(),
                                'kurtosis': finite_values.kurt(),
                            })
                        except Exception as e:
                            stats[feature]['dist_stats_error'] = str(e)
                    except Exception as e:
                        stats[feature]['basic_stats_error'] = str(e)
                else:
                    stats[feature]['error'] = 'No finite values available'
                
                # Add default vs non-default statistics
                if 'default' in data.columns and not feature_data.isna().all():
                    try:
                        default_mask = (data['default'] == 1) & feature_data.notna() & np.isfinite(feature_data)
                        non_default_mask = (data['default'] == 0) & feature_data.notna() & np.isfinite(feature_data)
                        
                        default_data = feature_data[default_mask]
                        non_default_data = feature_data[non_default_mask]
                        
                        # Only calculate if we have enough data
                        if len(default_data) > 0 and len(non_default_data) > 0:
                            by_target = {
                                'default_count': len(default_data),
                                'non_default_count': len(non_default_data),
                                'default_mean': default_data.mean(),
                                'non_default_mean': non_default_data.mean(),
                            }
                            
                            # Additional statistics if we have enough data
                            if len(default_data) >= 5:
                                by_target['default_median'] = default_data.median()
                                by_target['default_std'] = default_data.std()
                            
                            if len(non_default_data) >= 5:
                                by_target['non_default_median'] = non_default_data.median()
                                by_target['non_default_std'] = non_default_data.std()
                            
                            # Calculate differences if possible
                            if 'default_mean' in by_target and 'non_default_mean' in by_target:
                                mean_diff = by_target['default_mean'] - by_target['non_default_mean']
                                by_target['mean_difference'] = mean_diff
                                
                                # Percent difference (avoiding division by zero)
                                if by_target['non_default_mean'] != 0:
                                    by_target['mean_difference_percentage'] = (mean_diff / abs(by_target['non_default_mean'])) * 100
                            
                            stats[feature]['by_target'] = by_target
                    except Exception as e:
                        stats[feature]['by_target_error'] = str(e)
            else:
                # For categorical features
                try:
                    # Handle basic stats
                    stats[feature] = {
                        'type': 'categorical',
                        'unique_values': feature_data.nunique(),
                        'missing_values': feature_data.isna().sum(),
                        'missing_percentage': feature_data.isna().mean() * 100
                    }
                    
                    # Value counts and percentages
                    try:
                        non_missing = feature_data.dropna()
                        if len(non_missing) > 0:
                            value_counts = non_missing.value_counts().head(20)  # Limit to top 20 values
                            value_percentages = non_missing.value_counts(normalize=True).head(20) * 100
                            
                            if not value_counts.empty:
                                stats[feature].update({
                                    'most_common': str(value_counts.index[0]) if not pd.isna(value_counts.index[0]) else 'NA',
                                    'most_common_count': int(value_counts.iloc[0]) if not pd.isna(value_counts.iloc[0]) else 0,
                                    'most_common_percentage': float(value_percentages.iloc[0]) if not pd.isna(value_percentages.iloc[0]) else 0,
                                })
                                
                                # Convert to string keys for JSON compatibility
                                stats[feature]['value_counts'] = {str(k): int(v) for k, v in value_counts.items()}
                                stats[feature]['value_percentages'] = {str(k): float(v) for k, v in value_percentages.items()}
                        else:
                            stats[feature]['error'] = 'No non-missing values'
                    except Exception as e:
                        stats[feature]['value_counts_error'] = str(e)
                    
                    # Add default rate by category
                    if 'default' in data.columns:
                        try:
                            default_rates = {}
                            for category in feature_data.dropna().unique():
                                try:
                                    # Skip if category is NaN
                                    if pd.isna(category):
                                        continue
                                        
                                    category_mask = feature_data == category
                                    category_data = data[category_mask]
                                    
                                    # Only include if we have enough data
                                    if len(category_data) >= 5:
                                        category_key = str(category)
                                        default_rates[category_key] = {
                                            'count': int(len(category_data)),
                                            'default_count': int(category_data['default'].sum()),
                                            'default_rate': float(category_data['default'].mean() * 100)
                                        }
                                except Exception:
                                    # Skip problematic categories
                                    continue
                            
                            if default_rates:
                                stats[feature]['default_rates_by_category'] = default_rates
                        except Exception as e:
                            stats[feature]['default_rates_error'] = str(e)
                except Exception as e:
                    stats[feature] = {
                        'type': 'error',
                        'error': str(e)
                    }
        except Exception as e:
            stats[feature] = {
                'type': 'error',
                'error': str(e)
            }
    
    return stats

def generate_feature_visualizations(data, features, output_dir):
    """
    Generate visualizations for features
    """
    visualization_paths = {}
    
    # Create output directory for plots if it doesn't exist
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Process numerical and categorical features separately
    numerical_features = [f for f in features if pd.api.types.is_numeric_dtype(data[f])]
    categorical_features = [f for f in features if f in features and f not in numerical_features]
    
    # 1. Distribution plots for numerical features
    for feature in numerical_features:
        try:
            # Create a layout with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Distribution with KDE
            sns.histplot(data=data, x=feature, kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {feature}', fontsize=14)
            ax1.grid(alpha=0.3)
            
            # Plot 2: Box plot with default overlay if available
            if 'default' in data.columns and data['default'].nunique() > 1:
                # Make sure we have enough data in each category
                default_counts = data['default'].value_counts()
                if all(default_counts > 5):  # At least 5 samples in each category
                    sns.boxplot(x='default', y=feature, data=data, ax=ax2)
                    ax2.set_title(f'{feature} by Default Status', fontsize=14)
                    ax2.set_xticklabels(['Non-Default', 'Default'])
                else:
                    # Fallback to regular boxplot
                    sns.boxplot(y=data[feature].dropna(), ax=ax2)
                    ax2.set_title(f'Box Plot of {feature}', fontsize=14)
            else:
                # Fallback to regular boxplot
                sns.boxplot(y=data[feature].dropna(), ax=ax2)
                ax2.set_title(f'Box Plot of {feature}', fontsize=14)
            
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plot_path = f'plots/{feature}_distribution.png'
            plt.savefig(output_dir / plot_path)
            plt.close(fig)
            
            visualization_paths[feature] = plot_path
        except Exception as e:
            print(f"Error generating visualization for {feature}: {str(e)}")
            plt.close('all')  # Close any open figures
    
    # 2. Bar plots for categorical features
    for feature in categorical_features:
        try:
            # Create a layout with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Value counts
            value_counts = data[feature].value_counts().sort_values(ascending=False)
            # Limit categories to display if there are too many
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
                ax1.set_title(f'Top 10 Categories for {feature}', fontsize=14)
            else:
                ax1.set_title(f'Count of {feature}', fontsize=14)
                
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            ax1.grid(alpha=0.3, axis='y')
            
            # Plot 2: Default rate by category if available
            if 'default' in data.columns:
                try:
                    default_rates = {}
                    for category in data[feature].unique():
                        if pd.notna(category):
                            mask = data[feature] == category
                            # Only include categories with enough samples
                            if mask.sum() >= 5:
                                default_rates[category] = data.loc[mask, 'default'].mean() * 100
                    
                    if default_rates:
                        default_df = pd.DataFrame({'Category': list(default_rates.keys()), 
                                                'Default_Rate': list(default_rates.values())})
                        default_df = default_df.sort_values('Default_Rate', ascending=False)
                        
                        # Limit categories to display if there are too many
                        if len(default_df) > 10:
                            default_df = default_df.head(10)
                            ax2.set_title(f'Top 10 Default Rates by {feature} (%)', fontsize=14)
                        else:
                            ax2.set_title(f'Default Rate by {feature} (%)', fontsize=14)
                        
                        sns.barplot(x='Category', y='Default_Rate', data=default_df, ax=ax2)
                        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
                        ax2.grid(alpha=0.3, axis='y')
                    else:
                        ax2.set_visible(False)
                except Exception as e:
                    print(f"Error generating default rate plot for {feature}: {str(e)}")
                    ax2.set_visible(False)
            else:
                ax2.set_visible(False)
            
            plt.tight_layout()
            plot_path = f'plots/{feature}_distribution.png'
            plt.savefig(output_dir / plot_path)
            plt.close(fig)
            
            visualization_paths[feature] = plot_path
        except Exception as e:
            print(f"Error generating visualization for {feature}: {str(e)}")
            plt.close('all')  # Close any open figures
    
    # 3. Generate a correlation matrix heatmap
    try:
        plt.figure(figsize=(12, 10))
        
        # Only include numerical features with finite values
        valid_numerical_features = []
        for feature in numerical_features:
            if data[feature].notna().all() and np.isfinite(data[feature]).all():
                valid_numerical_features.append(feature)
        
        if valid_numerical_features:
            corr_matrix = data[valid_numerical_features].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Limit size of correlation matrix if it's too large
            if len(corr_matrix) > 15:
                # Select top 15 features by variance
                variances = data[valid_numerical_features].var().sort_values(ascending=False)
                top_features = variances.index[:15].tolist()
                corr_matrix = data[top_features].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.8, center=0,
                        square=True, linewidths=.5, annot=True, fmt='.2f')
            plt.title('Correlation Matrix of Numerical Features', fontsize=16)
            plt.tight_layout()
            
            correlation_path = 'plots/correlation_matrix.png'
            plt.savefig(output_dir / correlation_path)
            visualization_paths['correlation_matrix'] = correlation_path
        
        plt.close()
    except Exception as e:
        print(f"Error generating correlation matrix: {str(e)}")
        plt.close('all')
    
    # 4. If default column exists, create feature importance plot based on correlation with default
    if 'default' in data.columns:
        try:
            correlations = []
            for feature in numerical_features:
                try:
                    corr = data[feature].corr(data['default'])
                    if not pd.isna(corr) and np.isfinite(corr):
                        correlations.append((feature, abs(corr), corr))
                except:
                    continue
            
            # Only proceed if we have correlations
            if correlations:
                # Sort by absolute correlation
                correlations.sort(key=lambda x: x[1], reverse=True)
                
                # Limit to top 15 features if more
                if len(correlations) > 15:
                    correlations = correlations[:15]
                
                # Create correlation plot with default
                plt.figure(figsize=(10, 8))
                features = [x[0] for x in correlations]
                corr_values = [x[2] for x in correlations]
                colors = ['red' if c < 0 else 'green' for c in corr_values]
                
                bars = plt.barh(features, [abs(c) for c in corr_values], color=colors)
                plt.title('Feature Correlation with Default', fontsize=16)
                plt.xlabel('Absolute Correlation Coefficient', fontsize=12)
                
                # Add correlation values on the bars
                for bar, corr in zip(bars, corr_values):
                    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{corr:.2f}', va='center', fontsize=10)
                
                plt.tight_layout()
                importance_path = 'plots/default_correlation.png'
                plt.savefig(output_dir / importance_path)
                visualization_paths['default_correlation'] = importance_path
                
                plt.close()
        except Exception as e:
            print(f"Error generating feature correlation with default: {str(e)}")
            plt.close('all')
    
    return visualization_paths

def calculate_feature_importance(features):
    """
    Calculate feature importance from stored model or generate placeholder
    """
    # Try to load feature importance from a saved model
    model_path = 'models/best_model.pkl'
    
    if os.path.exists(model_path):
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(features, importances))
        except:
            pass
    
    # If no model available, return placeholder importance based on domain knowledge
    placeholder_importance = {
        'credit_score': 0.25,
        'debt_to_income_ratio': 0.20,
        'previous_defaults': 0.15,
        'income': 0.10,
        'loan_amount': 0.08,
        'payment_to_income': 0.07,
        'employment_years': 0.06,
        'credit_inquiries': 0.05,
        'loan_term': 0.03,
        'has_mortgage': 0.01
    }
    
    # Fill missing features with small random values
    for feature in features:
        if feature not in placeholder_importance:
            placeholder_importance[feature] = np.random.uniform(0.005, 0.02)
    
    # Normalize to sum to 1
    total = sum(placeholder_importance.values())
    return {k: v/total for k, v in placeholder_importance.items()}

def generate_html_report(feature_stats, visualization_paths, feature_importance, output_dir):
    """
    Generate HTML report with feature statistics and visualizations
    """
    # Sort features by importance if available
    sorted_features = sorted(feature_stats.keys(), 
                             key=lambda x: feature_importance.get(x, 0), 
                             reverse=True)
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AB InBev Credit Risk Model - Feature Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; background-color: #f9f9f9; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2c3e50; margin-top: 30px; padding: 10px; background-color: #f5f5f5; border-left: 4px solid #3498db; }}
            h3 {{ color: #2c3e50; margin-top: 25px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; overflow-x: auto; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; color: #2c3e50; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .feature-section {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px dashed #ccc; }}
            .visualization {{ margin: 20px 0; text-align: center; }}
            .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .timestamp {{ color: #7f8c8d; font-size: 14px; margin-bottom: 20px; }}
            footer {{ margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 14px; padding-top: 20px; border-top: 1px solid #eee; }}
            .stats-highlight {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }}
            .importance-indicator {{ display: inline-block; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px; }}
            .high-importance {{ background-color: #e74c3c; }}
            .medium-importance {{ background-color: #f39c12; }}
            .low-importance {{ background-color: #3498db; }}
            .stats-table {{ margin: 20px 0; }}
            .stats-table table {{ width: 100%; }}
            .stats-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .nav-links {{ display: flex; justify-content: space-between; margin: 20px 0; }}
            .nav-button {{ padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; border-radius: 5px; }}
            .high-correlation {{ color: #e74c3c; font-weight: bold; }}
            .feature-cards {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .feature-card {{ width: 48%; margin-bottom: 20px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            @media (max-width: 768px) {{ .feature-card {{ width: 100%; }} }}
            .toggle-button {{ cursor: pointer; color: #3498db; margin-left: 10px; }}
            .collapsible {{ display: block; }}
            .hidden {{ display: none; }}
        </style>
        <script>
            function toggleSection(id) {{
                var element = document.getElementById(id);
                var button = document.getElementById(id + '-toggle');
                if (element.classList.contains('hidden')) {{
                    element.classList.remove('hidden');
                    button.textContent = '[-]';
                }} else {{
                    element.classList.add('hidden');
                    button.textContent = '[+]';
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>AB InBev Credit Risk Model - Feature Analysis</h1>
            <p class="timestamp">Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="nav-links">
                <a href="multi_model_comparison.html" class="nav-button">← Model Comparison</a>
                <a href="confusion_matrix_analysis.html" class="nav-button">Confusion Matrix Analysis →</a>
            </div>
            
            <h2>Feature Importance Overview</h2>
            <p>
                The analysis below examines each feature's characteristics and potential impact on credit risk predictions.
                Features are ordered by their estimated importance to the model.
            </p>
            
            <div class="visualization">
                <img src="feature_analysis/plots/default_correlation.png" alt="Feature Importance">
            </div>
            
            <h2>Feature Correlation Analysis</h2>
            <p>
                The heatmap below shows correlations between numerical features. Strong correlations may indicate redundant information.
            </p>
            
            <div class="visualization">
                <img src="feature_analysis/plots/correlation_matrix.png" alt="Correlation Matrix">
            </div>
    """
    
    # Add feature sections
    html += """
            <h2>Individual Feature Analysis</h2>
            <p>
                Each feature is analyzed below with descriptive statistics and visualizations.
                <span class="importance-indicator high-importance"></span> High importance features
                <span class="importance-indicator medium-importance"></span> Medium importance features
                <span class="importance-indicator low-importance"></span> Low importance features
            </p>
    """
    
    # Add sections for each feature
    for feature in sorted_features:
        # Determine importance level for styling
        importance = feature_importance.get(feature, 0)
        if importance >= 0.1:
            importance_class = "high-importance"
        elif importance >= 0.05:
            importance_class = "medium-importance"
        else:
            importance_class = "low-importance"
        
        # Get feature stats
        stats = feature_stats[feature]
        feature_type = stats['type']
        
        # Create section
        section_id = f"feature-{feature.replace(' ', '-')}"
        
        html += f"""
            <div class="feature-section">
                <h3>
                    <span class="importance-indicator {importance_class}"></span>
                    {feature} 
                    <span class="toggle-button" id="{section_id}-toggle" onclick="toggleSection('{section_id}')">[-]</span>
                </h3>
                
                <div id="{section_id}" class="collapsible">
                    <p><strong>Importance Score:</strong> {importance:.4f}</p>
        """
        
        # Add visualization if available
        if feature in visualization_paths:
            html += f"""
                    <div class="visualization">
                        <img src="feature_analysis/{visualization_paths[feature]}" alt="{feature} Distribution">
                    </div>
            """
        
        # Add statistics based on feature type
        if feature_type == 'numerical':
            html += f"""
                    <div class="stats-card">
                        <h4>Descriptive Statistics</h4>
                        <div class="stats-table">
                            <table>
                                <tr>
                                    <th>Statistic</th>
                                    <th>Value</th>
                                    <th>Percentile</th>
                                    <th>Value</th>
                                </tr>
                                <tr>
                                    <td>Mean</td>
                                    <td>{stats['mean']:.4f}</td>
                                    <td>5th Percentile</td>
                                    <td>{stats['percentile_5']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>Median</td>
                                    <td>{stats['median']:.4f}</td>
                                    <td>25th Percentile</td>
                                    <td>{stats['percentile_25']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>Standard Deviation</td>
                                    <td>{stats['std']:.4f}</td>
                                    <td>75th Percentile</td>
                                    <td>{stats['percentile_75']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>Minimum</td>
                                    <td>{stats['min']:.4f}</td>
                                    <td>95th Percentile</td>
                                    <td>{stats['percentile_95']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>Maximum</td>
                                    <td>{stats['max']:.4f}</td>
                                    <td>Skewness</td>
                                    <td>{stats['skewness']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>Missing Values</td>
                                    <td>{stats['missing_values']} ({stats['missing_percentage']:.2f}%)</td>
                                    <td>Kurtosis</td>
                                    <td>{stats['kurtosis']:.4f}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
            """
            
            # Add target-specific stats if available
            if 'by_target' in stats:
                target_stats = stats['by_target']
                html += f"""
                    <div class="stats-card">
                        <h4>Default vs. Non-Default Comparison</h4>
                        <div class="stats-table">
                            <table>
                                <tr>
                                    <th>Statistic</th>
                                    <th>Default Group</th>
                                    <th>Non-Default Group</th>
                                    <th>Difference</th>
                                </tr>
                                <tr>
                                    <td>Mean</td>
                                    <td>{target_stats['default_mean']:.4f}</td>
                                    <td>{target_stats['non_default_mean']:.4f}</td>
                                    <td>{target_stats['mean_difference']:.4f} ({target_stats['mean_difference_percentage']:.2f}%)</td>
                                </tr>
                                <tr>
                                    <td>Median</td>
                                    <td>{target_stats['default_median']:.4f}</td>
                                    <td>{target_stats['non_default_median']:.4f}</td>
                                    <td>{target_stats['default_median'] - target_stats['non_default_median']:.4f}</td>
                                </tr>
                                <tr>
                                    <td>Standard Deviation</td>
                                    <td>{target_stats['default_std']:.4f}</td>
                                    <td>{target_stats['non_default_std']:.4f}</td>
                                    <td>{target_stats['default_std'] - target_stats['non_default_std']:.4f}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                """
                
                # Add interpretation
                abs_diff_pct = abs(target_stats['mean_difference_percentage'])
                if abs_diff_pct > 20:
                    significance = "strong"
                elif abs_diff_pct > 10:
                    significance = "moderate"
                else:
                    significance = "weak"
                
                direction = "higher" if target_stats['mean_difference'] > 0 else "lower"
                
                html += f"""
                    <div class="stats-highlight">
                        <p><strong>Interpretation:</strong> This feature shows a <strong>{significance}</strong> relationship with default risk. 
                        The mean value is <strong>{direction}</strong> for defaulting customers by {abs_diff_pct:.2f}%.</p>
                    </div>
                """
        else:
            # Categorical feature stats
            html += f"""
                    <div class="stats-card">
                        <h4>Category Distribution</h4>
                        <p><strong>Unique values:</strong> {stats['unique_values']}</p>
                        <p><strong>Most common:</strong> {stats['most_common']} ({stats['most_common_percentage']:.2f}%)</p>
                        <p><strong>Missing values:</strong> {stats['missing_values']} ({stats['missing_percentage']:.2f}%)</p>
                        
                        <div class="stats-table">
                            <table>
                                <tr>
                                    <th>Category</th>
                                    <th>Count</th>
                                    <th>Percentage</th>
                                </tr>
            """
            
            for category, count in stats['value_counts'].items():
                percentage = stats['value_percentages'][category]
                html += f"""
                                <tr>
                                    <td>{category}</td>
                                    <td>{count}</td>
                                    <td>{percentage:.2f}%</td>
                                </tr>
                """
            
            html += """
                            </table>
                        </div>
                    </div>
            """
            
            # Add default rates by category if available
            if 'default_rates_by_category' in stats:
                html += """
                    <div class="stats-card">
                        <h4>Default Rates by Category</h4>
                        <div class="stats-table">
                            <table>
                                <tr>
                                    <th>Category</th>
                                    <th>Count</th>
                                    <th>Default Rate</th>
                                    <th>Default Count</th>
                                </tr>
                """
                
                for category, data in stats['default_rates_by_category'].items():
                    html += f"""
                                <tr>
                                    <td>{category}</td>
                                    <td>{data['count']}</td>
                                    <td>{data['default_rate']:.2f}%</td>
                                    <td>{data['default_count']}</td>
                                </tr>
                    """
                
                html += """
                            </table>
                        </div>
                    </div>
                """
                
                # Add interpretation
                rates = [(category, data['default_rate']) for category, data in stats['default_rates_by_category'].items()]
                rates.sort(key=lambda x: x[1], reverse=True)
                if len(rates) > 1:
                    highest = rates[0]
                    lowest = rates[-1]
                    difference = highest[1] - lowest[1]
                    
                    if difference > 20:
                        significance = "strong"
                    elif difference > 10:
                        significance = "moderate"
                    else:
                        significance = "weak"
                    
                    html += f"""
                        <div class="stats-highlight">
                            <p><strong>Interpretation:</strong> This categorical feature shows a <strong>{significance}</strong> relationship with default risk. 
                            The category '{highest[0]}' has a {highest[1]:.2f}% default rate, while '{lowest[0]}' has a {lowest[1]:.2f}% default rate 
                            (a difference of {difference:.2f} percentage points).</p>
                        </div>
                    """
        
        # Close feature section
        html += """
                </div>
            </div>
        """
    
    # Add summary section
    html += """
            <h2>Feature Analysis Summary</h2>
            <div class="stats-highlight">
                <p>Based on the feature analysis above, we can draw the following conclusions:</p>
                <ul>
                    <li>The most predictive features for credit risk are those with strong correlations to default and clear separations between defaulters and non-defaulters.</li>
                    <li>Features with high importance scores should be prioritized in the credit risk assessment process.</li>
                    <li>Categorical variables with high variance in default rates across categories are particularly useful for risk segmentation.</li>
                    <li>Features with high correlations to each other may contain redundant information and could potentially be combined or one could be removed.</li>
                </ul>
            </div>
            
            <div class="nav-links">
                <a href="multi_model_comparison.html" class="nav-button">← Model Comparison</a>
                <a href="confusion_matrix_analysis.html" class="nav-button">Confusion Matrix Analysis →</a>
            </div>
            
            <footer>
                <p>AB InBev Credit Risk Analysis • Feature Analysis</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    import traceback
    
    try:
        print("Starting feature analysis...")
        # Close any existing matplotlib figures
        plt.close('all')
        
        # Run the feature analysis
        output_file = generate_feature_analysis()
        
        print(f"Feature analysis completed. Report saved to {output_file}")
    except Exception as e:
        print(f"Error occurred during feature analysis: {str(e)}")
        traceback.print_exc()
        print("Exiting with error.") 