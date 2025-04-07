import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import json
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set standard style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Create output directories
os.makedirs('reports', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('reports/data', exist_ok=True)

class CreditRiskAnalytics:
    """
    Comprehensive analytics and visualization class for credit risk modeling
    """
    def __init__(self, original_data=None, model=None, X_train=None, X_test=None, 
                 y_train=None, y_test=None, preprocessing_artifacts=None):
        """
        Initialize with data and model
        """
        self.original_data = original_data
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessing_artifacts = preprocessing_artifacts
        
        # Results container
        self.results = {
            'dataset_stats': {},
            'model_performance': {},
            'feature_analysis': {},
            'visualization_paths': []
        }
        
    def load_from_files(self, data_path=None, model_path='models/best_model.pkl', 
                      artifacts_path='models/preprocessing_artifacts.pkl'):
        """
        Load model, data and artifacts from files
        """
        # Load data if path provided
        if data_path:
            self.original_data = pd.read_csv(data_path)
            print(f"Loaded data from {data_path}, shape: {self.original_data.shape}")
        
        # Load model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        
        # Load preprocessing artifacts
        try:
            with open(artifacts_path, 'rb') as f:
                self.preprocessing_artifacts = pickle.load(f)
            print(f"Loaded preprocessing artifacts from {artifacts_path}")
        except Exception as e:
            print(f"Error loading preprocessing artifacts: {str(e)}")
            
        return self
    
    def analyze_dataset(self):
        """
        Perform comprehensive analysis of the dataset
        """
        print("Analyzing dataset...")
        df = self.original_data
        
        # Basic dataset statistics
        stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numerical column statistics
        numerical_stats = {}
        for col in numerical_cols:
            col_stats = df[col].describe().to_dict()
            col_stats['skewness'] = df[col].skew()
            col_stats['kurtosis'] = df[col].kurtosis()
            numerical_stats[col] = col_stats
        
        # Categorical column statistics
        categorical_stats = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts().to_dict()
            unique_count = df[col].nunique()
            categorical_stats[col] = {
                'unique_values': unique_count,
                'top_values': dict(list(value_counts.items())[:10]),
                'distribution': {
                    'values': list(value_counts.keys())[:20],
                    'counts': list(value_counts.values())[:20]
                }
            }
        
        # Target variable analysis
        target_col = self.preprocessing_artifacts.get('target_column')
        if target_col:
            if target_col in categorical_cols:
                target_stats = {
                    'type': 'categorical',
                    'classes': df[target_col].unique().tolist(),
                    'distribution': df[target_col].value_counts().to_dict(),
                    'class_balance': (df[target_col].value_counts() / len(df)).to_dict()
                }
            else:
                target_stats = {
                    'type': 'numerical',
                    'statistics': df[target_col].describe().to_dict(),
                    'skewness': df[target_col].skew(),
                    'kurtosis': df[target_col].kurtosis()
                }
            stats['target_variable'] = target_stats
        
        # Correlation analysis
        numeric_df = df[numerical_cols]
        correlation_matrix = numeric_df.corr().round(2)
        
        # Save correlation matrix
        correlation_csv_path = 'reports/data/correlation_matrix.csv'
        correlation_matrix.to_csv(correlation_csv_path)
        stats['correlation_matrix_path'] = correlation_csv_path
        
        # Store all statistics
        self.results['dataset_stats'] = {
            'basic_stats': stats,
            'numerical_stats': numerical_stats,
            'categorical_stats': categorical_stats,
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols
        }
        
        # Save dataset stats to JSON
        dataset_stats_path = 'reports/data/dataset_statistics.json'
        with open(dataset_stats_path, 'w') as f:
            # Convert any numpy types to Python native types before serializing
            json_stats = self._convert_to_serializable(self.results['dataset_stats'])
            json.dump(json_stats, f, indent=4)
        
        print(f"Dataset statistics saved to {dataset_stats_path}")
        self.results['visualization_paths'].append(dataset_stats_path)
        
        return self
    
    def _convert_to_serializable(self, obj):
        """Helper method to convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        elif isinstance(obj, pd.DataFrame):
            return self._convert_to_serializable(obj.to_dict('records'))
        elif isinstance(obj, pd.Series):
            return self._convert_to_serializable(obj.to_dict())
        else:
            return obj
    
    def visualize_dataset(self):
        """
        Create comprehensive visualizations of the dataset
        """
        print("Creating dataset visualizations...")
        df = self.original_data
        
        # Get column lists
        numerical_cols = self.results['dataset_stats']['numerical_columns']
        categorical_cols = self.results['dataset_stats']['categorical_columns']
        target_col = self.preprocessing_artifacts.get('target_column')
        
        # 1. Missing values visualization
        plt.figure(figsize=(12, 8))
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            missing_percent = (missing_data / len(df) * 100).round(2)
            missing_df = pd.DataFrame({'Count': missing_data, 'Percent': missing_percent})
            
            ax = missing_df.plot(kind='bar', figsize=(12, 8), secondary_y='Percent')
            plt.title('Missing Values in Dataset')
            plt.ylabel('Count')
            plt.xlabel('Features')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            missing_values_path = 'reports/figures/missing_values.png'
            plt.savefig(missing_values_path)
            self.results['visualization_paths'].append(missing_values_path)
        
        # 2. Numerical features distributions
        if numerical_cols:
            # Determine grid size
            n_cols = min(3, len(numerical_cols))
            n_rows = max(1, int(np.ceil(len(numerical_cols) / n_cols)))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
            fig.suptitle('Distribution of Numerical Features', fontsize=16)
            
            # Flatten axes array if it's multi-dimensional
            if n_rows > 1 or n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].grid(True, linestyle='--', alpha=0.7)
            
            # Hide unused subplots
            for j in range(len(numerical_cols), len(axes)):
                fig.delaxes(axes[j])
                
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            numerical_dist_path = 'reports/figures/numerical_distributions.png'
            plt.savefig(numerical_dist_path)
            self.results['visualization_paths'].append(numerical_dist_path)
        
        # 3. Categorical features distributions
        if categorical_cols:
            # Limit to top 10 categorical columns if there are too many
            cat_cols_to_plot = categorical_cols[:10]
            
            # Determine grid size
            n_cols = min(2, len(cat_cols_to_plot))
            n_rows = max(1, int(np.ceil(len(cat_cols_to_plot) / n_cols)))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            fig.suptitle('Distribution of Categorical Features', fontsize=16)
            
            # Flatten axes array if it's multi-dimensional
            if n_rows > 1 or n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            for i, col in enumerate(cat_cols_to_plot):
                if i < len(axes):
                    # Get top categories
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, 'Count']
                    
                    # Limit to top 10 categories for readability
                    if len(value_counts) > 10:
                        value_counts = pd.concat([
                            value_counts.head(9),
                            pd.DataFrame({col: ['Other'], 'Count': [value_counts.iloc[9:]['Count'].sum()]})
                        ])
                    
                    sns.barplot(x=col, y='Count', data=value_counts, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
                    axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Hide unused subplots
            for j in range(len(cat_cols_to_plot), len(axes)):
                fig.delaxes(axes[j])
                
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            categorical_dist_path = 'reports/figures/categorical_distributions.png'
            plt.savefig(categorical_dist_path)
            self.results['visualization_paths'].append(categorical_dist_path)
        
        # 4. Correlation heatmap
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numerical_cols].corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                        cmap='coolwarm', square=True, linewidths=0.5)
            plt.title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            correlation_path = 'reports/figures/correlation_heatmap.png'
            plt.savefig(correlation_path)
            self.results['visualization_paths'].append(correlation_path)
        
        # 5. Target variable distribution
        if target_col:
            plt.figure(figsize=(10, 6))
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 15:
                # Categorical or discrete target
                sns.countplot(y=target_col, data=df, order=df[target_col].value_counts().index)
                plt.title(f'Distribution of Target Variable: {target_col}')
                plt.xlabel('Count')
                plt.ylabel(target_col)
            else:
                # Numerical target
                sns.histplot(df[target_col].dropna(), kde=True)
                plt.title(f'Distribution of Target Variable: {target_col}')
                plt.xlabel(target_col)
                plt.ylabel('Frequency')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            target_dist_path = 'reports/figures/target_distribution.png'
            plt.savefig(target_dist_path)
            self.results['visualization_paths'].append(target_dist_path)
        
        # 6. Bivariate analysis: top features vs target
        if target_col and len(numerical_cols) > 0:
            # Only proceed if we have a trained model with feature importances
            if self.model and hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = self.preprocessing_artifacts.get('feature_names', [])
                
                if len(feature_names) > 0:
                    # Get top 6 numerical features by importance
                    indices = np.argsort(importances)[::-1]
                    top_features = [feature_names[i] for i in indices 
                                    if feature_names[i] in numerical_cols][:6]
                    
                    if len(top_features) > 0:
                        # Create plots for top features vs target
                        n_cols = min(3, len(top_features))
                        n_rows = max(1, int(np.ceil(len(top_features) / n_cols)))
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
                        fig.suptitle(f'Top Features vs {target_col}', fontsize=16)
                        
                        # Flatten axes array if it's multi-dimensional
                        if n_rows > 1 or n_cols > 1:
                            axes = axes.flatten()
                        else:
                            axes = [axes]
                        
                        for i, feature in enumerate(top_features):
                            if i < len(axes):
                                if df[target_col].dtype == 'object' or df[target_col].nunique() < 15:
                                    # Categorical target
                                    sns.boxplot(x=target_col, y=feature, data=df, ax=axes[i])
                                    axes[i].set_title(f'{feature} vs {target_col}')
                                else:
                                    # Numerical target
                                    sns.scatterplot(x=feature, y=target_col, data=df, ax=axes[i])
                                    axes[i].set_title(f'{feature} vs {target_col}')
                                
                                axes[i].grid(True, linestyle='--', alpha=0.7)
                        
                        # Hide unused subplots
                        for j in range(len(top_features), len(axes)):
                            fig.delaxes(axes[j])
                            
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.9)
                        bivariate_path = 'reports/figures/bivariate_analysis.png'
                        plt.savefig(bivariate_path)
                        self.results['visualization_paths'].append(bivariate_path)
        
        print(f"Created {len(self.results['visualization_paths'])} dataset visualizations")
        return self
    
    def evaluate_model_performance(self, y_pred=None, y_prob=None):
        """
        Evaluate model performance with detailed metrics and visualizations
        """
        print("Evaluating model performance...")
        
        # If predictions not provided, generate them
        if y_pred is None and self.model is not None and self.X_test is not None:
            y_pred = self.model.predict(self.X_test)
        
        if y_prob is None and self.model is not None and self.X_test is not None:
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(self.X_test)
        
        if y_pred is None or self.y_test is None:
            print("Cannot evaluate model: missing test data or predictions")
            return self
        
        y_test = self.y_test
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Class-specific metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store metrics
        self.results['model_performance'] = {
            'basic_metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        # Additional metrics for binary classification
        if len(np.unique(y_test)) == 2 and y_prob is not None:
            try:
                # ROC AUC
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                self.results['model_performance']['roc_auc'] = roc_auc
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                self.results['model_performance']['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                average_precision = average_precision_score(y_test, y_prob[:, 1])
                self.results['model_performance']['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'average_precision': average_precision
                }
                
                # Log loss
                self.results['model_performance']['log_loss'] = log_loss(y_test, y_prob)
                
                # Calibration curve (reliability diagram)
                prob_true, prob_pred = calibration_curve(y_test, y_prob[:, 1], n_bins=10)
                self.results['model_performance']['calibration_curve'] = {
                    'prob_true': prob_true.tolist(),
                    'prob_pred': prob_pred.tolist()
                }
            except Exception as e:
                print(f"Error calculating additional metrics: {str(e)}")
        
        # Save model performance metrics to JSON
        metrics_path = 'reports/data/model_performance.json'
        with open(metrics_path, 'w') as f:
            json.dump(self._convert_to_serializable(self.results['model_performance']), f, indent=4)
        
        print(f"Model performance metrics saved to {metrics_path}")
        self.results['visualization_paths'].append(metrics_path)
        
        # Create model performance visualizations
        self._create_model_performance_visualizations()
        
        return self
    
    def _create_model_performance_visualizations(self):
        """
        Create visualizations for model performance metrics
        """
        model_perf = self.results.get('model_performance', {})
        if not model_perf:
            return
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(model_perf.get('confusion_matrix', []))
        if cm.size > 0:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            cm_path = 'reports/figures/confusion_matrix.png'
            plt.savefig(cm_path)
            self.results['visualization_paths'].append(cm_path)
        
        # 2. ROC Curve for binary classification
        if 'roc_curve' in model_perf:
            plt.figure(figsize=(10, 8))
            fpr = np.array(model_perf['roc_curve']['fpr'])
            tpr = np.array(model_perf['roc_curve']['tpr'])
            roc_auc = model_perf.get('roc_auc', 0)
            
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            roc_path = 'reports/figures/roc_curve.png'
            plt.savefig(roc_path)
            self.results['visualization_paths'].append(roc_path)
        
        # 3. Precision-Recall Curve for binary classification
        if 'pr_curve' in model_perf:
            plt.figure(figsize=(10, 8))
            precision = np.array(model_perf['pr_curve']['precision'])
            recall = np.array(model_perf['pr_curve']['recall'])
            avg_precision = model_perf['pr_curve'].get('average_precision', 0)
            
            plt.plot(recall, precision, label=f'Avg Precision = {avg_precision:.4f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            pr_path = 'reports/figures/pr_curve.png'
            plt.savefig(pr_path)
            self.results['visualization_paths'].append(pr_path)
        
        # 4. Calibration Curve
        if 'calibration_curve' in model_perf:
            plt.figure(figsize=(10, 8))
            prob_true = np.array(model_perf['calibration_curve']['prob_true'])
            prob_pred = np.array(model_perf['calibration_curve']['prob_pred'])
            
            plt.plot(prob_pred, prob_true, marker='o', label='Model')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Actual Probability')
            plt.title('Calibration Curve (Reliability Diagram)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            calibration_path = 'reports/figures/calibration_curve.png'
            plt.savefig(calibration_path)
            self.results['visualization_paths'].append(calibration_path)
        
        # 5. Model Metrics Comparison (if multiple models are evaluated)
        basic_metrics = model_perf.get('basic_metrics', {})
        if basic_metrics:
            plt.figure(figsize=(10, 6))
            metrics_df = pd.DataFrame({
                'Metric': list(basic_metrics.keys()),
                'Value': list(basic_metrics.values())
            })
            sns.barplot(x='Metric', y='Value', data=metrics_df)
            plt.title('Model Performance Metrics')
            plt.ylim(0, 1)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            metrics_plot_path = 'reports/figures/model_metrics.png'
            plt.savefig(metrics_plot_path)
            self.results['visualization_paths'].append(metrics_plot_path)
        
        # 6. Class-specific metrics
        class_report = model_perf.get('classification_report', {})
        if class_report and isinstance(class_report, dict):
            # Filter out non-class keys
            class_keys = [k for k in class_report.keys() 
                          if k not in ['accuracy', 'macro avg', 'weighted avg', 'samples avg']]
            
            if class_keys:
                plt.figure(figsize=(12, 8))
                metrics_data = []
                
                for cls in class_keys:
                    cls_metrics = class_report[cls]
                    for metric, value in cls_metrics.items():
                        if metric in ['precision', 'recall', 'f1-score']:
                            metrics_data.append({
                                'Class': cls,
                                'Metric': metric,
                                'Value': value
                            })
                
                metrics_df = pd.DataFrame(metrics_data)
                
                # Create plot
                sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_df)
                plt.title('Performance Metrics by Class')
                plt.xlabel('Class')
                plt.ylabel('Value')
                plt.ylim(0, 1)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                class_metrics_path = 'reports/figures/class_metrics.png'
                plt.savefig(class_metrics_path)
                self.results['visualization_paths'].append(class_metrics_path)
                
        print(f"Created model performance visualizations in 'reports/figures/' directory")
        return 
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance
        """
        print("Analyzing feature importance...")
        
        if self.model is None or self.X_test is None or self.y_test is None:
            print("Cannot analyze feature importance: missing model or test data")
            return self
        
        feature_names = self.preprocessing_artifacts.get('feature_names', [])
        if not feature_names:
            print("Feature names not available in preprocessing artifacts")
            return self
            
        # Store feature importance results
        importance_results = {}
        
        # 1. Built-in feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Store in results
            importance_results['built_in'] = {
                'feature_names': [feature_names[i] for i in indices],
                'importance_values': importances[indices].tolist()
            }
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            top_n = min(30, len(indices))
            plt.title(f'Top {top_n} Feature Importances')
            plt.bar(range(top_n), importances[indices[:top_n]], align='center')
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
            plt.tight_layout()
            importance_path = 'reports/figures/feature_importance.png'
            plt.savefig(importance_path)
            self.results['visualization_paths'].append(importance_path)
        
        # 2. Permutation importance
        try:
            print("Calculating permutation importance...")
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            # Sort by importance
            perm_indices = perm_importance.importances_mean.argsort()[::-1]
            
            # Store in results
            importance_results['permutation'] = {
                'feature_names': [feature_names[i] for i in perm_indices],
                'importance_values': perm_importance.importances_mean[perm_indices].tolist(),
                'importance_std': perm_importance.importances_std[perm_indices].tolist()
            }
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            top_n = min(30, len(perm_indices))
            plt.title(f'Top {top_n} Permutation Feature Importances')
            plt.bar(range(top_n), 
                   perm_importance.importances_mean[perm_indices[:top_n]], 
                   yerr=perm_importance.importances_std[perm_indices[:top_n]],
                   align='center')
            plt.xticks(range(top_n), [feature_names[i] for i in perm_indices[:top_n]], rotation=90)
            plt.tight_layout()
            perm_importance_path = 'reports/figures/permutation_importance.png'
            plt.savefig(perm_importance_path)
            self.results['visualization_paths'].append(perm_importance_path)
        except Exception as e:
            print(f"Error calculating permutation importance: {str(e)}")
        
        # Store feature importance in results
        self.results['feature_analysis'] = importance_results
        
        # Save feature importance to CSV
        if 'built_in' in importance_results:
            importance_df = pd.DataFrame({
                'Feature': importance_results['built_in']['feature_names'],
                'Importance': importance_results['built_in']['importance_values']
            })
            importance_csv_path = 'reports/data/feature_importance.csv'
            importance_df.to_csv(importance_csv_path, index=False)
            print(f"Feature importance saved to {importance_csv_path}")
            self.results['visualization_paths'].append(importance_csv_path)
        
        return self
    
    def create_interactive_dashboard(self):
        """
        Create an interactive HTML dashboard with all results
        """
        print("Creating interactive dashboard...")
        
        # Initialize dashboard
        dashboard = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Model Performance Metrics', 'Confusion Matrix',
                'Top Feature Importance', 'ROC Curve',
                'Target Distribution', 'Feature Correlation Heatmap'
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.1
        )
        
        # 1. Model Performance Metrics
        model_perf = self.results.get('model_performance', {})
        basic_metrics = model_perf.get('basic_metrics', {})
        if basic_metrics:
            metrics_df = pd.DataFrame({
                'Metric': list(basic_metrics.keys()),
                'Value': list(basic_metrics.values())
            })
            
            dashboard.add_trace(
                go.Bar(
                    x=metrics_df['Metric'], 
                    y=metrics_df['Value'],
                    text=metrics_df['Value'].round(3),
                    textposition='auto',
                    marker_color='royalblue'
                ),
                row=1, col=1
            )
            
            dashboard.update_yaxes(title_text="Value", range=[0, 1], row=1, col=1)
        
        # 2. Confusion Matrix
        cm = np.array(model_perf.get('confusion_matrix', []))
        if cm.size > 0:
            dashboard.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Predicted ' + str(i) for i in range(cm.shape[1])],
                    y=['Actual ' + str(i) for i in range(cm.shape[0])],
                    colorscale='Blues',
                    showscale=True,
                    text=cm,
                    texttemplate="%{text}",
                ),
                row=1, col=2
            )
        
        # 3. Feature Importance
        feat_analysis = self.results.get('feature_analysis', {})
        if 'built_in' in feat_analysis:
            # Get top 10 features
            top_n = 10
            feature_names = feat_analysis['built_in']['feature_names'][:top_n]
            importance_values = feat_analysis['built_in']['importance_values'][:top_n]
            
            dashboard.add_trace(
                go.Bar(
                    x=feature_names,
                    y=importance_values,
                    marker_color='mediumseagreen'
                ),
                row=2, col=1
            )
            
            dashboard.update_xaxes(tickangle=45, row=2, col=1)
            dashboard.update_yaxes(title_text="Importance", row=2, col=1)
        
        # 4. ROC Curve
        if 'roc_curve' in model_perf:
            fpr = np.array(model_perf['roc_curve']['fpr'])
            tpr = np.array(model_perf['roc_curve']['tpr'])
            roc_auc = model_perf.get('roc_auc', 0)
            
            dashboard.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'AUC = {roc_auc:.4f}',
                    line=dict(color='darkorange', width=2)
                ),
                row=2, col=2
            )
            
            dashboard.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='navy', dash='dash')
                ),
                row=2, col=2
            )
            
            dashboard.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=2, col=2)
            dashboard.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=2, col=2)
        
        # 5. Target Distribution
        target_col = self.preprocessing_artifacts.get('target_column')
        if target_col and self.original_data is not None:
            df = self.original_data
            target_counts = df[target_col].value_counts().reset_index()
            target_counts.columns = [target_col, 'Count']
            
            dashboard.add_trace(
                go.Bar(
                    x=target_counts[target_col],
                    y=target_counts['Count'],
                    marker_color='lightcoral'
                ),
                row=3, col=1
            )
            
            dashboard.update_xaxes(title_text=target_col, row=3, col=1)
            dashboard.update_yaxes(title_text="Count", row=3, col=1)
        
        # 6. Feature Correlation Heatmap
        if self.original_data is not None:
            df = self.original_data
            numerical_cols = self.results['dataset_stats'].get('numerical_columns', [])
            
            if len(numerical_cols) > 1:
                correlation_matrix = df[numerical_cols].corr().round(2)
                
                # Select a subset of features if there are too many
                if len(numerical_cols) > 15:
                    # Select top correlated features
                    abs_corr = correlation_matrix.abs().sum().sort_values(ascending=False)
                    top_cols = abs_corr.index[:15]
                    correlation_matrix = correlation_matrix.loc[top_cols, top_cols]
                
                dashboard.add_trace(
                    go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        colorscale='RdBu_r',
                        zmid=0,
                        showscale=True,
                    ),
                    row=3, col=2
                )
                
                dashboard.update_xaxes(tickangle=45, row=3, col=2)
        
        # Update layout
        dashboard.update_layout(
            title_text="Credit Risk Model Analysis Dashboard",
            height=1200,
            width=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        # Save dashboard as HTML
        dashboard_path = 'reports/credit_risk_dashboard.html'
        pio.write_html(dashboard, file=dashboard_path, auto_open=False)
        self.results['visualization_paths'].append(dashboard_path)
        
        print(f"Interactive dashboard saved to {dashboard_path}")
        return self
    
    def generate_full_report(self):
        """
        Generate a comprehensive HTML report with all analysis and visualizations
        """
        print("Generating comprehensive report...")
        
        # Report content
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Credit Risk Model Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                h2 {{
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                .metric-card {{
                    background: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2980b9;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .metrics-container .metric-card {{
                    flex: 0 0 22%;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .image-container {{
                    margin: 20px 0;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                    border: 1px solid #ddd;
                }}
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Credit Risk Model Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Executive Summary</h2>
                <p>
                    This report presents a comprehensive analysis of the credit risk model developed using 
                    Random Forest classification. The analysis includes dataset exploration, model performance evaluation,
                    and feature importance analysis.
                </p>
        """
        
        # Dataset Statistics
        dataset_stats = self.results.get('dataset_stats', {})
        basic_stats = dataset_stats.get('basic_stats', {})
        
        if basic_stats:
            report_html += f"""
                <h2>Dataset Overview</h2>
                <p>
                    The analysis is based on a dataset with {basic_stats.get('shape', (0, 0))[0]} rows and 
                    {basic_stats.get('shape', (0, 0))[1]} columns. 
            """
            
            if 'missing_percentage' in basic_stats:
                missing_pct = basic_stats['missing_percentage']
                missing_cols = sum(1 for val in missing_pct.values() if val > 0)
                report_html += f"""
                    The dataset contains {missing_cols} columns with missing values.
                </p>
                """
        
        # Model Performance
        model_perf = self.results.get('model_performance', {})
        basic_metrics = model_perf.get('basic_metrics', {})
        
        if basic_metrics:
            report_html += f"""
                <h2>Model Performance</h2>
                <div class="metrics-container">
            """
            
            for metric, value in basic_metrics.items():
                report_html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{value:.4f}</div>
                        <div class="metric-label">{metric.capitalize()}</div>
                    </div>
                """
            
            report_html += """
                </div>
            """
        
        # Add visualizations
        report_html += """
                <h2>Visualizations</h2>
        """
        
        # Key visualizations to include in the report
        key_viz = [
            ('confusion_matrix.png', 'Confusion Matrix'),
            ('feature_importance.png', 'Feature Importance'),
            ('roc_curve.png', 'ROC Curve'),
            ('model_metrics.png', 'Model Performance Metrics'),
            ('target_distribution.png', 'Target Variable Distribution'),
            ('numerical_distributions.png', 'Numerical Feature Distributions'),
            ('correlation_heatmap.png', 'Feature Correlation Heatmap')
        ]
        
        for viz_file, viz_title in key_viz:
            viz_path = f'reports/figures/{viz_file}'
            if os.path.exists(viz_path):
                report_html += f"""
                <h3>{viz_title}</h3>
                <div class="image-container">
                    <img src="figures/{viz_file}" alt="{viz_title}">
                </div>
                """
        
        # Feature Importance Analysis
        feat_analysis = self.results.get('feature_analysis', {})
        if 'built_in' in feat_analysis:
            # Get top 10 features
            top_n = 10
            feature_names = feat_analysis['built_in']['feature_names'][:top_n]
            importance_values = feat_analysis['built_in']['importance_values'][:top_n]
            
            report_html += f"""
                <h2>Top {top_n} Important Features</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            for name, value in zip(feature_names, importance_values):
                report_html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{value:.4f}</td>
                    </tr>
                """
                
            report_html += """
                </table>
            """
        
        # Classification Report
        class_report = model_perf.get('classification_report', {})
        if class_report:
            # Filter out non-class keys
            class_keys = [k for k in class_report.keys() 
                          if k not in ['accuracy', 'macro avg', 'weighted avg', 'samples avg']]
            
            if class_keys:
                report_html += f"""
                    <h2>Performance by Class</h2>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>Support</th>
                        </tr>
                """
                
                for cls in class_keys:
                    cls_metrics = class_report[cls]
                    report_html += f"""
                        <tr>
                            <td>{cls}</td>
                            <td>{cls_metrics.get('precision', 0):.4f}</td>
                            <td>{cls_metrics.get('recall', 0):.4f}</td>
                            <td>{cls_metrics.get('f1-score', 0):.4f}</td>
                            <td>{cls_metrics.get('support', 0)}</td>
                        </tr>
                    """
                    
                report_html += """
                    </table>
                """
        
        # Conclusion and recommendations
        report_html += """
                <h2>Conclusion and Recommendations</h2>
                <p>
                    The Random Forest model demonstrates good performance for credit risk prediction.
                    Feature importance analysis highlights the key factors influencing credit risk predictions.
                </p>
                <p>
                    Recommendations:
                </p>
                <ul>
                    <li>Consider the top features identified in the model for credit risk assessment</li>
                    <li>Monitor model performance over time as new data becomes available</li>
                    <li>Evaluate model fairness across different demographic groups</li>
                    <li>Consider implementing model interpretability techniques for individual predictions</li>
                </ul>
                
                <div class="footer">
                    <p>AB InBev Credit Risk Analysis Report • Generated with CreditRiskAnalytics</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_path = 'reports/credit_risk_analysis_report.html'
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        print(f"Comprehensive report saved to {report_path}")
        self.results['visualization_paths'].append(report_path)
        
        return self

def run_analytics(data_path=None, model_path='models/best_model.pkl', 
                artifacts_path='models/preprocessing_artifacts.pkl',
                output_dir='reports'):
    """
    Run the full analytics pipeline
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/data', exist_ok=True)
    
    # Create and run analytics
    analyzer = CreditRiskAnalytics()
    analyzer.load_from_files(data_path, model_path, artifacts_path)
    
    # Run all analytics
    (analyzer
        .analyze_dataset()
        .visualize_dataset()
        .evaluate_model_performance()
        .analyze_feature_importance()
        .create_interactive_dashboard()
        .generate_full_report()
    )
    
    print("\nAnalytics completed successfully!")
    print(f"Results saved to {output_dir}/ directory")
    
    # Return paths to key output files
    return {
        'dashboard': 'reports/credit_risk_dashboard.html',
        'report': 'reports/credit_risk_analysis_report.html',
        'metrics': 'reports/data/model_performance.json'
    } 