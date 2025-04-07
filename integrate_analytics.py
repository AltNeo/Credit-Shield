import os
import sys

def integrate_analytics_to_main():
    """
    Integrates the analytics module into the existing latestmodel.py file
    """
    print("Integrating analytics module into latestmodel.py...")
    
    # Read the current latestmodel.py content
    with open('latestmodel.py', 'r') as f:
        content = f.read()
    
    # Prepare imports to add
    analytics_imports = """import pickle
# Import analytics module if available
try:
    from model_analytics import run_analytics
except ImportError:
    print("Analytics module not found. Run 'pip install plotly' to enable advanced analytics.")
    run_analytics = None
"""
    
    # Prepare the code to add at the end of main function
    analytics_code = """
    # Run comprehensive analytics if module is available
    if run_analytics:
        print("\\nRunning comprehensive analytics...")
        try:
            run_analytics(file_path, 'models/best_model.pkl', 'models/preprocessing_artifacts.pkl')
            print("Analytics completed! Check 'reports/' directory for results.")
        except Exception as e:
            print(f"Error running analytics: {str(e)}")
            print("You can run analytics separately with 'python run_analytics.py'")
    else:
        print("\\nFor comprehensive analytics, run 'python run_analytics.py'")
"""
    
    # Add imports if they don't exist
    if "from model_analytics import run_analytics" not in content:
        # Find the end of import section
        import_end = content.find("warnings.filterwarnings('ignore')")
        if import_end > 0:
            # Add after the last import
            import_end = content.find('\n', import_end) + 1
            content = content[:import_end] + analytics_imports + content[import_end:]
    
    # Add analytics code to the main function
    if "Running comprehensive analytics" not in content:
        # Find the end of the main function
        main_end = content.find('print("\\nCredit risk model development completed successfully!")')
        if main_end > 0:
            # Find the end of the main function
            main_function_end = content.find('\n\nif __name__', main_end)
            if main_function_end > 0:
                content = content[:main_function_end] + analytics_code + content[main_function_end:]
    
    # Write the modified content back
    with open('latestmodel_with_analytics.py', 'w') as f:
        f.write(content)
    
    print("Integration complete!")
    print("A new file 'latestmodel_with_analytics.py' has been created with the integrated analytics.")
    print("You can now run 'python latestmodel_with_analytics.py' to train the model and run analytics in one step.")

if __name__ == "__main__":
    integrate_analytics_to_main() 