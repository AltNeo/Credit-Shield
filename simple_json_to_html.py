import json
import os
import pandas as pd
import datetime

def generate_html_report():
    """
    Generate an HTML report from JSON files in reports/data directory
    """
    data_dir = 'reports/data'
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not json_files and not csv_files:
        print("No data files found in reports/data directory.")
        return
    
    # Start building HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AB InBev Credit Risk Model Parameters</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; color: #333; background-color: #f9f9f9; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2c3e50; margin-top: 30px; padding: 10px; background-color: #f5f5f5; border-left: 4px solid #3498db; }
            h3 { margin-top: 20px; color: #2980b9; }
            pre { background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow: auto; word-wrap: break-word; }
            .key { color: #2980b9; font-weight: bold; }
            .string { color: #27ae60; }
            .number { color: #e67e22; }
            .boolean { color: #8e44ad; }
            .null { color: #bdc3c7; }
            table { border-collapse: collapse; width: 100%; margin: 15px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .timestamp { color: #7f8c8d; font-size: 14px; margin-bottom: 20px; }
            .file-section { margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px dashed #ccc; }
            .toggle-btn { cursor: pointer; padding: 5px 10px; background: #3498db; color: white; border: none; border-radius: 4px; font-size: 12px; margin-left: 10px; }
            .hidden { display: none; }
            footer { margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 14px; padding-top: 20px; border-top: 1px solid #eee; }
        </style>
        <script>
            function toggleVisibility(id) {
                var element = document.getElementById(id);
                var button = document.getElementById(id + '-btn');
                if (element.classList.contains('hidden')) {
                    element.classList.remove('hidden');
                    button.textContent = 'Hide';
                } else {
                    element.classList.add('hidden');
                    button.textContent = 'Show';
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>AB InBev Credit Risk Model Parameters</h1>
            <p class="timestamp">Generated on: """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """
    
    # Process JSON files
    for i, json_file in enumerate(json_files):
        file_path = os.path.join(data_dir, json_file)
        file_id = f"json-{i}"
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Format JSON prettily
                pretty_json = json.dumps(data, indent=4)
                
                html += f"""
                <div class="file-section">
                    <h2>{json_file} <button class="toggle-btn" id="{file_id}-btn" onclick="toggleVisibility('{file_id}')">Hide</button></h2>
                    <div id="{file_id}">
                        <pre>{pretty_json}</pre>
                    </div>
                </div>
                """
        except Exception as e:
            html += f"""
            <div class="file-section">
                <h2>{json_file}</h2>
                <p>Error reading file: {str(e)}</p>
            </div>
            """
    
    # Process CSV files
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(data_dir, csv_file)
        file_id = f"csv-{i}"
        
        try:
            df = pd.read_csv(file_path)
            
            html += f"""
            <div class="file-section">
                <h2>{csv_file} <button class="toggle-btn" id="{file_id}-btn" onclick="toggleVisibility('{file_id}')">Hide</button></h2>
                <div id="{file_id}">
                    {df.to_html(classes='dataframe')}
                </div>
            </div>
            """
        except Exception as e:
            html += f"""
            <div class="file-section">
                <h2>{csv_file}</h2>
                <p>Error reading file: {str(e)}</p>
            </div>
            """
    
    # Close HTML
    html += """
            <footer>
                <p>AB InBev Credit Risk Analysis • Parameters Visualization</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    output_file = 'reports/parameters_visualization.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML report generated: {output_file}")
    
    # Try to open the report in the default browser
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
        print("Report opened in your default web browser.")
    except:
        print("Could not open report automatically.")

if __name__ == "__main__":
    generate_html_report() 