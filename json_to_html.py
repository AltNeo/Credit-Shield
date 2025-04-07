import json
import os
import datetime
import pandas as pd

def json_to_html(json_file, indent=4):
    """
    Convert JSON to formatted HTML with indentation and structure
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return f"<p>Error: Could not parse {json_file}</p>"
    
    # Function to convert JSON to HTML recursively
    def json_to_html_recursive(data, indent_level=0):
        html = ""
        if isinstance(data, dict):
            html += '<div class="json-object" style="margin-left: {}px;">'.format(indent_level * indent)
            
            for key, value in data.items():
                # For numerical statistics (handle special case)
                if isinstance(value, dict) and all(k in value for k in ['count', 'mean', 'std', 'min', 'max']):
                    html += '<div class="json-key-value">'
                    html += '<span class="json-key">{}</span>: '.format(key)
                    
                    # Create a small table for numerical statistics
                    html += '<table class="stats-table">'
                    html += '<tr><th>Stat</th><th>Value</th></tr>'
                    for k, v in value.items():
                        html += '<tr><td>{}</td><td>{}</td></tr>'.format(k, v)
                    html += '</table>'
                    html += '</div>'
                    
                else:
                    html += '<div class="json-key-value">'
                    html += '<span class="json-key">{}</span>: '.format(key)
                    
                    if isinstance(value, (dict, list)):
                        html += json_to_html_recursive(value, indent_level + 1)
                    else:
                        # Format numbers to be more readable
                        if isinstance(value, float):
                            formatted_value = "{:.4f}".format(value)
                        else:
                            formatted_value = str(value)
                        
                        html += '<span class="json-value">{}</span>'.format(formatted_value)
                    
                    html += '</div>'
            html += '</div>'
            
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and all(isinstance(item, dict) for item in data):
                # Try to create a table for lists of dictionaries (common format for data)
                # Get all possible keys
                all_keys = set()
                for item in data:
                    all_keys.update(item.keys())
                
                html += '<table class="json-table" style="margin-left: {}px;">'.format(indent_level * indent)
                
                # Table header
                html += '<tr>'
                for key in all_keys:
                    html += '<th>{}</th>'.format(key)
                html += '</tr>'
                
                # Table rows
                for item in data:
                    html += '<tr>'
                    for key in all_keys:
                        value = item.get(key, '')
                        if isinstance(value, (dict, list)):
                            cell_content = "complex"
                        else:
                            cell_content = str(value)
                        html += '<td>{}</td>'.format(cell_content)
                    html += '</tr>'
                
                html += '</table>'
            else:
                # Regular list
                html += '<ul class="json-array" style="margin-left: {}px;">'.format(indent_level * indent)
                
                for item in data:
                    html += '<li>'
                    if isinstance(item, (dict, list)):
                        html += json_to_html_recursive(item, indent_level + 1)
                    else:
                        html += '<span class="json-value">{}</span>'.format(item)
                    html += '</li>'
                
                html += '</ul>'
        else:
            html += '<span class="json-value">{}</span>'.format(data)
        
        return html
    
    # Generate HTML content
    json_filename = os.path.basename(json_file)
    html_content = json_to_html_recursive(data)
    
    return html_content

def create_html_report(json_files):
    """
    Create an HTML report from multiple JSON files
    """
    # Create HTML content for each JSON file
    content = ""
    
    for i, json_file in enumerate(json_files):
        if os.path.exists(json_file):
            file_id = f"file-{i}"
            file_name = os.path.basename(json_file)
            
            content += f'<div class="file-section">'
            content += f'<h2>{file_name} <span class="collapse-toggle" id="{file_id}-toggle" onclick="toggleCollapse(\'{file_id}\')">Hide</span></h2>'
            content += f'<div id="{file_id}" class="collapsible">'
            content += json_to_html(json_file)
            content += '</div></div>'
    
    # Add CSV files as tables if present
    csv_files = [f for f in os.listdir('reports/data') if f.endswith('.csv')]
    for csv_file in csv_files:
        file_path = os.path.join('reports/data', csv_file)
        file_id = f"file-csv-{csv_files.index(csv_file)}"
        
        try:
            df = pd.read_csv(file_path)
            
            content += f'<div class="file-section">'
            content += f'<h2>{csv_file} <span class="collapse-toggle" id="{file_id}-toggle" onclick="toggleCollapse(\'{file_id}\')">Hide</span></h2>'
            content += f'<div id="{file_id}" class="collapsible">'
            
            # Convert DataFrame to HTML table
            table_html = df.to_html(classes='json-table', border=0)
            content += table_html
            
            content += '</div></div>'
        except Exception as e:
            content += f'<div class="file-section"><h2>{csv_file}</h2><p>Error: {str(e)}</p></div>'
    
    # HTML template
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AB InBev Credit Risk Model Parameters</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                color: #333;
                background-color: #f9f9f9;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #2c3e50;
                margin-top: 30px;
                padding: 10px;
                background-color: #f5f5f5;
                border-left: 4px solid #3498db;
            }
            .json-object {
                margin-bottom: 10px;
            }
            .json-key {
                font-weight: bold;
                color: #2980b9;
            }
            .json-value {
                color: #27ae60;
            }
            .json-key-value {
                margin: 5px 0;
                line-height: 1.5;
            }
            .json-array {
                padding-left: 20px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .stats-table {
                width: auto;
                margin-left: 20px;
                display: inline-block;
            }
            .stats-table th, .stats-table td {
                padding: 3px 10px;
            }
            .json-table {
                margin: 10px 0;
            }
            .timestamp {
                color: #7f8c8d;
                font-size: 14px;
                margin-bottom: 20px;
            }
            .file-section {
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 1px dashed #ccc;
            }
            .collapse-toggle {
                cursor: pointer;
                color: #3498db;
                margin-left: 10px;
                font-size: 14px;
            }
            .collapsible {
                display: block;
            }
            footer {
                margin-top: 30px;
                text-align: center;
                color: #7f8c8d;
                font-size: 14px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }
        </style>
        <script>
            function toggleCollapse(id) {
                var element = document.getElementById(id);
                if (element.style.display === "none") {
                    element.style.display = "block";
                    document.getElementById(id + "-toggle").textContent = "Hide";
                } else {
                    element.style.display = "none";
                    document.getElementById(id + "-toggle").textContent = "Show";
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>AB InBev Credit Risk Model Parameters</h1>
            <p class="timestamp">Generated on: {timestamp}</p>
            
            {content}
            
            <footer>
                <p>AB InBev Credit Risk Analysis • Parameters Visualization</p>
            </footer>
        </div>
    </body>
    </html>
    """.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        content=content
    )
    
    return html

def main():
    # Get all JSON files in the reports/data directory
    data_dir = 'reports/data'
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found in reports/data directory.")
        return
    
    # Create HTML report
    html_content = create_html_report(json_files)
    
    # Write to file
    output_file = 'reports/parameters_visualization.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_file}")
    
    # Try to open the report in the default browser
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
        print("Report opened in your default web browser.")
    except:
        print("Could not open report automatically.")

if __name__ == "__main__":
    main() 