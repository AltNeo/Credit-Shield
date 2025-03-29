# CreditShield - Credit Risk Assessment Tool

CreditShield is a professional web application designed to assess credit risk for loans. It provides a user-friendly interface for financial analysts to input loan and borrower information and receive a credit risk grade evaluation.

## Features

- Multi-step form with tabbed navigation
- Comprehensive loan information collection
- Credit risk assessment algorithm
- Professional and responsive UI
- Form validation to ensure data quality
- Risk grade visualization with A-G grading system
- Risk factors analysis and reporting
- Flask backend for data processing and model integration

## Technology Stack

- Python (Flask)
- HTML5
- CSS3
- JavaScript
- FontAwesome for icons
- Chart.js for data visualization

## Project Structure

```
creditshield/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── static/                 # Static assets
│   ├── css/
│   │   └── styles.css      # CSS styles for the application
│   └── js/
│       └── script.js       # JavaScript for form handling and risk calculation
├── templates/              # Flask templates
│   ├── index.html          # Main application page
│   ├── about.html          # About page
│   └── model_performance.html  # Model metrics page
├── logs/                   # Directory for assessment logs
└── README.md               # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd creditshield
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```
   python app.py
   ```

5. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Usage Guide

1. Fill out the Loan Information tab with details about the loan amount, term, interest rate, and monthly payment.
2. Navigate to the Borrower Information tab to provide details about the borrower's income, employment length, home ownership status, and loan purpose.
3. Complete the Credit Information tab with debt-to-income ratio, FICO score range, and delinquency history.
4. Select a risk assessment model on the Model Selection tab.
5. Click "Calculate Credit Risk" to generate the risk assessment.
6. Review the results showing the credit risk grade, risk percentage, and key risk factors.
7. Use the "Print Report" button to print the assessment or "Edit Information" to make changes.

## Credit Risk Grading System

The application uses a comprehensive A-G grading system:

- **A**: Very Low Credit Risk (Score: 90-100)
- **B**: Low Credit Risk (Score: 80-89)
- **C**: Moderate Credit Risk (Score: 70-79)
- **D**: Average Credit Risk (Score: 60-69)
- **E**: Elevated Credit Risk (Score: 50-59)
- **F**: High Credit Risk (Score: 40-49)
- **G**: Very High Credit Risk (Score: <40)

## Risk Calculation Algorithm

The credit risk algorithm considers multiple factors, including:

- FICO score range
- Debt-to-income ratio
- Employment history length
- Delinquency history
- Loan amount relative to income

Each factor is weighted to calculate a comprehensive risk score that determines the final grade.

## API Endpoints

### `/api/calculate-risk` (POST)

Calculates credit risk based on provided loan and borrower information.

**Request Body:**
```json
{
  "loanAmount": 10000,
  "loanTerm": 36,
  "interestRate": 10.0,
  "monthlyPayment": 325.0,
  "annualIncome": 60000,
  "employmentLength": 5,
  "homeOwnership": "rent",
  "loanPurpose": "debt-consolidation",
  "debtToIncome": 15.0,
  "ficoLow": 700,
  "ficoHigh": 720,
  "delinquencies": 0,
  "modelType": "default"
}
```

**Response:**
```json
{
  "grade": "B",
  "riskLevel": "Low Credit Risk",
  "riskPercent": 20,
  "circleColor": "#8bc34a",
  "score": 85,
  "factors": [
    {
      "positive": true,
      "text": "Strong credit score"
    },
    {
      "positive": true,
      "text": "Low debt-to-income ratio"
    },
    {
      "positive": true,
      "text": "Stable employment history"
    },
    {
      "positive": true,
      "text": "No delinquencies in past 2 years"
    }
  ],
  "timestamp": "2023-12-20T14:30:25.123456"
}
```

## Development

This project was developed as a capstone project for AB InBev. The code follows professional standards and best practices for maintainability and readability.

## License

© 2023 CreditShield. All rights reserved. 