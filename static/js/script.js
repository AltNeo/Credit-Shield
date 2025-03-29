document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const nextToBorrower = document.getElementById('next-to-borrower');
    const nextToCredit = document.getElementById('next-to-credit');
    const nextToModel = document.getElementById('next-to-model');
    const prevToLoan = document.getElementById('prev-to-loan');
    const prevToBorrower = document.getElementById('prev-to-borrower');
    const prevToCredit = document.getElementById('prev-to-credit');
    const submitAssessment = document.getElementById('submit-assessment');
    const resultsContainer = document.getElementById('results');
    const backToForm = document.getElementById('back-to-form');
    const printReport = document.getElementById('print-report');
    const modelCards = document.querySelectorAll('.model-card');
    
    // Tab Navigation Logic
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Get the tab to show
            const tabId = this.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });
    
    // Form Navigation
    if (nextToBorrower) {
        nextToBorrower.addEventListener('click', function() {
            if (validateLoanInfo()) {
                switchToTab('borrower');
            }
        });
    }
    
    if (nextToCredit) {
        nextToCredit.addEventListener('click', function() {
            if (validateBorrowerInfo()) {
                switchToTab('credit');
            }
        });
    }
    
    if (nextToModel) {
        nextToModel.addEventListener('click', function() {
            if (validateCreditInfo()) {
                switchToTab('model');
            }
        });
    }
    
    if (prevToLoan) {
        prevToLoan.addEventListener('click', function() {
            switchToTab('loan');
        });
    }
    
    if (prevToBorrower) {
        prevToBorrower.addEventListener('click', function() {
            switchToTab('borrower');
        });
    }
    
    if (prevToCredit) {
        prevToCredit.addEventListener('click', function() {
            switchToTab('credit');
        });
    }
    
    // Model Card Selection
    if (modelCards) {
        modelCards.forEach(card => {
            card.addEventListener('click', function() {
                modelCards.forEach(c => c.classList.remove('selected'));
                this.classList.add('selected');
            });
        });
    }
    
    // Submit Assessment Button
    if (submitAssessment) {
        submitAssessment.addEventListener('click', function() {
            if (validateAll()) {
                // Get selected model type
                const selectedModel = document.querySelector('.model-card.selected');
                const modelType = selectedModel ? selectedModel.getAttribute('data-model') : 'default';
                
                // Prepare form data
                const formData = {
                    loanAmount: parseFloat(document.getElementById('loan-amount').value),
                    loanTerm: parseInt(document.getElementById('loan-term').value),
                    interestRate: parseFloat(document.getElementById('interest-rate').value),
                    monthlyPayment: parseFloat(document.getElementById('monthly-payment').value),
                    annualIncome: parseFloat(document.getElementById('annual-income').value),
                    employmentLength: parseInt(document.getElementById('employment-length').value),
                    homeOwnership: document.getElementById('home-ownership').value,
                    loanPurpose: document.getElementById('loan-purpose').value,
                    debtToIncome: parseFloat(document.getElementById('debt-to-income').value),
                    ficoLow: parseInt(document.getElementById('fico-low').value),
                    ficoHigh: parseInt(document.getElementById('fico-high').value),
                    delinquencies: parseInt(document.getElementById('delinquencies').value),
                    modelType: modelType
                };
                
                // Call the API for risk assessment
                fetch('/api/calculate-risk', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData),
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    // Update the UI with the results
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Error calculating risk:', error);
                    alert('There was an error calculating the risk assessment. Please try again.');
                });
            }
        });
    }
    
    // Back to Form Button
    if (backToForm) {
        backToForm.addEventListener('click', function() {
            resultsContainer.classList.add('hidden');
            document.querySelector('.form-container').style.display = 'block';
            switchToTab('loan');
        });
    }
    
    // Print Report Button
    if (printReport) {
        printReport.addEventListener('click', function() {
            window.print();
        });
    }
    
    // Form Validation Functions
    function validateLoanInfo() {
        const loanAmount = document.getElementById('loan-amount').value;
        const interestRate = document.getElementById('interest-rate').value;
        const monthlyPayment = document.getElementById('monthly-payment').value;
        
        if (!loanAmount || isNaN(loanAmount) || loanAmount <= 0) {
            showError('loan-amount', 'Please enter a valid loan amount');
            return false;
        }
        
        if (!interestRate || isNaN(interestRate) || interestRate <= 0) {
            showError('interest-rate', 'Please enter a valid interest rate');
            return false;
        }
        
        if (!monthlyPayment || isNaN(monthlyPayment) || monthlyPayment <= 0) {
            showError('monthly-payment', 'Please enter a valid monthly payment');
            return false;
        }
        
        return true;
    }
    
    function validateBorrowerInfo() {
        const annualIncome = document.getElementById('annual-income').value;
        
        if (!annualIncome || isNaN(annualIncome) || annualIncome <= 0) {
            showError('annual-income', 'Please enter a valid annual income');
            return false;
        }
        
        return true;
    }
    
    function validateCreditInfo() {
        const debtToIncome = document.getElementById('debt-to-income').value;
        const ficoLow = document.getElementById('fico-low').value;
        const ficoHigh = document.getElementById('fico-high').value;
        
        if (!debtToIncome || isNaN(debtToIncome) || debtToIncome < 0) {
            showError('debt-to-income', 'Please enter a valid debt-to-income ratio');
            return false;
        }
        
        if (!ficoLow || isNaN(ficoLow) || ficoLow < 300 || ficoLow > 850) {
            showError('fico-low', 'Please enter a valid FICO score (300-850)');
            return false;
        }
        
        if (!ficoHigh || isNaN(ficoHigh) || ficoHigh < 300 || ficoHigh > 850) {
            showError('fico-high', 'Please enter a valid FICO score (300-850)');
            return false;
        }
        
        if (parseInt(ficoLow) > parseInt(ficoHigh)) {
            showError('fico-high', 'High range must be greater than or equal to low range');
            return false;
        }
        
        return true;
    }
    
    function validateAll() {
        return validateLoanInfo() && validateBorrowerInfo() && validateCreditInfo();
    }
    
    function showError(inputId, message) {
        const input = document.getElementById(inputId);
        if (input) {
            input.classList.add('error');
            alert(message);
            input.focus();
        }
    }
    
    // Helper Functions
    function switchToTab(tabId) {
        // Remove active class from all buttons and panes
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabPanes.forEach(pane => pane.classList.remove('active'));
        
        // Add active class to specified tab
        const tabButton = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
        const tabPane = document.getElementById(`${tabId}-tab`);
        
        if (tabButton && tabPane) {
            tabButton.classList.add('active');
            tabPane.classList.add('active');
        }
    }
    
    function displayResults(result) {
        // Update the results UI
        document.querySelector('.grade').textContent = result.grade;
        document.querySelector('.grade-circle').style.backgroundColor = result.circleColor;
        document.querySelector('.risk-grade h3').textContent = result.riskLevel;
        document.querySelector('.progress-bar').style.width = `${result.riskPercent}%`;
        document.querySelector('.score-value').textContent = `${result.riskPercent}%`;
        
        // Generate risk factors list
        const riskFactorsList = document.querySelector('.risk-factors ul');
        riskFactorsList.innerHTML = '';
        
        // Create list based on the analysis
        result.factors.forEach(factor => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fas ${factor.positive ? 'fa-check-circle' : 'fa-exclamation-triangle'}"></i> ${factor.text}`;
            riskFactorsList.appendChild(li);
        });
        
        // Show results section
        document.querySelector('.form-container').style.display = 'none';
        resultsContainer.classList.remove('hidden');
    }
    
    // Initialize form input validation events
    const formInputs = document.querySelectorAll('.form-control');
    formInputs.forEach(input => {
        input.addEventListener('input', function() {
            this.classList.remove('error');
        });
    });
}); 