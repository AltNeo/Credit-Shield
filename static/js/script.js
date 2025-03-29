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
        let isValid = true;
        
        // Validate loan amount
        const loanAmount = document.getElementById('loan-amount');
        if (!loanAmount.value || isNaN(loanAmount.value) || parseFloat(loanAmount.value) <= 0) {
            markInvalid(loanAmount, 'Please enter a valid loan amount');
            isValid = false;
        } else {
            markValid(loanAmount);
        }
        
        // Validate loan term
        const loanTerm = document.getElementById('loan-term');
        if (!loanTerm.value || isNaN(loanTerm.value) || parseInt(loanTerm.value) <= 0) {
            markInvalid(loanTerm, 'Please enter a valid loan term');
            isValid = false;
        } else {
            markValid(loanTerm);
        }
        
        // Validate interest rate
        const interestRate = document.getElementById('interest-rate');
        if (!interestRate.value || isNaN(interestRate.value) || parseFloat(interestRate.value) <= 0) {
            markInvalid(interestRate, 'Please enter a valid interest rate');
            isValid = false;
        } else {
            markValid(interestRate);
        }
        
        // Validate monthly payment
        const monthlyPayment = document.getElementById('monthly-payment');
        if (!monthlyPayment.value || isNaN(monthlyPayment.value) || parseFloat(monthlyPayment.value) <= 0) {
            markInvalid(monthlyPayment, 'Please enter a valid monthly payment');
            isValid = false;
        } else {
            markValid(monthlyPayment);
        }
        
        return isValid;
    }
    
    function validateBorrowerInfo() {
        let isValid = true;
        
        // Validate annual income
        const annualIncome = document.getElementById('annual-income');
        if (!annualIncome.value || isNaN(annualIncome.value) || parseFloat(annualIncome.value) <= 0) {
            markInvalid(annualIncome, 'Please enter a valid annual income');
            isValid = false;
        } else {
            markValid(annualIncome);
        }
        
        // Validate employment length
        const employmentLength = document.getElementById('employment-length');
        if (!employmentLength.value || isNaN(employmentLength.value) || parseInt(employmentLength.value) < 0) {
            markInvalid(employmentLength, 'Please enter a valid employment length');
            isValid = false;
        } else {
            markValid(employmentLength);
        }
        
        // Validate select fields
        const homeOwnership = document.getElementById('home-ownership');
        if (!homeOwnership.value) {
            markInvalid(homeOwnership, 'Please select a home ownership status');
            isValid = false;
        } else {
            markValid(homeOwnership);
        }
        
        const loanPurpose = document.getElementById('loan-purpose');
        if (!loanPurpose.value) {
            markInvalid(loanPurpose, 'Please select a loan purpose');
            isValid = false;
        } else {
            markValid(loanPurpose);
        }
        
        return isValid;
    }
    
    function validateCreditInfo() {
        let isValid = true;
        
        // Validate debt to income
        const debtToIncome = document.getElementById('debt-to-income');
        if (!debtToIncome.value || isNaN(debtToIncome.value) || parseFloat(debtToIncome.value) < 0) {
            markInvalid(debtToIncome, 'Please enter a valid debt-to-income ratio');
            isValid = false;
        } else {
            markValid(debtToIncome);
        }
        
        // Validate FICO scores
        const ficoLow = document.getElementById('fico-low');
        if (!ficoLow.value || isNaN(ficoLow.value) || parseInt(ficoLow.value) < 300 || parseInt(ficoLow.value) > 850) {
            markInvalid(ficoLow, 'Please enter a valid FICO score (300-850)');
            isValid = false;
        } else {
            markValid(ficoLow);
        }
        
        const ficoHigh = document.getElementById('fico-high');
        if (!ficoHigh.value || isNaN(ficoHigh.value) || parseInt(ficoHigh.value) < 300 || parseInt(ficoHigh.value) > 850 || 
            (ficoLow.value && parseInt(ficoHigh.value) < parseInt(ficoLow.value))) {
            markInvalid(ficoHigh, 'Please enter a valid FICO score (must be ≥ low score)');
            isValid = false;
        } else {
            markValid(ficoHigh);
        }
        
        // Validate delinquencies
        const delinquencies = document.getElementById('delinquencies');
        if (!delinquencies.value || isNaN(delinquencies.value) || parseInt(delinquencies.value) < 0) {
            markInvalid(delinquencies, 'Please enter a valid number of delinquencies');
            isValid = false;
        } else {
            markValid(delinquencies);
        }
        
        return isValid;
    }
    
    function validateAll() {
        return validateLoanInfo() && validateBorrowerInfo() && validateCreditInfo();
    }
    
    function markValid(input) {
        input.classList.remove('invalid');
        input.classList.add('valid');
    }
    
    function markInvalid(input, message) {
        input.classList.remove('valid');
        input.classList.add('invalid');
        
        // Show error message only if explicitly submitting a section
        if (message) {
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
            // Perform immediate validation when user types
            const inputId = this.id;
            
            if (inputId.includes('loan-') || inputId === 'interest-rate' || inputId === 'monthly-payment') {
                // Validate loan inputs
                if (inputId === 'loan-amount') {
                    if (!this.value || isNaN(this.value) || parseFloat(this.value) <= 0) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                } else if (inputId === 'loan-term') {
                    if (!this.value || isNaN(this.value) || parseInt(this.value) <= 0) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                } else if (inputId === 'interest-rate') {
                    if (!this.value || isNaN(this.value) || parseFloat(this.value) <= 0) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                } else if (inputId === 'monthly-payment') {
                    if (!this.value || isNaN(this.value) || parseFloat(this.value) <= 0) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                }
            } else if (inputId === 'annual-income' || inputId === 'employment-length' || 
                        inputId === 'home-ownership' || inputId === 'loan-purpose') {
                // Validate borrower inputs
                if (inputId === 'annual-income') {
                    if (!this.value || isNaN(this.value) || parseFloat(this.value) <= 0) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                } else if (inputId === 'employment-length') {
                    if (!this.value || isNaN(this.value) || parseInt(this.value) < 0) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                } else if (inputId === 'home-ownership' || inputId === 'loan-purpose') {
                    if (!this.value) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                }
            } else if (inputId === 'debt-to-income' || inputId === 'fico-low' || 
                        inputId === 'fico-high' || inputId === 'delinquencies') {
                // Validate credit inputs
                if (inputId === 'debt-to-income') {
                    if (!this.value || isNaN(this.value) || parseFloat(this.value) < 0) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                } else if (inputId === 'fico-low') {
                    if (!this.value || isNaN(this.value) || parseInt(this.value) < 300 || parseInt(this.value) > 850) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                        // Also revalidate the high score field if it exists
                        const ficoHigh = document.getElementById('fico-high');
                        if (ficoHigh && ficoHigh.value) {
                            if (parseInt(ficoHigh.value) < parseInt(this.value)) {
                                markInvalid(ficoHigh);
                            } else {
                                markValid(ficoHigh);
                            }
                        }
                    }
                } else if (inputId === 'fico-high') {
                    const ficoLow = document.getElementById('fico-low');
                    if (!this.value || isNaN(this.value) || parseInt(this.value) < 300 || parseInt(this.value) > 850 || 
                        (ficoLow && ficoLow.value && parseInt(this.value) < parseInt(ficoLow.value))) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                } else if (inputId === 'delinquencies') {
                    if (!this.value || isNaN(this.value) || parseInt(this.value) < 0) {
                        markInvalid(this);
                    } else {
                        markValid(this);
                    }
                }
            }
        });
        
        // Initial validation when the page loads
        input.dispatchEvent(new Event('input'));
    });
}); 