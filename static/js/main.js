/**
 * Credit Risk Assessment Application
 * Main JavaScript File
 */

document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const creditRiskForm = document.getElementById('credit-risk-form');
    if (creditRiskForm) {
        creditRiskForm.addEventListener('submit', validateForm);
    }

    // Update risk indicator position if results are present
    updateRiskIndicator();

    // Initialize tooltips
    initTooltips();

    // Initialize form sliders if they exist
    initFormSliders();

    // Initialize charts on dashboard if we're on the dashboard page
    if (document.getElementById('dashboard-charts')) {
        initDashboardCharts();
    }

    // Fill form fields with default values if they exist
    // This helps pre-populate the form with reasonable values
    const defaultValues = {
        'age': 30,
        'income': 50000,
        'employment_years': 5,
        'debt_to_income_ratio': 0.3,
        'credit_score': 700,
        'loan_amount': 20000,
        'loan_term': 36,
        'loan_purpose': 'car',
        'previous_defaults': 0,
        'credit_inquiries': 0
    };

    // Set default values for the form fields if they're empty
    Object.keys(defaultValues).forEach(key => {
        const element = document.getElementById(key);
        if (element && !element.value) {
            if (element.tagName === 'SELECT') {
                // Handle select elements
                Array.from(element.options).forEach(option => {
                    if (option.value == defaultValues[key]) {
                        option.selected = true;
                    }
                });
            } else {
                // Handle input elements
                element.value = defaultValues[key];
            }
        }
    });

    // Check checkboxes based on defaults
    const mortgageCheckbox = document.getElementById('has_mortgage');
    if (mortgageCheckbox) {
        mortgageCheckbox.checked = false; // Default to unchecked
    }

    const creditCardCheckbox = document.getElementById('has_credit_card');
    if (creditCardCheckbox) {
        creditCardCheckbox.checked = true; // Default to checked
    }

    // Fix tooltips to ensure they appear correctly
    const tooltips = document.querySelectorAll('.tooltip');
    tooltips.forEach(tooltip => {
        tooltip.title = tooltip.getAttribute('data-tooltip');
    });
});

/**
 * Validate the credit risk assessment form
 */
function validateForm(e) {
    let isValid = true;
    const form = e.target;
    const inputs = form.querySelectorAll('input[required], select[required]');
    
    // Remove any existing error messages
    const errorMessages = form.querySelectorAll('.error-message');
    errorMessages.forEach(message => message.remove());
    
    // Check each required field
    inputs.forEach(input => {
        input.classList.remove('error');
        
        if (!input.value.trim()) {
            isValid = false;
            markFieldAsError(input, 'This field is required');
        } else if (input.type === 'number' || input.dataset.type === 'number') {
            const value = parseFloat(input.value);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            
            if (isNaN(value)) {
                isValid = false;
                markFieldAsError(input, 'Please enter a valid number');
            } else if (min !== undefined && value < min) {
                isValid = false;
                markFieldAsError(input, `Minimum value is ${min}`);
            } else if (max !== undefined && value > max) {
                isValid = false;
                markFieldAsError(input, `Maximum value is ${max}`);
            }
        }
    });
    
    if (!isValid) {
        e.preventDefault();
        
        // Scroll to the first error
        const firstError = form.querySelector('.error');
        if (firstError) {
            firstError.focus();
            firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    } else {
        // Show loading indicator
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.innerHTML = '<span class="loading-spinner"></span> Processing...';
            submitBtn.disabled = true;
        }
    }
}

/**
 * Mark a form field as having an error
 */
function markFieldAsError(field, message) {
    field.classList.add('error');
    
    const errorMessage = document.createElement('div');
    errorMessage.classList.add('error-message');
    errorMessage.innerText = message;
    
    const formGroup = field.closest('.form-group');
    if (formGroup) {
        formGroup.appendChild(errorMessage);
    } else {
        field.insertAdjacentElement('afterend', errorMessage);
    }
}

/**
 * Update the risk indicator position based on the risk probability
 */
function updateRiskIndicator() {
    const riskMeter = document.querySelector('.risk-meter');
    const riskIndicator = document.querySelector('.risk-indicator');
    const probabilityElement = document.getElementById('risk-probability');
    
    if (riskMeter && riskIndicator && probabilityElement) {
        const probability = parseFloat(probabilityElement.dataset.value) / 100;
        const position = probability * riskMeter.offsetWidth;
        riskIndicator.style.left = `${position}px`;
    }
}

/**
 * Initialize tooltips
 */
function initTooltips() {
    const tooltips = document.querySelectorAll('.tooltip');
    
    tooltips.forEach(tooltip => {
        const tooltipText = tooltip.querySelector('.tooltip-text');
        if (!tooltipText) {
            const span = document.createElement('span');
            span.classList.add('tooltip-text');
            span.textContent = tooltip.dataset.tooltip || 'Info';
            tooltip.appendChild(span);
        }
    });
}

/**
 * Initialize form sliders for numeric inputs
 */
function initFormSliders() {
    const sliderInputs = document.querySelectorAll('.range-slider');
    
    sliderInputs.forEach(input => {
        const slider = input.querySelector('input[type="range"]');
        const valueDisplay = input.querySelector('.slider-value');
        
        if (slider && valueDisplay) {
            // Initial value
            valueDisplay.textContent = slider.value;
            
            // Update value display when slider moves
            slider.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
                
                // Also update any linked input
                const linkedInputId = slider.dataset.linkedInput;
                if (linkedInputId) {
                    const linkedInput = document.getElementById(linkedInputId);
                    if (linkedInput) {
                        linkedInput.value = this.value;
                    }
                }
            });
        }
    });
}

/**
 * Initialize dashboard charts
 */
function initDashboardCharts() {
    // Sample data for charts
    const riskDistributionData = {
        labels: ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'],
        datasets: [{
            label: 'Risk Distribution',
            data: [25, 35, 20, 15, 5],
            backgroundColor: [
                '#2ecc71',
                '#27ae60',
                '#f39c12',
                '#e67e22',
                '#e74c3c'
            ]
        }]
    };
    
    const timeSeriesData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [{
            label: 'Average Risk Score',
            data: [35, 40, 38, 42, 33, 29],
            borderColor: '#3498db',
            backgroundColor: 'rgba(52, 152, 219, 0.1)',
            borderWidth: 2,
            fill: true
        }]
    };
    
    // Risk distribution chart (Pie Chart)
    const riskDistributionCtx = document.getElementById('risk-distribution-chart');
    if (riskDistributionCtx) {
        new Chart(riskDistributionCtx, {
            type: 'doughnut',
            data: riskDistributionData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    title: {
                        display: true,
                        text: 'Risk Distribution'
                    }
                }
            }
        });
    }
    
    // Time series chart (Line Chart)
    const timeSeriesCtx = document.getElementById('time-series-chart');
    if (timeSeriesCtx) {
        new Chart(timeSeriesCtx, {
            type: 'line',
            data: timeSeriesData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Risk Score Trend'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
}

/**
 * Make an AJAX request to the prediction API
 */
function getPrediction(data, callback) {
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (callback && typeof callback === 'function') {
            callback(null, data);
        }
    })
    .catch(error => {
        console.error('Error during prediction:', error);
        if (callback && typeof callback === 'function') {
            callback(error, null);
        }
    });
}

/**
 * Show a notification message
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.classList.add('notification', `notification-${type}`);
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

/**
 * Format currency values
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

/**
 * Format percentage values
 */
function formatPercent(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(value / 100);
} 