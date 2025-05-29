var ctx = document.getElementById('modelComparisonChart').getContext('2d');
var modelComparisonChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression', 'SVM'],
        datasets: [{
            label: 'Accuracy',
            data: [0.85, 0.82, 0.80, 0.78, 0.76],
            backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0', '#9966FF'],
            borderColor: ['#2E86C1', '#E74C3C', '#F1C40F', '#3AAFAF', '#7A52CC'],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Accuracy' }
            },
            x: {
                title: { display: true, text: 'Model' }
            }
        },
        plugins: {
            legend: { display: true },
            title: { display: true, text: 'Model Comparison - Accuracy' }
        }
    }
});