/**
 * Chart initialization and management
 * Creates and updates charts for system metrics
 */

// Chart configuration
const chartConfig = {
    system: {
        type: 'line',
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'second',
                        tooltipFormat: 'HH:mm:ss',
                        displayFormats: {
                            second: 'HH:mm:ss'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8e8e8e'
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8e8e8e'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e0e0'
                    }
                }
            }
        },
        data: {
            datasets: [
                {
                    label: 'CPU',
                    data: [],
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'Memory',
                    data: [],
                    borderColor: '#2196f3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'GPU',
                    data: [],
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        }
    },
    latency: {
        type: 'line',
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'second',
                        tooltipFormat: 'HH:mm:ss',
                        displayFormats: {
                            second: 'HH:mm:ss'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8e8e8e'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8e8e8e'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e0e0'
                    }
                }
            }
        },
        data: {
            datasets: [
                {
                    label: 'End-to-End',
                    data: [],
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'ML Inference',
                    data: [],
                    borderColor: '#2196f3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        }
    },
    pnl: {
        type: 'line',
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'second',
                        tooltipFormat: 'HH:mm:ss',
                        displayFormats: {
                            second: 'HH:mm:ss'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8e8e8e'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8e8e8e',
                        callback: function(value) {
                            return '$' + value;
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e0e0'
                    }
                }
            }
        },
        data: {
            datasets: [
                {
                    label: 'P&L',
                    data: [],
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        }
    }
};

// Chart instances
const charts = {};

// Initialize charts
function initCharts() {
    // System chart
    const systemCtx = document.getElementById('system-chart').getContext('2d');
    charts.system = new Chart(systemCtx, chartConfig.system);
    
    // Latency chart
    const latencyCtx = document.getElementById('latency-chart').getContext('2d');
    charts.latency = new Chart(latencyCtx, chartConfig.latency);
    
    // P&L chart
    const pnlCtx = document.getElementById('pnl-chart').getContext('2d');
    charts.pnl = new Chart(pnlCtx, chartConfig.pnl);
}

// Update charts with new data
function updateCharts(metrics) {
    const timestamp = new Date();
    
    // Update system chart
    charts.system.data.datasets[0].data.push({
        x: timestamp,
        y: metrics.system.cpu_usage
    });
    
    charts.system.data.datasets[1].data.push({
        x: timestamp,
        y: metrics.system.memory_usage
    });
    
    charts.system.data.datasets[2].data.push({
        x: timestamp,
        y: metrics.system.gpu_usage
    });
    
    // Limit data points
    if (charts.system.data.datasets[0].data.length > 100) {
        charts.system.data.datasets.forEach(dataset => {
            dataset.data.shift();
        });
    }
    
    // Update latency chart
    charts.latency.data.datasets[0].data.push({
        x: timestamp,
        y: metrics.latency.end_to_end
    });
    
    charts.latency.data.datasets[1].data.push({
        x: timestamp,
        y: metrics.latency.ml_inference
    });
    
    // Limit data points
    if (charts.latency.data.datasets[0].data.length > 100) {
        charts.latency.data.datasets.forEach(dataset => {
            dataset.data.shift();
        });
    }
    
    // Update P&L chart
    charts.pnl.data.datasets[0].data.push({
        x: timestamp,
        y: metrics.trading.pnl
    });
    
    // Limit data points
    if (charts.pnl.data.datasets[0].data.length > 100) {
        charts.pnl.data.datasets[0].data.shift();
    }
    
    // Update all charts
    Object.values(charts).forEach(chart => chart.update());
}

// Initialize charts when page loads
window.addEventListener('load', initCharts);