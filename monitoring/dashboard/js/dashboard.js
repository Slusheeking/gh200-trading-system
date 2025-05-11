/**
 * Main dashboard logic
 * Handles UI updates and WebSocket data processing
 */

// Initialize dashboard
function initDashboard() {
    // Register WebSocket event handlers
    wsClient.on('onMetricsUpdate', updateDashboard);
    wsClient.on('onConnect', handleConnect);
    wsClient.on('onDisconnect', handleDisconnect);
}

// Handle WebSocket connection
function handleConnect() {
    console.log('Dashboard connected to metrics API');
}

// Handle WebSocket disconnection
function handleDisconnect() {
    console.log('Dashboard disconnected from metrics API');
}

// Update dashboard with new metrics
function updateDashboard(metrics) {
    // Update system metrics
    updateSystemMetrics(metrics.system);
    
    // Update latency metrics
    updateLatencyMetrics(metrics.latency);
    
    // Update trading metrics
    updateTradingMetrics(metrics.trading);
    
    // Update charts
    updateCharts(metrics);
}

// Update system metrics
function updateSystemMetrics(system) {
    document.getElementById('cpu-usage').textContent = system.cpu_usage.toFixed(1) + '%';
    document.getElementById('memory-usage').textContent = system.memory_usage.toFixed(1) + '%';
    document.getElementById('gpu-usage').textContent = system.gpu_usage.toFixed(1) + '%';
    
    // Color coding
    colorCode('cpu-usage', system.cpu_usage, 70, 90);
    colorCode('memory-usage', system.memory_usage, 70, 90);
    colorCode('gpu-usage', system.gpu_usage, 70, 90);
}

// Update latency metrics
function updateLatencyMetrics(latency) {
    updateLatencyBar('data-latency', 'data-latency-fill', latency.data_ingestion, 1000);
    updateLatencyBar('ml-latency', 'ml-latency-fill', latency.ml_inference, 2000);
    updateLatencyBar('risk-latency', 'risk-latency-fill', latency.risk_check, 500);
    updateLatencyBar('exec-latency', 'exec-latency-fill', latency.execution, 1000);
    updateLatencyBar('e2e-latency', 'e2e-latency-fill', latency.end_to_end, 5000);
}

// Update trading metrics
function updateTradingMetrics(trading) {
    document.getElementById('positions').textContent = trading.positions;
    document.getElementById('signals').textContent = trading.signals;
    document.getElementById('trades').textContent = trading.trades;
    
    // Update P&L with color coding
    const pnlElement = document.getElementById('pnl');
    const pnl = trading.pnl;
    pnlElement.textContent = '$' + pnl.toFixed(2);
    
    if (pnl > 0) {
        pnlElement.className = 'metric-value good';
    } else if (pnl < 0) {
        pnlElement.className = 'metric-value critical';
    } else {
        pnlElement.className = 'metric-value';
    }
}

// Update latency bar
function updateLatencyBar(labelId, fillId, value, maxValue) {
    const percent = Math.min((value / maxValue) * 100, 100);
    document.getElementById(labelId).textContent = value.toFixed(0) + ' μs';
    document.getElementById(fillId).style.width = percent + '%';
    
    // Color coding
    const fillElement = document.getElementById(fillId);
    if (percent < 50) {
        fillElement.style.backgroundColor = '#4caf50'; // Green
    } else if (percent < 80) {
        fillElement.style.backgroundColor = '#ff9800'; // Orange
    } else {
        fillElement.style.backgroundColor = '#f44336'; // Red
    }
}

// Color code an element based on value
function colorCode(elementId, value, warningThreshold, criticalThreshold) {
    const element = document.getElementById(elementId);
    
    if (value >= criticalThreshold) {
        element.className = 'metric-value critical';
    } else if (value >= warningThreshold) {
        element.className = 'metric-value warning';
    } else {
        element.className = 'metric-value good';
    }
}

// Initialize dashboard when page loads
window.addEventListener('load', initDashboard);