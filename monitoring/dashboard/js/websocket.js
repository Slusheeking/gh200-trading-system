/**
 * WebSocket connection management
 * Handles connection to metrics API and data processing
 */

// Configuration
const METRICS_API_URL = 'https://c206b1f8aca8.ngrok.app'; // Ngrok URL for metrics API

// Socket.IO connection
let socket;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
let reconnectTimeout;

// Event callbacks
const eventCallbacks = {
    onMetricsUpdate: [],
    onConnect: [],
    onDisconnect: []
};

// Connect to Socket.IO server
function connect() {
    // Clear any existing reconnect timeout
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
    }
    
    // Update connection status
    updateConnectionStatus(false, 'Connecting...');
    
    // Create Socket.IO connection
    socket = io(METRICS_API_URL, {
        reconnection: false, // We'll handle reconnection ourselves
        transports: ['websocket']
    });
    
    socket.on('connect', function() {
        console.log('Connected to metrics API');
        reconnectAttempts = 0;
        updateConnectionStatus(true, 'Connected');
        
        // Trigger onConnect callbacks
        eventCallbacks.onConnect.forEach(callback => callback());
    });
    
    socket.on('metrics', function(data) {
        try {
            // Trigger onMetricsUpdate callbacks
            eventCallbacks.onMetricsUpdate.forEach(callback => callback(data));
        } catch (error) {
            console.error('Error processing Socket.IO message:', error);
        }
    });
    
    socket.on('disconnect', function() {
        updateConnectionStatus(false, 'Disconnected');
        
        // Trigger onDisconnect callbacks
        eventCallbacks.onDisconnect.forEach(callback => callback());
        
        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})...`);
            
            reconnectTimeout = setTimeout(connect, delay);
        } else {
            console.error('Max reconnect attempts reached');
            updateConnectionStatus(false, 'Connection failed');
        }
    });
    
    socket.on('connect_error', function(error) {
        console.error('Socket.IO connection error:', error);
    });
}

// Update connection status UI
function updateConnectionStatus(connected, message) {
    const statusElement = document.getElementById('connection-status');
    if (statusElement) {
        statusElement.textContent = message || (connected ? 'Connected' : 'Disconnected');
        statusElement.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
    }
}

// Register event callback
function on(event, callback) {
    if (eventCallbacks[event]) {
        eventCallbacks[event].push(callback);
    }
}

// WebSocket API
const wsClient = {
    connect,
    on
};

// Connect when page loads
window.addEventListener('load', connect);