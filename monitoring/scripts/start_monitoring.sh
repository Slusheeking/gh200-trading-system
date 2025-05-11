#!/bin/bash
# start_monitoring.sh - Start the monitoring services for GH200 Trading System

# Load environment variables
if [ -f "../../.env" ]; then
    source ../../.env
else
    echo "Error: .env file not found"
    exit 1
fi

# Set default ports if not in environment
METRICS_API_PORT=${METRICS_API_PORT:-8000}
DASHBOARD_PORT=${DASHBOARD_PORT:-3000}

# Create log directory
mkdir -p ../logs

echo "Starting GH200 Trading System Monitoring..."

# Start metrics API server
echo "Starting Metrics API on port $METRICS_API_PORT..."
cd ../api
nohup node metrics_api.js > ../logs/metrics_api.log 2>&1 &
METRICS_PID=$!
echo "Metrics API started with PID $METRICS_PID"

# Start dashboard server
echo "Starting Dashboard on port $DASHBOARD_PORT..."
cd ..
nohup node dashboard_server.js > ./logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "Dashboard started with PID $DASHBOARD_PID"

# Save PIDs for later shutdown
echo "$METRICS_PID" > ./logs/metrics_api.pid
echo "$DASHBOARD_PID" > ./logs/dashboard.pid

echo "Monitoring services started successfully!"
echo "Metrics API: http://localhost:$METRICS_API_PORT"
echo "Dashboard: http://localhost:$DASHBOARD_PORT"