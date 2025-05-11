#!/bin/bash
# stop_monitoring.sh - Stop the monitoring services for GH200 Trading System

echo "Stopping GH200 Trading System Monitoring..."

# Check for PID files
if [ -f "../logs/metrics_api.pid" ]; then
    METRICS_PID=$(cat ../logs/metrics_api.pid)
    echo "Stopping Metrics API (PID $METRICS_PID)..."
    kill $METRICS_PID 2>/dev/null || true
    rm ../logs/metrics_api.pid
else
    echo "Metrics API PID file not found"
fi

if [ -f "../logs/dashboard.pid" ]; then
    DASHBOARD_PID=$(cat ../logs/dashboard.pid)
    echo "Stopping Dashboard (PID $DASHBOARD_PID)..."
    kill $DASHBOARD_PID 2>/dev/null || true
    rm ../logs/dashboard.pid
else
    echo "Dashboard PID file not found"
fi


echo "Monitoring services stopped"