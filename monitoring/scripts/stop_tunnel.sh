#!/bin/bash
# stop_tunnel.sh - Stop ngrok tunnels for GH200 monitoring

echo "Stopping ngrok tunnels..."

# Check for PID files
if [ -f "../logs/ngrok_metrics.pid" ]; then
    NGROK_METRICS_PID=$(cat ../logs/ngrok_metrics.pid)
    echo "Stopping Metrics API tunnel (PID $NGROK_METRICS_PID)..."
    kill $NGROK_METRICS_PID 2>/dev/null || true
    rm ../logs/ngrok_metrics.pid
else
    echo "Metrics API tunnel PID file not found"
fi

if [ -f "../logs/ngrok_dashboard.pid" ]; then
    NGROK_DASHBOARD_PID=$(cat ../logs/ngrok_dashboard.pid)
    echo "Stopping Dashboard tunnel (PID $NGROK_DASHBOARD_PID)..."
    kill $NGROK_DASHBOARD_PID 2>/dev/null || true
    rm ../logs/ngrok_dashboard.pid
else
    echo "Dashboard tunnel PID file not found"
fi

echo "Ngrok tunnels stopped"