#!/bin/bash
# setup_tunnel.sh - Set up ngrok tunnel for GH200 monitoring

# Load environment variables
if [ -f "../../.env" ]; then
    source ../../.env
else
    echo "Error: .env file not found"
    exit 1
fi

# Check for ngrok auth token
if [ -z "$NGROK_AUTH_TOKEN" ]; then
    echo "Error: NGROK_AUTH_TOKEN not found in .env file"
    exit 1
fi

# Set default ports if not in environment
METRICS_API_PORT=${METRICS_API_PORT:-8000}
DASHBOARD_PORT=${DASHBOARD_PORT:-3000}

echo "Setting up ngrok tunnels for GH200 monitoring..."

# Configure ngrok
ngrok config add-authtoken $NGROK_AUTH_TOKEN

# Start ngrok for metrics API
echo "Starting ngrok tunnel for Metrics API (port $METRICS_API_PORT)..."
nohup ngrok http $METRICS_API_PORT --log=stdout > ../logs/ngrok_metrics.log &
NGROK_METRICS_PID=$!

# Start ngrok for dashboard
echo "Starting ngrok tunnel for Dashboard (port $DASHBOARD_PORT)..."
nohup ngrok http $DASHBOARD_PORT --log=stdout > ../logs/ngrok_dashboard.log &
NGROK_DASHBOARD_PID=$!

# Save PIDs for later shutdown
echo "$NGROK_METRICS_PID" > ../logs/ngrok_metrics.pid
echo "$NGROK_DASHBOARD_PID" > ../logs/ngrok_dashboard.pid

# Wait for tunnels to be established
sleep 5

# Get tunnel URLs
METRICS_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o 'http[^"]*' | head -1)
DASHBOARD_URL=$(curl -s http://localhost:4041/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o 'http[^"]*' | head -1)

echo "Ngrok tunnels established!"
echo "Metrics API: $METRICS_URL"
echo "Dashboard: $DASHBOARD_URL"
echo ""
echo "You can access these URLs from anywhere with internet access."
echo "To stop the tunnels, run ./stop_tunnel.sh"