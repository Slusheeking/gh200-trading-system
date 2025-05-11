#!/bin/bash
# ssh_tunnel.sh - Set up SSH tunnel for GH200 monitoring

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
LOCAL_METRICS_PORT=8081
LOCAL_DASHBOARD_PORT=8082

echo "Setting up SSH tunnel for GH200 monitoring..."
echo "This will forward:"
echo "  - Local port $LOCAL_METRICS_PORT to remote port $METRICS_API_PORT"
echo "  - Local port $LOCAL_DASHBOARD_PORT to remote port $DASHBOARD_PORT"
echo ""
echo "After connecting, you can access:"
echo "  - Metrics API at http://localhost:$LOCAL_METRICS_PORT"
echo "  - Dashboard at http://localhost:$LOCAL_DASHBOARD_PORT"
echo ""
echo "Press Ctrl+C to close the tunnel when done."
echo ""

# Start SSH tunnel
ssh -L $LOCAL_METRICS_PORT:localhost:$METRICS_API_PORT -L $LOCAL_DASHBOARD_PORT:localhost:$DASHBOARD_PORT -N -T user@gh200-server