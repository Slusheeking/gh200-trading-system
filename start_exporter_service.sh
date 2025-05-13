#!/bin/bash
# Start the system exporter API service

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

# Start the service
echo "Starting system exporter API service..."
systemctl start system-exporter-api.service

# Check status
echo "Service status:"
systemctl status system-exporter-api.service --no-pager

echo "System exporter API service started"
echo "The ngrok tunnel URL will be available in the service logs"
echo "View logs with: sudo journalctl -u system-exporter-api.service -f"
