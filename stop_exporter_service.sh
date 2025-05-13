#!/bin/bash
# Stop the system exporter API service

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

# Stop the service
echo "Stopping system exporter API service..."
systemctl stop system-exporter-api.service

# Check status
echo "Service status:"
systemctl status system-exporter-api.service --no-pager || true

echo "System exporter API service stopped"
