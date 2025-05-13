#!/bin/bash
# Install systemd services for trading system and trainer

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

# Copy service files to systemd directory
echo "Installing systemd service files..."
cp "$SCRIPT_DIR/systemd/trading-scheduler.service" /etc/systemd/system/
cp "$SCRIPT_DIR/systemd/trainer-scheduler.service" /etc/systemd/system/
cp "$SCRIPT_DIR/systemd/system-exporter-api.service" /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable and start services
echo "Enabling and starting services..."
systemctl enable trading-scheduler.service
systemctl enable trainer-scheduler.service
systemctl enable system-exporter-api.service
systemctl start trading-scheduler.service
systemctl start trainer-scheduler.service
systemctl start system-exporter-api.service

echo "Services installed and started"
echo "Check status with: systemctl status trading-scheduler.service"
echo "Check status with: systemctl status trainer-scheduler.service"
echo "Check status with: systemctl status system-exporter-api.service"
