#!/bin/bash
# setup_trading_scheduler.sh - Setup script for the trading scheduler service

# Exit on error
set -e

echo "Setting up Trading System Scheduler..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root to set up the service"
  exit 1
fi

# Install required Python packages
echo "Installing required Python packages..."
pip3 install pytz

# Create log directory if it doesn't exist
if [ ! -d "/home/ubuntu/inavvi2/logs" ]; then
  echo "Creating logs directory..."
  mkdir -p /home/ubuntu/inavvi2/logs
fi

# Set permissions for log directory
chown -R ubuntu:ubuntu /home/ubuntu/inavvi2/logs
chmod -R 755 /home/ubuntu/inavvi2/logs

# Copy service file to systemd directory
echo "Installing systemd service..."
cp /home/ubuntu/inavvi2/scripts/trading-scheduler.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable and start the service
echo "Enabling and starting the service..."
systemctl enable trading-scheduler.service
systemctl start trading-scheduler.service

echo "Trading System Scheduler has been set up successfully!"
echo "You can check its status with: sudo systemctl status trading-scheduler.service"
echo "View logs with: sudo journalctl -u trading-scheduler.service -f"