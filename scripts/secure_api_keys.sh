#!/bin/bash
# secure_api_keys.sh - Securely store API keys in the credentials file
# This script should be run as root or with sudo

# Exit on error
set -e

echo "Securely storing API keys for GH200 Trading System..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root to set up API keys"
  exit 1
fi

# Check if credentials directory exists
if [ ! -d "/etc/trading-system" ]; then
  echo "Creating /etc/trading-system directory..."
  mkdir -p /etc/trading-system
fi

# Prompt for API keys
read -p "Polygon API Key: " polygon_key
read -p "Alpaca API Key: " alpaca_key
read -sp "Alpaca API Secret: " alpaca_secret
echo

# Check if credentials file exists
if [ -f "/etc/trading-system/credentials" ]; then
  # Backup existing credentials
  cp /etc/trading-system/credentials /etc/trading-system/credentials.bak
  echo "Backed up existing credentials to /etc/trading-system/credentials.bak"
  
  # Remove any existing API key entries
  sed -i '/POLYGON_API_KEY/d' /etc/trading-system/credentials
  sed -i '/ALPACA_API_KEY/d' /etc/trading-system/credentials
  sed -i '/ALPACA_API_SECRET/d' /etc/trading-system/credentials
else
  # Create new credentials file
  touch /etc/trading-system/credentials
fi

# Add API keys to credentials file
echo "POLYGON_API_KEY=$polygon_key" >> /etc/trading-system/credentials
echo "ALPACA_API_KEY=$alpaca_key" >> /etc/trading-system/credentials
echo "ALPACA_API_SECRET=$alpaca_secret" >> /etc/trading-system/credentials

# Set proper permissions
chmod 600 /etc/trading-system/credentials
chown trading-user:trading-user /etc/trading-system/credentials

echo "API keys have been securely stored in /etc/trading-system/credentials"
echo "File permissions set to 600 (read/write for owner only)"
echo "File ownership set to trading-user"