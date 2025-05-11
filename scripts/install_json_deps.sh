#!/bin/bash
# install_json_deps.sh - Install nlohmann_json library for development

echo "Installing nlohmann_json library for GH200 Trading System..."

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run with sudo to install system packages"
  exit 1
fi

# Install nlohmann_json and spdlog
echo "Installing nlohmann_json and spdlog packages..."
apt-get update
apt-get install -y nlohmann-json3-dev spdlog-dev

echo "Dependencies installed successfully!"
echo "You can now build the trading system with CMake."