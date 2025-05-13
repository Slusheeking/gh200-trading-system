#!/bin/bash

# Start System Exporter API with Ngrok Tunnel
# This script starts the system exporter, API server, and ngrok tunnel.

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "ngrok is not installed. Please install it first."
    exit 1
fi

# Check if required Python packages are installed
echo "Checking required Python packages..."
required_packages=("fastapi" "uvicorn" "pydantic" "requests" "pandas" "numpy" "psutil")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        missing_packages+=("$package")
    fi
done

# Install missing packages if any
if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "Installing missing Python packages: ${missing_packages[*]}"
    pip install "${missing_packages[@]}"
fi

# Create necessary directories
mkdir -p logs
mkdir -p exports
mkdir -p cache/yahoo_finance

# Make sure the collectors directory has an __init__.py file
mkdir -p src/monitoring/collectors
touch src/monitoring/collectors/__init__.py

# Start the system exporter API with ngrok tunnel
echo "Starting System Exporter API with Ngrok Tunnel..."
python3 src/api/start_exporter_api.py "$@"
