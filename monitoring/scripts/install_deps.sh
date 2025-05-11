#!/bin/bash
# install_deps.sh - Install dependencies for GH200 monitoring

echo "Installing dependencies for GH200 monitoring system..."

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "Node.js not found, installing..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "npm not found, installing..."
    sudo apt-get install -y npm
fi

# Check for ngrok
if ! command -v ngrok &> /dev/null; then
    echo "ngrok not found, installing..."
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
    sudo apt update
    sudo apt install -y ngrok
fi

# Create logs directory
mkdir -p ../logs

# Install Node.js dependencies for API
echo "Installing Node.js dependencies for API..."
cd ../api
npm install

# Install Node.js dependencies for main monitoring package
echo "Installing Node.js dependencies for monitoring package..."
cd ..
npm install

# Create .env file if it doesn't exist
if [ ! -f "../../.env" ]; then
    echo "Creating default .env file..."
    echo "METRICS_API_PORT=8000" > ../../.env
    echo "DASHBOARD_PORT=3000" >> ../../.env
    echo "# Add your NGROK_AUTH_TOKEN here" >> ../../.env
    echo "# NGROK_AUTH_TOKEN=your_token_here" >> ../../.env
    echo "LOG_LEVEL=info" >> ../../.env
    echo ".env file created. Please edit it to add your ngrok auth token."
fi

echo "Dependencies installed successfully!"
echo "To start the monitoring system, run ./start_monitoring.sh"