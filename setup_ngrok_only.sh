#!/bin/bash
# Setup Ngrok Domain Only
# This script sets up the ngrok domain without starting the entire system

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Ngrok auth token and API key
AUTH_TOKEN="2vB4mEpkOKCPryJJTqcnQZu17mU_2mHUjAc8Gp4egYp8iDVRJ"
API_KEY="2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
DOMAIN_NAME="inavvi"

echo "=== Setting up Ngrok Domain Only ==="

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "ngrok is not installed. Please install it first."
    exit 1
fi

# Configure ngrok with auth token
echo "Configuring ngrok with auth token..."
ngrok config add-authtoken $AUTH_TOKEN

# Check if the domain is already in use
echo "Checking current ngrok tunnels..."
TUNNEL_CHECK=$(curl -s -H "Authorization: Bearer $API_KEY" -H "Ngrok-Version: 2" https://api.ngrok.com/tunnels | grep -c "$DOMAIN_NAME")

if [ "$TUNNEL_CHECK" -gt 0 ]; then
    echo "Domain $DOMAIN_NAME.ngrok.io is already in use."
    
    # Get the current tunnel URL
    TUNNEL_URL=$(curl -s -H "Authorization: Bearer $API_KEY" -H "Ngrok-Version: 2" https://api.ngrok.com/tunnels | grep -o "\"public_url\":\"[^\"]*\"" | head -1)
    echo "Current tunnel URL: ${TUNNEL_URL//\"public_url\":\"}"
else
    echo "Domain $DOMAIN_NAME.ngrok.io is not currently in use."
    echo "To use this domain, you need to:"
    echo "1. Make sure you have a paid ngrok account ($10/month)"
    echo "2. Reserve the domain in the ngrok dashboard (https://dashboard.ngrok.com/cloud-edge/domains)"
    echo "3. Use the domain with the --domain flag when starting ngrok:"
    echo "   ngrok http 8000 --domain=$DOMAIN_NAME.ngrok.io"
fi

echo ""
echo "=== Ngrok Domain Setup Complete ==="
echo "The domain $DOMAIN_NAME.ngrok.io is configured in src/api/ngrok_tunnel.py"
echo "When you start the system with ./start_system.sh, it will use this domain."