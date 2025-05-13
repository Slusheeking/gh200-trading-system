#!/bin/bash

# This script sets up a reserved domain for ngrok

# Ngrok auth token
AUTH_TOKEN="2vB4mEpkOKCPryJJTqcnQZu17mU_2mHUjAc8Gp4egYp8iDVRJ"

# Domain name to reserve
DOMAIN_NAME="inavvi"

# Configure ngrok with auth token
echo "Configuring ngrok with auth token..."
ngrok config add-authtoken $AUTH_TOKEN

# Check ngrok version
echo "Checking ngrok version..."
ngrok --version

# List available domains
echo "Listing available domains..."
ngrok api endpoints list

# For paid accounts, you need to use the ngrok dashboard to reserve domains
echo "For paid ngrok accounts, please follow these steps:"
echo "1. Go to https://dashboard.ngrok.com/cloud-edge/domains"
echo "2. Click 'New Domain'"
echo "3. Enter '$DOMAIN_NAME' as your domain name"
echo "4. Click 'Create Domain'"
echo ""
echo "After reserving your domain, you can use it with the --domain flag in ngrok commands:"
echo "ngrok http 8000 --domain=$DOMAIN_NAME.ngrok.io"

echo "Domain setup complete. You can now use $DOMAIN_NAME.ngrok.io in your ngrok tunnel configuration."