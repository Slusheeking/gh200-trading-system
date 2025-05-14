#!/bin/bash

# Start Redis with our custom configuration
# This script starts Redis server with the optimized configuration for trading

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONFIG_PATH="$SCRIPT_DIR/redis.conf"

echo "Starting Redis with configuration: $CONFIG_PATH"
redis-server "$CONFIG_PATH"
