#!/bin/bash
# Start the GH200 trading system with all required services
# This script starts Redis, system exporter API, and the scheduler services

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== Starting GH200 Trading System ==="

# Check if Redis is running, start if not (24/7 service)
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis server..."
    "$SCRIPT_DIR/redis_integration/start_redis.sh" &
    sleep 2  # Give Redis time to start
    if pgrep -x "redis-server" > /dev/null; then
        echo "✓ Redis server started successfully"
    else
        echo "✗ Failed to start Redis server"
        exit 1
    fi
else
    echo "✓ Redis server is already running"
fi

# Start system exporter API (24/7 service)
echo "Starting System Exporter API..."
if pgrep -f "python3.*start_exporter_api.py" > /dev/null; then
    echo "✓ System Exporter API is already running"
else
    "$SCRIPT_DIR/start_exporter_api.sh" > "$LOG_DIR/exporter_api.log" 2>&1 &
    EXPORTER_PID=$!
    echo $EXPORTER_PID > "$SCRIPT_DIR/.exporter.pid"
    echo "✓ System Exporter API started with PID: $EXPORTER_PID"
fi

# Start the trading scheduler service
echo "Starting trading scheduler service..."
if systemctl is-active --quiet trading-scheduler.service; then
    echo "✓ Trading scheduler service is already running"
else
    sudo systemctl start trading-scheduler.service
    if systemctl is-active --quiet trading-scheduler.service; then
        echo "✓ Trading scheduler service started successfully"
    else
        echo "✗ Failed to start trading scheduler service"
        exit 1
    fi
fi

# Start the trainer scheduler service
echo "Starting trainer scheduler service..."
if systemctl is-active --quiet trainer-scheduler.service; then
    echo "✓ Trainer scheduler service is already running"
else
    sudo systemctl start trainer-scheduler.service
    if systemctl is-active --quiet trainer-scheduler.service; then
        echo "✓ Trainer scheduler service started successfully"
    else
        echo "✗ Failed to start trainer scheduler service"
        exit 1
    fi
fi

echo "=== All services started successfully ==="
echo "The schedulers will automatically manage the trading system and trainer based on their configured schedules"
