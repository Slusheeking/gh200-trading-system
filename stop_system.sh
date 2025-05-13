#!/bin/bash
# Stop the GH200 trading system scheduler services
# This script stops the scheduler services but leaves Redis and system exporter running

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "=== Stopping GH200 Trading System Schedulers ==="

# Stop the trading scheduler service
echo "Stopping trading scheduler service..."
if systemctl is-active --quiet trading-scheduler.service; then
    sudo systemctl stop trading-scheduler.service
    if ! systemctl is-active --quiet trading-scheduler.service; then
        echo "✓ Trading scheduler service stopped successfully"
    else
        echo "✗ Failed to stop trading scheduler service"
        exit 1
    fi
else
    echo "✓ Trading scheduler service is already stopped"
fi

# Stop the trainer scheduler service
echo "Stopping trainer scheduler service..."
if systemctl is-active --quiet trainer-scheduler.service; then
    sudo systemctl stop trainer-scheduler.service
    if ! systemctl is-active --quiet trainer-scheduler.service; then
        echo "✓ Trainer scheduler service stopped successfully"
    else
        echo "✗ Failed to stop trainer scheduler service"
        exit 1
    fi
else
    echo "✓ Trainer scheduler service is already stopped"
fi

echo "=== Trading system schedulers stopped successfully ==="
echo "Note: Redis and System Exporter API are still running (24/7 services)"
