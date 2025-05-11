#!/bin/bash
# stop.sh - Stop all components of the trading system

# Configuration
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/shutdown_${TIMESTAMP}.log"
MONGODB_DATA_DIR="/tmp/mongodb-data"
CUSTOM_MONGODB_PORT=27117

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Function to log messages
log() {
  local message="$1"
  local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[${timestamp}] $message" | tee -a ${LOG_FILE}
}

log "=== Stopping Trading System ($(date)) ==="

# Stop the main trading system
log "Stopping main trading system..."
pkill -f "trading_system" && log "Trading system stopped" || log "Trading system was not running"

# Stop monitoring services
log "Stopping monitoring services..."
if [ -d "monitoring" ] && [ -f "monitoring/scripts/stop_monitoring.sh" ]; then
  (cd monitoring && ./scripts/stop_monitoring.sh)
  log "Monitoring services stopped"
else
  log "Monitoring scripts not found"
fi

# Stop metrics scripts
log "Stopping metrics collection..."
pkill -f "node simulate_metrics.js" && log "Metrics simulation stopped" || log "Metrics simulation was not running"
pkill -f "node real_metrics.js" && log "Real metrics collection stopped" || log "Real metrics collection was not running"

# Stop tunnels
log "Stopping tunnels..."
if [ -d "monitoring" ] && [ -f "monitoring/scripts/stop_tunnel.sh" ]; then
  (cd monitoring && ./scripts/stop_tunnel.sh)
  log "Tunnels stopped"
else
  log "Tunnel scripts not found"
fi

# Stop Redis (if not managed by systemd)
log "Stopping Redis..."
if command -v redis-cli >/dev/null 2>&1; then
  if ! systemctl is-active --quiet redis-server 2>/dev/null; then
    redis-cli shutdown && log "Redis stopped" || log "Redis was not running or failed to stop"
  else
    log "Redis is managed by systemd, skipping shutdown"
  fi
else
  log "Redis CLI not found"
fi

# Stop MongoDB (if not managed by systemd)
log "Stopping MongoDB..."
if command -v mongod >/dev/null 2>&1; then
  if ! systemctl is-active --quiet mongodb 2>/dev/null; then
    # Try to stop the custom MongoDB instance
    if [ -d "${MONGODB_DATA_DIR}/db" ]; then
      log "Stopping custom MongoDB instance on port ${CUSTOM_MONGODB_PORT}..."
      mongod --dbpath ${MONGODB_DATA_DIR}/db --shutdown && log "MongoDB stopped" || {
        # Try alternative method using admin command
        if command -v mongosh >/dev/null 2>&1; then
          mongosh --port ${CUSTOM_MONGODB_PORT} --eval "db.adminCommand({shutdown: 1})" && log "MongoDB stopped via mongosh" || log "Failed to stop MongoDB via mongosh"
        elif command -v mongo >/dev/null 2>&1; then
          mongo --port ${CUSTOM_MONGODB_PORT} --eval "db.adminCommand({shutdown: 1})" && log "MongoDB stopped via mongo" || log "Failed to stop MongoDB via mongo"
        else
          # Last resort: kill the process
          pkill -f "mongod.*${CUSTOM_MONGODB_PORT}" && log "MongoDB stopped via pkill" || log "MongoDB was not running or failed to stop"
        fi
      }
    else
      # Try standard shutdown
      mongod --shutdown && log "MongoDB stopped" || log "MongoDB was not running or failed to stop"
    fi
  else
    log "MongoDB is managed by systemd, skipping shutdown"
  fi
else
  log "MongoDB not found"
fi

log "=== All components stopped ==="
