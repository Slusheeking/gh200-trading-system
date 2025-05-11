#!/bin/bash
# start.sh - Start all components of the trading system for production
# This script starts databases, monitoring, and the main trading system

# Configuration
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/startup_${TIMESTAMP}.log"
MONGODB_PORT=27017
REDIS_PORT=6379
METRICS_API_PORT=8000
DASHBOARD_PORT=3000
MONGODB_DATA_DIR="/tmp/mongodb-data"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Function to display help
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Start all components of the trading system for production."
  echo ""
  echo "Options:"
  echo "  --help          Display this help message and exit"
  echo "  --no-db         Skip starting databases (MongoDB and Redis)"
  echo "  --no-monitoring Skip starting monitoring components"
  echo "  --no-trading    Skip starting the main trading system"
  echo ""
  echo "Examples:"
  echo "  $0              Start everything"
  echo "  $0 --no-db      Start everything except databases"
  exit 0
}

# Parse command line arguments
NO_DB=false
NO_MONITORING=false
NO_TRADING=false

for arg in "$@"; do
  case $arg in
    --help)
      show_help
      ;;
    --no-db)
      NO_DB=true
      ;;
    --no-monitoring)
      NO_MONITORING=true
      ;;
    --no-trading)
      NO_TRADING=true
      ;;
    *)
      echo "Unknown option: $arg"
      show_help
      ;;
  esac
done

# Function to log messages
log() {
  local message="$1"
  local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[${timestamp}] $message" | tee -a ${LOG_FILE}
}

# Function to check if a port is in use
is_port_in_use() {
  local port=$1
  if lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null 2>&1; then
    return 0  # Port is in use
  else
    return 1  # Port is not in use
  fi
}

# Function to check if a process is running
is_process_running() {
  local process_name=$1
  if pgrep -f "$process_name" > /dev/null 2>&1; then
    return 0  # Process is running
  else
    return 1  # Process is not running
  fi
}

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to start MongoDB
start_mongodb() {
  if [ "$NO_DB" = true ]; then
    log "Skipping MongoDB startup (--no-db flag provided)"
    return 0
  fi

  log "Starting MongoDB..."
  
  # Use a different port to avoid conflicts
  CUSTOM_MONGODB_PORT=27117
  
  if is_port_in_use ${CUSTOM_MONGODB_PORT}; then
    log "MongoDB is already running on port ${CUSTOM_MONGODB_PORT}"
    # Set the global port variable to the custom port
    MONGODB_PORT=${CUSTOM_MONGODB_PORT}
    return 0
  fi

  # Check if MongoDB is managed by systemd
  if command_exists systemctl && systemctl is-active --quiet mongodb 2>/dev/null; then
    log "MongoDB is managed by systemd and is already running"
    return 0
  fi

  # Check if MongoDB is installed
  if ! command_exists mongod; then
    log "WARNING: MongoDB is not installed. Skipping MongoDB startup."
    log "To install MongoDB: sudo apt-get install -y mongodb"
    return 1
  fi

  # Create data directory if it doesn't exist
  mkdir -p ${MONGODB_DATA_DIR}
  mkdir -p ${MONGODB_DATA_DIR}/db
  mkdir -p ${MONGODB_DATA_DIR}/socket
  log "Using MongoDB data directory: ${MONGODB_DATA_DIR}"
  
  # Create a custom MongoDB configuration file with a custom socket path
  MONGODB_CONFIG="${LOG_DIR}/mongodb.conf"
  cat > ${MONGODB_CONFIG} << EOF
storage:
  dbPath: ${MONGODB_DATA_DIR}/db
  journal:
    enabled: true
systemLog:
  destination: file
  path: ${LOG_DIR}/mongodb.log
  logAppend: true
net:
  port: ${CUSTOM_MONGODB_PORT}
  bindIp: 127.0.0.1
  unixDomainSocket:
    enabled: true
    pathPrefix: ${MONGODB_DATA_DIR}/socket
processManagement:
  fork: true
EOF

  # Start MongoDB with the custom configuration
  log "Starting MongoDB server on port ${CUSTOM_MONGODB_PORT}..."
  mongod --config ${MONGODB_CONFIG} || {
    log "ERROR: Failed to start MongoDB. Check ${LOG_DIR}/mongodb.log for details."
    # Try to start with minimal options as a fallback
    log "Trying fallback method..."
    mongod --fork --logpath ${LOG_DIR}/mongodb.log --dbpath ${MONGODB_DATA_DIR}/db --port ${CUSTOM_MONGODB_PORT} || {
      log "ERROR: MongoDB startup failed with fallback method too."
      log "WARNING: Continuing without MongoDB. Some features may not work properly."
      return 1
    }
  }
  
  # Verify MongoDB is running
  sleep 2
  if is_port_in_use ${CUSTOM_MONGODB_PORT}; then
    log "MongoDB started successfully on port ${CUSTOM_MONGODB_PORT}"
    # Set the global port variable to the custom port
    MONGODB_PORT=${CUSTOM_MONGODB_PORT}
    return 0
  else
    log "ERROR: MongoDB failed to start properly"
    log "WARNING: Continuing without MongoDB. Some features may not work properly."
    return 1
  fi
}

# Function to start Redis
start_redis() {
  if [ "$NO_DB" = true ]; then
    log "Skipping Redis startup (--no-db flag provided)"
    return 0
  fi

  log "Starting Redis..."
  if is_port_in_use ${REDIS_PORT}; then
    log "Redis is already running on port ${REDIS_PORT}"
    return 0
  fi

  # Check if Redis is managed by systemd
  if command_exists systemctl && systemctl is-active --quiet redis-server 2>/dev/null; then
    log "Redis is managed by systemd and is already running"
    return 0
  fi

  # Check if Redis is installed
  if ! command_exists redis-server; then
    log "WARNING: Redis is not installed. Skipping Redis startup."
    log "To install Redis: sudo apt-get install -y redis-server"
    return 1
  fi

  # Create Redis data directory
  REDIS_DATA_DIR="${MONGODB_DATA_DIR}/redis"
  mkdir -p ${REDIS_DATA_DIR}
  log "Using Redis data directory: ${REDIS_DATA_DIR}"

  # Create a custom Redis configuration file
  REDIS_CONFIG="${LOG_DIR}/redis.conf"
  cat > ${REDIS_CONFIG} << EOF
port ${REDIS_PORT}
daemonize yes
dir ${REDIS_DATA_DIR}
logfile ${LOG_DIR}/redis.log
pidfile ${REDIS_DATA_DIR}/redis.pid
bind 127.0.0.1
EOF

  # Start Redis with the custom configuration
  log "Starting Redis server..."
  redis-server ${REDIS_CONFIG} || {
    log "ERROR: Failed to start Redis with custom config. Trying fallback method..."
    redis-server --daemonize yes --logfile ${LOG_DIR}/redis.log --dir ${REDIS_DATA_DIR} || {
      log "ERROR: Failed to start Redis. Check ${LOG_DIR}/redis.log for details."
      return 1
    }
  }
  
  # Verify Redis is running
  sleep 1
  if is_port_in_use ${REDIS_PORT}; then
    log "Redis started successfully"
    return 0
  else
    log "ERROR: Redis failed to start properly"
    return 1
  fi
}

# Function to start monitoring services
start_monitoring() {
  if [ "$NO_MONITORING" = true ]; then
    log "Skipping monitoring startup (--no-monitoring flag provided)"
    return 0
  fi

  log "Starting monitoring services..."
  
  # Check if Node.js is installed
  if ! command_exists node; then
    log "WARNING: Node.js is not installed. Skipping monitoring startup."
    log "To install Node.js: curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash - && sudo apt-get install -y nodejs"
    return 1
  fi
  
  # Start metrics API
  if is_port_in_use ${METRICS_API_PORT}; then
    log "Metrics API is already running on port ${METRICS_API_PORT}"
  else
    log "Starting Metrics API server..."
    if [ -d "monitoring" ] && [ -f "monitoring/api/metrics_api.js" ]; then
      cd monitoring && nohup node api/metrics_api.js > ${LOG_DIR}/metrics_api.log 2>&1 &
      local metrics_api_pid=$!
      cd - > /dev/null
      log "Metrics API server started (PID: ${metrics_api_pid})"
    else
      log "ERROR: Metrics API files not found. Check monitoring/api directory."
      return 1
    fi
  fi
  
  # Start dashboard server
  if is_port_in_use ${DASHBOARD_PORT}; then
    log "Dashboard server is already running on port ${DASHBOARD_PORT}"
  else
    log "Starting Dashboard server..."
    if [ -d "monitoring" ] && [ -f "monitoring/dashboard_server.js" ]; then
      cd monitoring && nohup node dashboard_server.js > ${LOG_DIR}/dashboard.log 2>&1 &
      local dashboard_pid=$!
      cd - > /dev/null
      log "Dashboard server started (PID: ${dashboard_pid})"
    else
      log "ERROR: Dashboard server file not found. Check monitoring directory."
      return 1
    fi
  fi
  
  # Start real metrics collection
  log "Starting real metrics collection..."
  if [ -d "monitoring" ] && [ -f "monitoring/real_metrics.js" ]; then
    cd monitoring && nohup node real_metrics.js > ${LOG_DIR}/real_metrics.log 2>&1 &
    local metrics_pid=$!
    cd - > /dev/null
    log "Real metrics collection started (PID: ${metrics_pid})"
  else
    log "WARNING: Real metrics script not found at monitoring/real_metrics.js"
  fi
  
  # Start ngrok tunnels
  if is_process_running "ngrok"; then
    log "Ngrok is already running"
  else
    if command_exists ngrok; then
      log "Starting Ngrok tunnels..."
      if [ -f "config/ngrok_tunnels.yml" ]; then
        nohup ngrok start --all --config=config/ngrok_tunnels.yml > ${LOG_DIR}/ngrok.log 2>&1 &
        local ngrok_pid=$!
        log "Ngrok tunnels started (PID: ${ngrok_pid})"
      else
        log "ERROR: Ngrok configuration file not found at config/ngrok_tunnels.yml"
        return 1
      fi
    else
      log "WARNING: Ngrok is not installed. Skipping tunnel setup."
      log "To install Ngrok: curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok"
    fi
  fi
  
  return 0
}

# Function to start the main trading system
start_trading_system() {
  if [ "$NO_TRADING" = true ]; then
    log "Skipping trading system startup (--no-trading flag provided)"
    return 0
  fi

  log "Starting main trading system..."
  
  # Check if trading system is already running
  if is_process_running "trading_system"; then
    log "Trading system is already running"
    return 0
  fi
  
  # Check if the binary exists
  if [ -f "./bin/trading_system" ]; then
    # Check if configuration files exist
    if [ ! -f "./config/system.yaml" ]; then
      log "ERROR: System configuration file not found at ./config/system.yaml"
      return 1
    fi
    
    if [ ! -f "./config/trading.yaml" ]; then
      log "ERROR: Trading configuration file not found at ./config/trading.yaml"
      return 1
    fi
    
    # Start the trading system with configuration
    log "Starting trading system binary..."
    nohup ./bin/trading_system --config ./config/system.yaml --trading ./config/trading.yaml > ${LOG_DIR}/trading_system.log 2>&1 &
    local trading_pid=$!
    log "Trading system started (PID: ${trading_pid})"
    return 0
  else
    log "ERROR: Trading system binary not found at ./bin/trading_system"
    log "Please build the system first using: cmake . && make"
    return 1
  fi
}

# Function to check credentials
check_credentials() {
  log "Checking credentials..."
  if [ ! -f "/etc/trading-system/credentials" ]; then
    log "WARNING: Secure credentials file not found at /etc/trading-system/credentials"
    log "You may need to run scripts/secure_api_keys.sh to set up API keys"
    return 1
  else
    log "Credentials file found"
    return 0
  fi
}

# Main execution
log "=== Starting Trading System ($(date)) ==="

# Check credentials first
check_credentials

# Start databases (if not skipped)
if [ "$NO_DB" != true ]; then
  start_mongodb
  start_redis
fi

# Start monitoring (if not skipped)
if [ "$NO_MONITORING" != true ]; then
  start_monitoring
fi

# Start the main trading system (if not skipped)
if [ "$NO_TRADING" != true ]; then
  start_trading_system
fi

log "=== All components started successfully ==="

# Display access information
if [ "$NO_MONITORING" != true ]; then
  log "Monitoring dashboard: http://localhost:${DASHBOARD_PORT}"
  log "Metrics API: http://localhost:${METRICS_API_PORT}/metrics"
fi

log "Log files are available in ${LOG_DIR}"

# Display ngrok URLs if available
if [ -f "${LOG_DIR}/ngrok.log" ] && [ "$NO_MONITORING" != true ]; then
  sleep 5  # Give ngrok time to establish tunnels
  log "Ngrok tunnel URLs:"
  grep -o "https://.*\.ngrok\.io" ${LOG_DIR}/ngrok.log 2>/dev/null | sort | uniq | while read url; do
    log "  $url"
  done
fi

log "To stop all services, run: ./stop.sh"

# Create a simple stop script if it doesn't exist
if [ ! -f "stop.sh" ]; then
  cat > stop.sh << 'EOF'
#!/bin/bash
# stop.sh - Stop all components of the trading system

echo "Stopping trading system components..."

# Stop the main trading system
pkill -f "trading_system" || echo "Trading system not running"

# Stop monitoring services
if [ -d "monitoring" ] && [ -f "monitoring/scripts/stop_monitoring.sh" ]; then
  cd monitoring && ./scripts/stop_monitoring.sh
  echo "Monitoring services stopped"
fi

if [ -d "monitoring" ] && [ -f "monitoring/scripts/stop_tunnel.sh" ]; then
  cd monitoring && ./scripts/stop_tunnel.sh
  echo "Tunnels stopped"
fi

# Stop Redis (if not managed by systemd)
if command -v redis-cli >/dev/null 2>&1; then
  if ! systemctl is-active --quiet redis-server 2>/dev/null; then
    redis-cli shutdown || echo "Redis not running"
  fi
fi

# Stop MongoDB (if not managed by systemd)
if command -v mongod >/dev/null 2>&1; then
  if ! systemctl is-active --quiet mongodb 2>/dev/null; then
    mongod --shutdown || echo "MongoDB not running"
  fi
fi

echo "All components stopped"
EOF
  chmod +x stop.sh
  log "Created stop.sh script for shutting down all services"
fi

exit 0