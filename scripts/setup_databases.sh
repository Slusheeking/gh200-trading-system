#!/bin/bash
# setup_databases.sh - Install and configure Redis and MongoDB for GH200 Trading System

# Exit on error
set -e

echo "Setting up databases for GH200 Trading System..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root to set up databases"
  exit 1
fi

# Install Redis
echo "Installing Redis..."
apt-get update
apt-get install -y redis-server

# Configure Redis for better performance
echo "Configuring Redis..."
cat > /etc/redis/redis.conf.new << EOF
bind 127.0.0.1
port 6379
daemonize yes
supervised systemd
maxmemory 1gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
EOF

# Backup original config and apply new one
cp /etc/redis/redis.conf /etc/redis/redis.conf.bak
mv /etc/redis/redis.conf.new /etc/redis/redis.conf

# Restart Redis to apply changes
systemctl restart redis-server
systemctl enable redis-server

echo "Redis installed and configured."

# Install MongoDB
echo "Installing MongoDB..."
apt-get install -y gnupg
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
apt-get update
apt-get install -y mongodb-org

# Configure MongoDB for better performance
echo "Configuring MongoDB..."
cat > /etc/mongod.conf.new << EOF
storage:
  dbPath: /var/lib/mongodb
  journal:
    enabled: true
  wiredTiger:
    engineConfig:
      cacheSizeGB: 2

systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log

net:
  port: 27017
  bindIp: 127.0.0.1

processManagement:
  timeZoneInfo: /usr/share/zoneinfo
EOF

# Backup original config and apply new one
cp /etc/mongod.conf /etc/mongod.conf.bak
mv /etc/mongod.conf.new /etc/mongod.conf

# Start MongoDB and enable on boot
systemctl start mongod
systemctl enable mongod

echo "MongoDB installed and configured."

# Create trading system database and user
echo "Setting up MongoDB database and user for trading system..."
mongosh --eval "
  db = db.getSiblingDB('trading_system');
  db.createUser({
    user: 'trading_user',
    pwd: 'trading_password',
    roles: [{ role: 'readWrite', db: 'trading_system' }]
  });
  db.createCollection('metrics');
  db.createCollection('trades');
  db.createCollection('positions');
"

# Add database credentials to environment
echo "Adding database credentials to environment..."
cat >> /etc/trading-system/credentials << EOF
REDIS_HOST=localhost
REDIS_PORT=6379
MONGODB_URI=mongodb://trading_user:trading_password@localhost:27017/trading_system
EOF

echo "Database setup complete!"
echo "Redis is running on port 6379"
echo "MongoDB is running on port 27017"
echo "MongoDB credentials have been added to /etc/trading-system/credentials"