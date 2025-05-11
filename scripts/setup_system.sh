#!/bin/bash
# setup_system.sh - System setup for GH200 Trading System

# Exit on error
set -e

echo "Setting up GH200 Trading System..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root to set up system"
  exit 1
fi

# Install required packages
echo "Installing required packages..."
apt-get update
apt-get install -y build-essential cmake libboost-all-dev libssl-dev \
                   libyaml-cpp-dev libcurl4-openssl-dev libtbb-dev \
                   python3-dev python3-pip nlohmann-json3-dev spdlog-dev

# Install NVIDIA drivers and CUDA if not already installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found, installing..."
    # Add NVIDIA repository
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt-get update
    
    # Install latest drivers
    apt-get install -y nvidia-driver-535
    
    echo "NVIDIA drivers installed, please reboot before continuing"
    exit 0
fi

# Install CUDA if not already installed
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found, installing CUDA 12.0..."
    
    # Download CUDA installer
    wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
    
    # Install CUDA
    sh cuda_12.0.0_525.60.13_linux.run --silent --toolkit
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile.d/cuda.sh
    
    # Clean up
    rm cuda_12.0.0_525.60.13_linux.run
    
    echo "CUDA installed"
fi

# System tuning
echo "Applying system optimizations..."

# CPU isolation (reserve cores 0-1 for system, 2-31 for application)
echo "isolcpus=2-31 nohz_full=2-31 rcu_nocbs=2-31" >> /etc/default/grub
update-grub

# Disable CPU power management
echo "performance" > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Enable huge pages
echo "vm.nr_hugepages = 128" >> /etc/sysctl.conf

# Network optimizations
echo "net.core.rmem_max = 16777216" >> /etc/sysctl.conf
echo "net.core.wmem_max = 16777216" >> /etc/sysctl.conf
echo "net.ipv4.tcp_rmem = 4096 87380 16777216" >> /etc/sysctl.conf
echo "net.ipv4.tcp_wmem = 4096 65536 16777216" >> /etc/sysctl.conf
echo "net.ipv4.tcp_no_metrics_save = 1" >> /etc/sysctl.conf
echo "net.core.netdev_max_backlog = 30000" >> /etc/sysctl.conf
echo "net.ipv4.tcp_low_latency = 1" >> /etc/sysctl.conf

# Apply sysctl changes
sysctl -p

# Disable NUMA balancing
echo "kernel.numa_balancing = 0" >> /etc/sysctl.conf
sysctl -p

# Set up NIC interrupt affinity
for irq in $(grep eth /proc/interrupts | cut -d: -f1); do
    echo 1 > /proc/irq/$irq/smp_affinity
done

# Disable unnecessary services
systemctl disable bluetooth
systemctl disable cups
systemctl disable avahi-daemon
systemctl disable snapd

# Create directories
mkdir -p /opt/trading-system/bin
mkdir -p /opt/trading-system/lib
mkdir -p /opt/trading-system/config
mkdir -p /opt/trading-system/logs
mkdir -p /opt/trading-system/models

# Set up environment variables
cat > /etc/profile.d/trading-system.sh << EOF
export TRADING_SYSTEM_HOME=/opt/trading-system
export LD_LIBRARY_PATH=\$TRADING_SYSTEM_HOME/lib:\$LD_LIBRARY_PATH
export PATH=\$TRADING_SYSTEM_HOME/bin:\$PATH
EOF

# Create trading system user
useradd -m -s /bin/bash trading-user

# Set up credentials directory
mkdir -p /etc/trading-system
chmod 700 /etc/trading-system

# Prompt for API keys
read -p "Polygon API Key: " polygon_key
read -p "Alpaca API Key: " alpaca_key
read -sp "Alpaca API Secret: " alpaca_secret
echo

# Write to secure credentials file
cat > /etc/trading-system/credentials << EOF
POLYGON_API_KEY=$polygon_key
ALPACA_API_KEY=$alpaca_key
ALPACA_API_SECRET=$alpaca_secret
EOF

# Set permissions
chmod 600 /etc/trading-system/credentials
chown trading-user:trading-user /etc/trading-system/credentials

# Create systemd service
cat > /etc/systemd/system/trading-system.service << EOF
[Unit]
Description=GH200 Trading System
After=network.target

[Service]
Type=simple
User=trading-user
Group=trading-user
EnvironmentFile=/etc/trading-system/credentials
ExecStart=/opt/trading-system/bin/trading_system
Restart=on-failure
RestartSec=5
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

echo "System setup complete!"
echo "Please reboot the system to apply all changes."
echo "After reboot, build and install the trading system."