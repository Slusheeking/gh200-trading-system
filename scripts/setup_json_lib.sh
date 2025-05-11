#!/bin/bash
# setup_json_lib.sh - Download and set up nlohmann/json library

echo "Setting up nlohmann/json library for GH200 Trading System..."

# Create third-party directory if it doesn't exist
mkdir -p include/third_party

# Download the single-header version of nlohmann/json
echo "Downloading nlohmann/json library..."
curl -s -o include/third_party/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Successfully downloaded nlohmann/json library to include/third_party/json.hpp"
else
    echo "Failed to download nlohmann/json library"
    exit 1
fi

# Update the metrics_publisher.cpp file to use the local copy
echo "Updating metrics_publisher.cpp to use the local copy..."
sed -i 's|#include <nlohmann/json.hpp>|#include "../../include/third_party/json.hpp"|' src/monitoring/metrics_publisher.cpp

# Check if update was successful
if [ $? -eq 0 ]; then
    echo "Successfully updated metrics_publisher.cpp"
else
    echo "Failed to update metrics_publisher.cpp"
    exit 1
fi

echo "nlohmann/json library setup complete!"
echo "You can now build the trading system with CMake."