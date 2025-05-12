#!/bin/bash
# setup_simdjson_lib.sh - Download and set up simdjson library

echo "Setting up simdjson library for GH200 Trading System..."

mkdir -p include/third_party

# Download the single-header version of simdjson
echo "Downloading simdjson library..."
curl -s -o include/third_party/simdjson.h https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.h
curl -s -o include/third_party/simdjson.cpp https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.cpp

if [ $? -eq 0 ]; then
    echo "Successfully downloaded simdjson library to include/third_party/"
else
    echo "Failed to download simdjson library"
    exit 1
fi

# Remove the old nlohmann/json.hpp file
echo "Removing old nlohmann/json.hpp file..."
rm -f include/third_party/json.hpp

# Make the script executable
chmod +x scripts/setup_simdjson_lib.sh

echo "simdjson library setup complete!"
echo "You can now build the trading system with CMake."