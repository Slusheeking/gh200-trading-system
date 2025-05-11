/**
 * WebSocket client for market data
 */

#pragma once

#include <memory>
#include "trading_system/common/config.h"
#include "trading_system/data/market_data.h"

namespace trading_system {
namespace data {

class WebSocketClient {
public:
    WebSocketClient(const common::Config& config);
    ~WebSocketClient();
    
    // Connect to WebSocket server
    void connect();
    
    // Disconnect from WebSocket server
    void disconnect();
    
    // Get latest market data
    void getLatestData(MarketData& market_data);
    
    // Set thread affinity for WebSocket thread
    void setThreadAffinity(int core_id);
    
private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace data
} // namespace trading_system