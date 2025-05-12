/**
 * REST API client for market data
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include "trading_system/common/config.h"
#include "trading_system/data/market_data.h"

namespace trading_system {
namespace data {

/**
 * REST API client for fetching market data
 * Supports Polygon.io and other data providers
 */
class RestClient {
public:
    RestClient(const common::Config& config);
    ~RestClient();
    
    // Initialize the client
    void initialize();
    
    // Fetch full market snapshot
    std::future<ParsedMarketData> fetchFullMarketSnapshot();
    
    // Fetch data for specific symbols
    std::future<ParsedMarketData> fetchSymbolData(const std::vector<std::string>& symbols);
    
    // Set thread affinity
    void setThreadAffinity(int core_id);
    
    // Set callback for data updates
    using DataCallback = std::function<void(const ParsedMarketData&)>;
    void setDataCallback(DataCallback callback);
    
    // Start/stop periodic fetching
    void startPeriodicFetching(int interval_ms);
    void stopPeriodicFetching();
    
private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace data
} // namespace trading_system