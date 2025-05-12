/**
 * Polygon.io REST API client for market snapshots
 * Optimized for the hybrid HFT architecture
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
 * Polygon.io REST API client specifically designed for market snapshots
 * Optimized for the hybrid HFT architecture
 */
class PolygonClient {
public:
    PolygonClient(const common::Config& config);
    ~PolygonClient();
    
    // Initialize the client
    bool initialize();
    
    // Fetch full market snapshot for all tickers
    std::future<ParsedMarketData> fetchFullMarketSnapshot();
    
    // Fetch data for specific symbols
    std::future<ParsedMarketData> fetchSymbolData(const std::vector<std::string>& symbols);
    
    // Fetch snapshot with aggregates (OHLCV) data for market analysis
    std::future<ParsedMarketData> fetchAggregatesSnapshot(
        const std::vector<std::string>& symbols,
        const std::string& timespan = "minute",
        int multiplier = 1,
        int limit = 5
    );
    
    // Fetch snapshot with technical indicators precomputed
    std::future<ParsedMarketData> fetchEnhancedSnapshot(const std::vector<std::string>& symbols);
    
    // Fetch latest quotes for symbols
    std::future<ParsedMarketData> fetchLatestQuotes(const std::vector<std::string>& symbols);
    
    // Fetch latest trades for symbols
    std::future<ParsedMarketData> fetchLatestTrades(const std::vector<std::string>& symbols);
    
    // Set thread affinity
    void setThreadAffinity(int core_id);
    
    // Set callback for data updates
    using DataCallback = std::function<void(const ParsedMarketData&)>;
    void setDataCallback(DataCallback callback);
    
    // Start/stop periodic fetching
    void startPeriodicFetching(int interval_ms);
    void stopPeriodicFetching();
    
    // Set API key (if not provided in config)
    void setApiKey(const std::string& api_key);
    
private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace data
} // namespace trading_system