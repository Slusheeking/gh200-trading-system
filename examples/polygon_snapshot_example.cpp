/**
 * Example demonstrating the use of the Polygon.io REST API client
 * for fetching market snapshots
 */

#include <iostream>
#include <vector>
#include <string>
#include <future>
#include <chrono>
#include <thread>
#include <stdlib.h>

#include "trading_system/common/config.h"
#include "trading_system/data/polygon_client.h"

int main(int argc, char* argv[]) {
    // Check for API key
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <polygon_api_key>" << std::endl;
        return 1;
    }
    
    std::string api_key = argv[1];
    
    // Set API key as environment variable
    setenv("POLYGON_API_KEY", api_key.c_str(), 1);
    
    // Create config by reading file
    trading_system::common::Config config("config/system.yaml");
    
    try {
        // Create Polygon client
        trading_system::data::PolygonClient client(config);
        
        // Initialize client
        if (!client.initialize()) {
            std::cerr << "Failed to initialize Polygon client" << std::endl;
            return 1;
        }
        
        // API key is already set via environment variable
        
        std::cout << "Polygon client initialized successfully" << std::endl;
        
        // Define symbols to fetch
        std::vector<std::string> symbols = {"AAPL", "MSFT", "AMZN", "GOOGL", "META"};
        
        // 1. Fetch regular snapshot
        std::cout << "\n1. Fetching regular snapshot..." << std::endl;
        auto snapshot_future = client.fetchSymbolData(symbols);
        auto snapshot = snapshot_future.get();
        
        // Display results
        std::cout << "Received data for " << snapshot.symbol_data.size() << " symbols" << std::endl;
        for (const auto& pair : snapshot.symbol_data) {
            const auto& symbol_data = pair.second;
            std::cout << symbol_data.symbol << ": "
                      << "Last Price: " << symbol_data.last_price << ", "
                      << "Volume: " << symbol_data.volume << ", "
                      << "VWAP: " << symbol_data.vwap << std::endl;
        }
        
        // 2. Fetch enhanced snapshot with technical indicators
        std::cout << "\n2. Fetching enhanced snapshot..." << std::endl;
        auto enhanced_future = client.fetchEnhancedSnapshot(symbols);
        auto enhanced = enhanced_future.get();
        
        // Display results with technical indicators
        std::cout << "Received enhanced data for " << enhanced.symbol_data.size() << " symbols" << std::endl;
        for (const auto& pair : enhanced.symbol_data) {
            const auto& symbol_data = pair.second;
            std::cout << symbol_data.symbol << ": "
                      << "Last Price: " << symbol_data.last_price << ", "
                      << "RSI-14: " << symbol_data.rsi_14 << ", "
                      << "MACD: " << symbol_data.macd << ", "
                      << "BB Width: " << (symbol_data.bb_upper - symbol_data.bb_lower) / symbol_data.bb_middle << std::endl;
        }
        
        // 3. Fetch aggregates snapshot
        std::cout << "\n3. Fetching aggregates snapshot..." << std::endl;
        auto agg_future = client.fetchAggregatesSnapshot(symbols, "minute", 5, 10);  // 5-minute bars, last 10 bars
        auto agg = agg_future.get();
        
        // Display results
        std::cout << "Received aggregate data for " << agg.symbol_data.size() << " symbols" << std::endl;
        for (const auto& pair : agg.symbol_data) {
            const auto& symbol_data = pair.second;
            std::cout << symbol_data.symbol << ": "
                      << "Open: " << symbol_data.open_price << ", "
                      << "High: " << symbol_data.high_price << ", "
                      << "Low: " << symbol_data.low_price << ", "
                      << "Close: " << symbol_data.last_price << ", "
                      << "Volume: " << symbol_data.volume << std::endl;
        }
        
        // 4. Fetch latest quotes
        std::cout << "\n4. Fetching latest quotes..." << std::endl;
        auto quotes_future = client.fetchLatestQuotes(symbols);
        auto quotes = quotes_future.get();
        
        // Display results
        std::cout << "Received quotes for " << quotes.symbol_data.size() << " symbols" << std::endl;
        for (const auto& pair : quotes.symbol_data) {
            const auto& symbol_data = pair.second;
            std::cout << symbol_data.symbol << ": "
                      << "Bid: " << symbol_data.bid_price << ", "
                      << "Ask: " << symbol_data.ask_price << ", "
                      << "Spread: " << symbol_data.bid_ask_spread << std::endl;
        }
        
        // 5. Fetch latest trades
        std::cout << "\n5. Fetching latest trades..." << std::endl;
        auto trades_future = client.fetchLatestTrades(symbols);
        auto trades = trades_future.get();
        
        // Display results
        std::cout << "Received trades for " << trades.symbol_data.size() << " symbols" << std::endl;
        for (const auto& pair : trades.symbol_data) {
            const auto& symbol_data = pair.second;
            std::cout << symbol_data.symbol << ": "
                      << "Price: " << symbol_data.last_price << ", "
                      << "Timestamp: " << symbol_data.timestamp << std::endl;
        }
        
        // 6. Demonstrate periodic fetching
        std::cout << "\n6. Starting periodic fetching for 10 seconds..." << std::endl;
        
        // Define callback
        client.setDataCallback([](const trading_system::data::ParsedMarketData& data) {
            std::cout << "Received periodic update with " << data.symbol_data.size() << " symbols" << std::endl;
            
            // Just display first symbol as example
            if (!data.symbol_data.empty()) {
                auto it = data.symbol_data.begin();
                const auto& symbol_data = it->second;
                std::cout << symbol_data.symbol << ": "
                          << "Last Price: " << symbol_data.last_price << ", "
                          << "Bid: " << symbol_data.bid_price << ", "
                          << "Ask: " << symbol_data.ask_price << std::endl;
            }
        });
        
        // Define periodic fetching interval
        const int periodic_interval_ms = 2000;
        
        // Start periodic fetching
        client.startPeriodicFetching(periodic_interval_ms);  // 2 second interval
        
        // Wait for a while
        std::this_thread::sleep_for(std::chrono::seconds(10));
        
        // Stop fetching
        client.stopPeriodicFetching();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}