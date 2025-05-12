/**
 * REST API client implementation
 */

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <sstream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <ta-lib/ta_libc.h> // Include TA-Lib header

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include "trading_system/data/rest_client.h"

using json = nlohmann::json;

namespace trading_system {
namespace data {

// Callback function for CURL
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Implementation class
class RestClient::Impl {
public:
    Impl(const common::Config& config)
        : config_(config),
          running_(false) {
        
        // Initialize CURL
        curl_global_init(CURL_GLOBAL_ALL);
        
        // Get API keys from config
        const auto& polygon_config = config.getDataSourceConfig("polygon");
        if (polygon_config.enabled) {
            polygon_api_key_ = polygon_config.api_key;
            polygon_base_url_ = polygon_config.base_url.empty() ? 
                "https://api.polygon.io" : polygon_config.base_url;
        }
        
        const auto& alpaca_config = config.getDataSourceConfig("alpaca");
        if (alpaca_config.enabled) {
            alpaca_api_key_ = alpaca_config.api_key;
            alpaca_api_secret_ = alpaca_config.api_secret;
            alpaca_base_url_ = alpaca_config.base_url;
        }
    }
    
    ~Impl() {
        stopPeriodicFetching();
        curl_global_cleanup();
    }
    
    void initialize() {
        std::cout << "Initializing REST client" << std::endl;
        
        // Check if API keys are available
        if (polygon_api_key_.empty() && alpaca_api_key_.empty()) {
            std::cerr << "Warning: No API keys configured for REST client" << std::endl;
        }
    }
    
    std::future<ParsedMarketData> fetchFullMarketSnapshot() {
        return std::async(std::launch::async, [this]() {
            ParsedMarketData data;
            
            // Use Polygon.io API if available
            if (!polygon_api_key_.empty()) {
                fetchPolygonFullMarketSnapshot(data);
            } 
            // Fallback to Alpaca if Polygon is not available
            else if (!alpaca_api_key_.empty()) {
                fetchAlpacaFullMarketSnapshot(data);
            } else {
                std::cerr << "Error: No API keys available for fetching market data" << std::endl;
            }
            
            return data;
        });
    }
    
    std::future<ParsedMarketData> fetchSymbolData(const std::vector<std::string>& symbols) {
        return std::async(std::launch::async, [this, symbols]() {
            ParsedMarketData data;
            
            // Use Polygon.io API if available
            if (!polygon_api_key_.empty()) {
                fetchPolygonSymbolData(symbols, data);
            } 
            // Fallback to Alpaca if Polygon is not available
            else if (!alpaca_api_key_.empty()) {
                fetchAlpacaSymbolData(symbols, data);
            } else {
                std::cerr << "Error: No API keys available for fetching symbol data" << std::endl;
            }
            
            return data;
        });
    }
    
    void setThreadAffinity(int core_id) {
        thread_affinity_ = core_id;
        
        // Apply to running thread if exists
        if (fetch_thread_.joinable()) {
            common::pinThreadToCore(core_id);
        }
    }
    
    void setDataCallback(RestClient::DataCallback callback) {
        data_callback_ = callback;
    }
    
    void startPeriodicFetching(int interval_ms) {
        if (running_) {
            return;
        }
        
        running_ = true;
        fetch_interval_ms_ = interval_ms;
        
        // Start fetch thread
        fetch_thread_ = std::thread([this]() {
            // Set thread affinity if specified
            if (thread_affinity_ >= 0) {
                common::pinThreadToCore(thread_affinity_);
            }
            
            while (running_) {
                // Fetch data
                ParsedMarketData data;
                
                try {
                    // Use Polygon.io API if available
                    if (!polygon_api_key_.empty()) {
                        fetchPolygonFullMarketSnapshot(data);
                    } 
                    // Fallback to Alpaca if Polygon is not available
                    else if (!alpaca_api_key_.empty()) {
                        fetchAlpacaFullMarketSnapshot(data);
                    }
                    
                    // Call callback if set
                    if (data_callback_ && !data.symbol_data.empty()) {
                        data_callback_(data);
                    }
                } catch (const std::exception&amp; e) {
                    std::cerr << "Error fetching market data: " << e.what() << std::endl;
                }
                
                // Wait for next interval
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(fetch_interval_ms_), 
                            [this]() { return !running_; });
            }
        });
    }
    
    void stopPeriodicFetching() {
        if (!running_) {
            return;
        }
        
        running_ = false;
        cv_.notify_all();
        
        if (fetch_thread_.joinable()) {
            fetch_thread_.join();
        }
    }
    
private:
    // Configuration
    const common::Config& config_;
    
    // API keys
    std::string polygon_api_key_;
    std::string polygon_base_url_;
    std::string alpaca_api_key_;
    std::string alpaca_api_secret_;
    std::string alpaca_base_url_;
    
    // Thread management
    std::thread fetch_thread_;
    int thread_affinity_ = -1;
    std::atomic<bool> running_;
    std::mutex mutex_;
    std::condition_variable cv_;
    int fetch_interval_ms_ = 1000;
    
    // Callback
    RestClient::DataCallback data_callback_;
    
    // Fetch data from Polygon.io
    void fetchPolygonFullMarketSnapshot(ParsedMarketData& data) {
        // Set up CURL
        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL");
        }
        
        // Build URL
        std::string url = polygon_base_url_ + "/v2/snapshot/locale/us/markets/stocks/tickers";
        url += "?apiKey=" + polygon_api_key_;
        
        // Set up request
        std::string response_data;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        
        // Perform request
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            curl_easy_cleanup(curl);
            throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
        }
        
        // Clean up
        curl_easy_cleanup(curl);
        
        // Parse JSON response
        json response_json = json::parse(response_data);
        
        // Check status
        if (response_json["status"] != "OK") {
            throw std::runtime_error("API error: " + response_json["status"].get<std::string>());
        }
        
        // Set timestamp
        data.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        // Process tickers
        for (const auto& ticker : response_json["tickers"]) {
            // Create symbol data
            trading_system::data::ParsedMarketData::SymbolData symbol_data;
            symbol_data.symbol = ticker["ticker"].get<std::string>();
            
            // Set price data
            if (ticker.contains("lastTrade") && !ticker["lastTrade"].is_null()) {
                symbol_data.last_price = ticker["lastTrade"]["p"].get<double>();
                symbol_data.timestamp = ticker["lastTrade"]["t"].get<uint64_t>();
            }
            
            if (ticker.contains("lastQuote") && !ticker["lastQuote"].is_null()) {
                symbol_data.bid_price = ticker["lastQuote"]["p"].get<double>();
                symbol_data.ask_price = ticker["lastQuote"]["P"].get<double>();
                symbol_data.bid_ask_spread = symbol_data.ask_price - symbol_data.bid_price;
            }
            
            // Set day data
            if (ticker.contains("day") && !ticker["day"].is_null()) {
                symbol_data.open_price = ticker["day"]["o"].get<double>();
                symbol_data.high_price = ticker["day"]["h"].get<double>();
                symbol_data.low_price = ticker["day"]["l"].get<double>();
                symbol_data.volume = ticker["day"]["v"].get<int64_t>();
                symbol_data.vwap = ticker["day"]["vw"].get<double>();
            }
            
            // Set previous day data
            if (ticker.contains("prevDay") && !ticker["prevDay"].is_null()) {
                symbol_data.prev_close = ticker["prevDay"]["c"].get<double>();
            }
            
            // Calculate technical indicators (simplified)
            calculateTechnicalIndicators(symbol_data);
            
            // Add to data
            data.symbol_data[symbol_data.symbol] = symbol_data;
        }
        
        // Set counts
        data.num_trades_processed = data.symbol_data.size();
        data.num_quotes_processed = data.symbol_data.size();
        
        std::cout << "Fetched data for " << data.symbol_data.size() << " symbols" << std::endl;
    }
    
    // Fetch data for specific symbols from Polygon.io
    void fetchPolygonSymbolData(const std::vector<std::string>& symbols, ParsedMarketData& data) {
        // Build comma-separated list of symbols
        std::string symbol_list;
        for (size_t i = 0; i < symbols.size(); ++i) {
            symbol_list += symbols[i];
            if (i < symbols.size() - 1) {
                symbol_list += ",";
            }
        }
        
        // Set up CURL
        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL");
        }
        
        // Build URL
        std::string url = polygon_base_url_ + "/v2/snapshot/locale/us/markets/stocks/tickers";
        url += "?tickers=" + symbol_list;
        url += "&apiKey=" + polygon_api_key_;
        
        // Set up request
        std::string response_data;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        
        // Perform request
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            curl_easy_cleanup(curl);
            throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
        }
        
        // Clean up
        curl_easy_cleanup(curl);
        
        // Parse JSON response
        json response_json = json::parse(response_data);
        
        // Check status
        if (response_json["status"] != "OK") {
            throw std::runtime_error("API error: " + response_json["status"].get<std::string>());
        }
        
        // Set timestamp
        data.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        // Process tickers (same as fetchPolygonFullMarketSnapshot)
        for (const auto& ticker : response_json["tickers"]) {
            // Create symbol data
            trading_system::data::ParsedMarketData::SymbolData symbol_data;
            symbol_data.symbol = ticker["ticker"].get<std::string>();
            
            // Set price data
            if (ticker.contains("lastTrade") && !ticker["lastTrade"].is_null()) {
                symbol_data.last_price = ticker["lastTrade"]["p"].get<double>();
                symbol_data.timestamp = ticker["lastTrade"]["t"].get<uint64_t>();
            }
            
            if (ticker.contains("lastQuote") && !ticker["lastQuote"].is_null()) {
                symbol_data.bid_price = ticker["lastQuote"]["p"].get<double>();
                symbol_data.ask_price = ticker["lastQuote"]["P"].get<double>();
                symbol_data.bid_ask_spread = symbol_data.ask_price - symbol_data.bid_price;
            }
            
            // Set day data
            if (ticker.contains("day") && !ticker["day"].is_null()) {
                symbol_data.open_price = ticker["day"]["o"].get<double>();
                symbol_data.high_price = ticker["day"]["h"].get<double>();
                symbol_data.low_price = ticker["day"]["l"].get<double>();
                symbol_data.volume = ticker["day"]["v"].get<int64_t>();
                symbol_data.vwap = ticker["day"]["vw"].get<double>();
            }
            
            // Set previous day data
            if (ticker.contains("prevDay") && !ticker["prevDay"].is_null()) {
                symbol_data.prev_close = ticker["prevDay"]["c"].get<double>();
            }
            
            // Calculate technical indicators (simplified)
            calculateTechnicalIndicators(symbol_data);
            
            // Add to data
            data.symbol_data[symbol_data.symbol] = symbol_data;
        }
        
        // Set counts
        data.num_trades_processed = data.symbol_data.size();
        data.num_quotes_processed = data.symbol_data.size();
        
        std::cout << "Fetched data for " << data.symbol_data.size() << " symbols" << std::endl;
    }
    
    // Fetch data from Alpaca
    void fetchAlpacaFullMarketSnapshot(ParsedMarketData& data) {
        if (alpaca_api_key_.empty() || alpaca_api_secret_.empty() || alpaca_base_url_.empty()) {
            std::cerr << "Alpaca API keys or base URL not configured." << std::endl;
            return;
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL for Alpaca");
        }

        std::string url = alpaca_base_url_ + "/v2/stocks/snapshots"; // Example endpoint, adjust as needed
        // Note: Alpaca API for full market snapshot might require iterating through symbols or using a different endpoint.
        // This is a simplified example. A real implementation would handle pagination and symbol lists.

        std::string response_data;
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, ("APCA-API-KEY-ID: " + alpaca_api_key_).c_str());
        headers = curl_slist_append(headers, ("APCA-API-SECRET-KEY: " + alpaca_api_secret_).c_str());

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            throw std::runtime_error("CURL request to Alpaca failed: " + std::string(curl_easy_strerror(res)));
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        // Parse JSON response and populate data
        try {
            json response_json = json::parse(response_data);
            // Assuming the response is a map of symbol -> snapshot data
            for (auto const& [symbol, snapshot_json] : response_json.items()) {
                 if (snapshot_json.contains("latestTrade") && !snapshot_json["latestTrade"].is_null()) {
                    trading_system::data::ParsedMarketData::SymbolData symbol_data;
                    symbol_data.symbol = symbol;
                    symbol_data.last_price = snapshot_json["latestTrade"]["p"].get<double>();
                    symbol_data.timestamp = snapshot_json["latestTrade"]["t"].get<uint64_t>(); // Check Alpaca timestamp format

                    if (snapshot_json.contains("latestQuote") && !snapshot_json["latestQuote"].is_null()) {
                        symbol_data.bid_price = snapshot_json["latestQuote"]["bp"].get<double>();
                        symbol_data.ask_price = snapshot_json["latestQuote"]["ap"].get<double>();
                        symbol_data.bid_ask_spread = symbol_data.ask_price - symbol_data.bid_price;
                    }

                    if (snapshot_json.contains("dailyBar") && !snapshot_json["dailyBar"].is_null()) {
                        symbol_data.open_price = snapshot_json["dailyBar"]["o"].get<double>();
                        symbol_data.high_price = snapshot_json["dailyBar"]["h"].get<double>();
                        symbol_data.low_price = snapshot_json["dailyBar"]["l"].get<double>();
                        symbol_data.volume = snapshot_json["dailyBar"]["v"].get<int64_t>();
                        symbol_data.vwap = snapshot_json["dailyBar"]["vwap"].get<double>();
                    }

                    if (snapshot_json.contains("prevDailyBar") && !snapshot_json["prevDailyBar"].is_null()) {
                         symbol_data.prev_close = snapshot_json["prevDailyBar"]["c"].get<double>();
                    }

                    calculateTechnicalIndicators(symbol_data); // Keep existing technical indicator calculation
                    data.symbol_data[symbol_data.symbol] = symbol_data;
                 }
            }
             data.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
             data.num_trades_processed = data.symbol_data.size();
             data.num_quotes_processed = data.symbol_data.size();
             std::cout << "Fetched data for " << data.symbol_data.size() << " symbols from Alpaca" << std::endl;

        } catch (const json::exception& e) {
            std::cerr << "JSON parsing error for Alpaca snapshot: " << e.what() << std::endl;
        } catch (const std::exception&amp; e) {
            std::cerr << "Error processing Alpaca snapshot data: " << e.what() << std::endl;
        }
    }
    
    // Fetch data for specific symbols from Alpaca
    void fetchAlpacaSymbolData(const std::vector<std::string>& symbols, ParsedMarketData& data) {
        if (alpaca_api_key_.empty() || alpaca_api_secret_.empty() || alpaca_base_url_.empty()) {
            std::cerr << "Alpaca API keys or base URL not configured." << std::endl;
            return;
        }

        if (symbols.empty()) {
            return; // Nothing to fetch
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL for Alpaca symbol data");
        }

        std::string symbol_list;
        for (size_t i = 0; i < symbols.size(); ++i) {
            symbol_list += symbols[i];
            if (i < symbols.size() - 1) {
                symbol_list += ",";
            }
        }

        std::string url = alpaca_base_url_ + "/v2/stocks/snapshots?symbols=" + symbol_list; // Endpoint for specific symbols

        std::string response_data;
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, ("APCA-API-KEY-ID: " + alpaca_api_key_).c_str());
        headers = curl_slist_append(headers, ("APCA-API-SECRET-KEY: " + alpaca_api_secret_).c_str());

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            throw std::runtime_error("CURL request to Alpaca failed: " + std::string(curl_easy_strerror(res)));
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        // Parse JSON response and populate data (similar to fetchAlpacaFullMarketSnapshot)
        try {
            json response_json = json::parse(response_data);
             for (auto const& [symbol, snapshot_json] : response_json.items()) {
                 if (snapshot_json.contains("latestTrade") && !snapshot_json["latestTrade"].is_null()) {
                    trading_system::data::ParsedMarketData::SymbolData symbol_data;
                    symbol_data.symbol = symbol;
                    symbol_data.last_price = snapshot_json["latestTrade"]["p"].get<double>();
                    symbol_data.timestamp = snapshot_json["latestTrade"]["t"].get<uint64_t>(); // Check Alpaca timestamp format

                    if (snapshot_json.contains("latestQuote") && !snapshot_json["latestQuote"].is_null()) {
                        symbol_data.bid_price = snapshot_json["latestQuote"]["bp"].get<double>();
                        symbol_data.ask_price = snapshot_json["latestQuote"]["ap"].get<double>();
                        symbol_data.bid_ask_spread = symbol_data.ask_price - symbol_data.bid_price;
                    }

                    if (snapshot_json.contains("dailyBar") && !snapshot_json["dailyBar"].is_null()) {
                        symbol_data.open_price = snapshot_json["dailyBar"]["o"].get<double>();
                        symbol_data.high_price = snapshot_json["dailyBar"]["h"].get<double>();
                        symbol_data.low_price = snapshot_json["dailyBar"]["l"].get<double>();
                        symbol_data.volume = snapshot_json["dailyBar"]["v"].get<int64_t>();
                        symbol_data.vwap = snapshot_json["dailyBar"]["vwap"].get<double>();
                    }

                    if (snapshot_json.contains("prevDailyBar") && !snapshot_json["prevDailyBar"].is_null()) {
                         symbol_data.prev_close = snapshot_json["prevDailyBar"]["c"].get<double>();
                    }

                    calculateTechnicalIndicators(symbol_data); // Keep existing technical indicator calculation
                    data.symbol_data[symbol_data.symbol] = symbol_data;
                 }
            }
             data.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
             data.num_trades_processed = data.symbol_data.size();
             data.num_quotes_processed = data.symbol_data.size();
             std::cout << "Fetched data for " << data.symbol_data.size() << " symbols from Alpaca" << std::endl;

        } catch (const json::exception& e) {
            std::cerr << "JSON parsing error for Alpaca symbol data: " << e.what() << std::endl;
        } catch (const std::exception&amp; e) {
            std::cerr << "Error processing Alpaca symbol data: " << e.what() << std::endl;
        }
    }
    
    // Calculate technical indicators
    void calculateTechnicalIndicators(trading_system::data::ParsedMarketData::SymbolData& data) {
        // This is a simplified implementation
        // In a real system, these would be calculated using proper TA libraries
        // This function assumes historical data is available for the symbol.
        // A real implementation would need to manage historical data storage and retrieval.

        // Placeholder for historical data arrays (replace with actual data)
        // const int history_size = 100; // Example history size
        // double inHigh[history_size];
        // double inLow[history_size];
        // double inClose[history_size];
        // double inVolume[history_size]; // For volume-based indicators

        // TODO: Populate inHigh, inLow, inClose, inVolume with historical data for data.symbol

        // Example TA-Lib usage (replace with actual calls and parameters)
        // Removed unused variables

        // RSI
        // double outRsi[history_size];
        // TA_RetCode retCodeRsi = TA_RSI(0, history_size - 1, inClose, 14, &outBegIdx, &outNbElement, outRsi);
        // if (retCodeRsi == TA_SUCCESS && outNbElement > 0) {
        //     data.rsi_14 = outRsi[outNbElement - 1]; // Get the latest RSI value
        // } else {
            data.rsi_14 = 50.0;  // Placeholder or error value
        // }

        // MACD
        // double outMacd[history_size], outMacdSignal[history_size], outMacdHist[history_size];
        // TA_RetCode retCodeMacd = TA_MACD(0, history_size - 1, inClose, 12, 26, 9, &outBegIdx, &outNbElement, outMacd, outMacdSignal, outMacdHist);
        // if (retCodeMacd == TA_SUCCESS && outNbElement > 0) {
        //     data.macd = outMacd[outNbElement - 1];
        //     data.macd_signal = outMacdSignal[outNbElement - 1];
        //     data.macd_histogram = outMacdHist[outNbElement - 1];
        // } else {
            data.macd = 0.0;  // Placeholder or error value
            data.macd_signal = 0.0;  // Placeholder or error value
            data.macd_histogram = 0.0;  // Placeholder or error value
        // }

        // Bollinger Bands
        // double outBbUpper[history_size], outBbMiddle[history_size], outBbLower[history_size];
        // TA_RetCode retCodeBb = TA_BBANDS(0, history_size - 1, inClose, 20, 2.0, 2.0, TA_MAType_SMA, &outBegIdx, &outNbElement, outBbUpper, outBbMiddle, outBbLower);
        // if (retCodeBb == TA_SUCCESS && outNbElement > 0) {
        //     data.bb_upper = outBbUpper[outNbElement - 1];
        //     data.bb_middle = outBbMiddle[outNbElement - 1];
        //     data.bb_lower = outBbLower[outNbElement - 1];
        // } else {
            data.bb_middle = data.last_price; // Placeholder or error value
            data.bb_upper = data.last_price * 1.02; // Placeholder or error value
            data.bb_lower = data.last_price * 0.98; // Placeholder or error value
        // }

        // ATR
        // double outAtr[history_size];
        // TA_RetCode retCodeAtr = TA_ATR(0, history_size - 1, inHigh, inLow, inClose, 14, &outBegIdx, &outNbElement, outAtr);
        // if (retCodeAtr == TA_SUCCESS && outNbElement > 0) {
        //     data.atr = outAtr[outNbElement - 1];
        // } else {
            data.atr = (data.high_price - data.low_price) * 0.1;  // Placeholder or error value
        // }

        // Additional indicators for fast path (placeholders - implement with TA-Lib or other logic)
        data.avg_volume = data.volume;  // Placeholder
        data.volume_acceleration = 0.0;  // Placeholder
        data.volume_spike = 0.0;  // Placeholder
        data.volume_profile_imbalance = 0.0;  // Placeholder

        // Price dynamics (placeholders - implement with appropriate calculations)
        data.price_change_1m = 0.0;  // Placeholder
        data.price_change_5m = 0.0;  // Placeholder
        data.momentum_1m = 0.0;  // Placeholder
        data.price_trend_strength = 0.0;  // Placeholder
        data.volume_trend_strength = 0.0;  // Placeholder
        data.volatility_ratio = 0.0;  // Placeholder
        data.volatility_change = 0.0;  // Placeholder

        // Market context (placeholders - implement with appropriate calculations)
        data.market_regime = 0.0;  // Placeholder
        data.sector_performance = 0.0;  // Placeholder
        data.relative_strength = 0.0;  // Placeholder
        data.support_resistance_proximity = 0.0;  // Placeholder
        data.sma_cross_signal = 0.0;  // Placeholder

        // Order book metrics (placeholders - implement with appropriate calculations)
        data.bid_ask_imbalance = 0.0;  // Placeholder
        data.bid_ask_spread_change = 0.0;  // Placeholder
        data.trade_count = 0;  // Placeholder
        data.avg_trade_size = 0.0;  // Placeholder
        data.large_trade_ratio = 0.0;  // Placeholder
    }
};

// RestClient implementation
RestClient::RestClient(const common::Config& config)
    : impl_(std::make_unique<Impl>(config)) {
}

RestClient::~RestClient() = default;

void RestClient::initialize() {
    impl_->initialize();
}

std::future<ParsedMarketData> RestClient::fetchFullMarketSnapshot() {
    return impl_->fetchFullMarketSnapshot();
}

std::future<ParsedMarketData> RestClient::fetchSymbolData(const std::vector<std::string>& symbols) {
    return impl_->fetchSymbolData(symbols);
}

void RestClient::setThreadAffinity(int core_id) {
    impl_->setThreadAffinity(core_id);
}

void RestClient::setDataCallback(DataCallback callback) {
    impl_->setDataCallback(callback);
}

void RestClient::startPeriodicFetching(int interval_ms) {
    impl_->startPeriodicFetching(interval_ms);
}

void RestClient::stopPeriodicFetching() {
    impl_->stopPeriodicFetching();
}

} // namespace data
} // namespace trading_system