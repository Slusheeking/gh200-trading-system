/**
 * Polygon.io REST API client implementation
 * Optimized for the hybrid HFT architecture
 */

#include "trading_system/common/logging.h"

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
#include <ta-lib/ta_libc.h>  // Include TA-Lib for technical indicators

#include "trading_system/common/config.h"
#include "trading_system/data/polygon_client.h"

using json = nlohmann::json;

namespace trading_system {
namespace data {

extern common::ZeroAllocLogger g_logger;

// Callback function for CURL
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Implementation class
class PolygonClient::Impl {
public:
    Impl(const common::Config& config)
        : config_(config),
          running_(false) {
        
        // Initialize CURL
        curl_global_init(CURL_GLOBAL_ALL);
        
        // Get API key from config
        const auto& polygon_config = config.getDataSourceConfig("polygon");
        if (polygon_config.enabled) {
            api_key_ = polygon_config.api_key;
            base_url_ = polygon_config.base_url.empty() ? 
                "https://api.polygon.io" : polygon_config.base_url;
        }
    }
    
    ~Impl() {
        stopPeriodicFetching();
        curl_global_cleanup();
    }
    
    bool initialize() {
        std::cout << "Initializing Polygon.io client" << std::endl;
        
        // Check if API key is available
        if (api_key_.empty()) {
            std::cerr << "Error: No API key configured for Polygon.io client" << std::endl;
            return false;
        }
        
        return true;
    }
    
    std::future<ParsedMarketData> fetchFullMarketSnapshot() {
        return std::async(std::launch::async, [this]() {
            ParsedMarketData data;
            
            // Build URL for full market snapshot
            std::string url = base_url_ + "/v2/snapshot/locale/us/markets/stocks/tickers";
            url += "?apiKey=" + api_key_;
            
            // Make request
            std::string response_data;
            makeRequest(url, response_data);
            
            // Parse response
            parseSnapshotResponse(response_data, data);
            
            return data;
        });
    }
    
    std::future<ParsedMarketData> fetchSymbolData(const std::vector<std::string>& symbols) {
        return std::async(std::launch::async, [this, symbols]() {
            ParsedMarketData data;
            
            // Build comma-separated list of symbols
            std::string symbol_list;
            for (size_t i = 0; i < symbols.size(); ++i) {
                symbol_list += symbols[i];
                if (i < symbols.size() - 1) {
                    symbol_list += ",";
                }
            }
            
            // Build URL for symbols snapshot
            std::string url = base_url_ + "/v2/snapshot/locale/us/markets/stocks/tickers";
            url += "?tickers=" + symbol_list;
            url += "&apiKey=" + api_key_;
            
            // Make request
            std::string response_data;
            makeRequest(url, response_data);
            
            // Parse response
            parseSnapshotResponse(response_data, data);
            
            return data;
        });
    }
    
    std::future<ParsedMarketData> fetchAggregatesSnapshot(
        const std::vector<std::string>& symbols,
        const std::string& timespan,
        int multiplier,
        int limit) {
        
        return std::async(std::launch::async, [this, symbols, timespan, multiplier, limit]() {
            ParsedMarketData data;
            
            // Set timestamp
            data.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
            // Process each symbol individually
            for (const auto& symbol : symbols) {
                // Calculate from date (5 days ago)
                auto now = std::chrono::system_clock::now();
                auto five_days_ago = now - std::chrono::hours(24 * 5);
                auto from_time_t = std::chrono::system_clock::to_time_t(five_days_ago);
                auto to_time_t = std::chrono::system_clock::to_time_t(now);
                
                std::stringstream from_ss, to_ss;
                from_ss << std::put_time(std::localtime(&from_time_t), "%Y-%m-%d");
                to_ss << std::put_time(std::localtime(&to_time_t), "%Y-%m-%d");
                
                std::string from_date = from_ss.str();
                std::string to_date = to_ss.str();
                
                // Build URL for aggregates data
                std::string url = base_url_ + "/v2/aggs/ticker/" + symbol + 
                                 "/range/" + std::to_string(multiplier) + "/" + timespan + 
                                 "/" + from_date + "/" + to_date + 
                                 "?adjusted=true&sort=desc&limit=" + std::to_string(limit);
                url += "&apiKey=" + api_key_;
                
                // Make request
                std::string response_data;
                makeRequest(url, response_data);
                
                // Parse aggregate data
                ParsedMarketData::SymbolData symbol_data;
                symbol_data.symbol = symbol;
                
                try {
                    json response_json = json::parse(response_data);
                    
                    // Check status
                    if (response_json["status"] != "OK") {
                        std::cerr << "API error for " << symbol << ": " << response_json["status"].get<std::string>() << std::endl;
                        continue;
                    }
                    
                    // Process results
                    if (response_json.contains("results") && !response_json["results"].is_null() && 
                        response_json["results"].is_array() && !response_json["results"].empty()) {
                        
                        // Most recent bar
                        const auto& latest = response_json["results"][0];
                        symbol_data.open_price = latest["o"].get<double>();
                        symbol_data.high_price = latest["h"].get<double>();
                        symbol_data.low_price = latest["l"].get<double>();
                        symbol_data.last_price = latest["c"].get<double>();
                        symbol_data.volume = latest["v"].get<int64_t>();
                        symbol_data.vwap = latest["vw"].get<double>();
                        symbol_data.timestamp = latest["t"].get<uint64_t>();
                        
                        // Calculate technical indicators
                        calculateTechnicalIndicators(symbol_data, response_json["results"]);
                        
                        // Add to data
                        data.symbol_data[symbol] = symbol_data;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing aggregate data for " << symbol << ": " << e.what() << std::endl;
                }
            }
            
            // Set counts
            data.num_trades_processed = data.symbol_data.size();
            data.num_quotes_processed = data.symbol_data.size();
            
            return data;
        });
    }
    
    std::future<ParsedMarketData> fetchEnhancedSnapshot(const std::vector<std::string>& symbols) {
        return std::async(std::launch::async, [this, symbols]() {
            // First get regular snapshot data
            auto snapshot_future = fetchSymbolData(symbols);
            ParsedMarketData data = snapshot_future.get();
            
            // Then fetch aggregates data to enhance with technical indicators
            auto agg_future = fetchAggregatesSnapshot(symbols, "minute", 5, 30);
            ParsedMarketData agg_data = agg_future.get();
            
            // Merge the data
            for (const auto& pair : agg_data.symbol_data) {
                const std::string& symbol = pair.first;
                const auto& agg_symbol_data = pair.second;
                
                if (data.symbol_data.find(symbol) != data.symbol_data.end()) {
                    // Copy technical indicators from aggregates data
                    auto& symbol_data = data.symbol_data[symbol];
                    symbol_data.rsi_14 = agg_symbol_data.rsi_14;
                    symbol_data.macd = agg_symbol_data.macd;
                    symbol_data.macd_signal = agg_symbol_data.macd_signal;
                    symbol_data.macd_histogram = agg_symbol_data.macd_histogram;
                    symbol_data.bb_upper = agg_symbol_data.bb_upper;
                    symbol_data.bb_middle = agg_symbol_data.bb_middle;
                    symbol_data.bb_lower = agg_symbol_data.bb_lower;
                    symbol_data.atr = agg_symbol_data.atr;
                }
            }
            
            // Now fetch previous day's data for comparison
            for (auto& pair : data.symbol_data) {
                std::string symbol = pair.first;
                auto& symbol_data = pair.second;
                
                // Build URL for previous close
                std::string url = base_url_ + "/v2/aggs/ticker/" + symbol + "/prev";
                url += "?apiKey=" + api_key_;
                
                // Make request
                std::string response_data;
                makeRequest(url, response_data);
                
                try {
                    json response_json = json::parse(response_data);
                    
                    // Check status
                    if (response_json["status"] != "OK") {
                        continue;
                    }
                    
                    // Process results
                    if (response_json.contains("results") && !response_json["results"].is_null()) {
                        const auto& results = response_json["results"];
                        symbol_data.prev_close = results["c"].get<double>();
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing previous day data for " << symbol << ": " << e.what() << std::endl;
                }
            }
            
            return data;
        });
    }
    
    std::future<ParsedMarketData> fetchLatestQuotes(const std::vector<std::string>& symbols) {
        return std::async(std::launch::async, [this, symbols]() {
            ParsedMarketData data;
            
            // Set timestamp
            data.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
            // Process each symbol individually
            for (const auto& symbol : symbols) {
                // Build URL for latest quote
                std::string url = base_url_ + "/v2/last/nbbo/" + symbol;
                url += "?apiKey=" + api_key_;
                
                // Make request
                std::string response_data;
                makeRequest(url, response_data);
                
                // Parse quote data
                ParsedMarketData::SymbolData symbol_data;
                symbol_data.symbol = symbol;
                
                try {
                    json response_json = json::parse(response_data);
                    
                    // Check status
                    if (response_json["status"] != "OK") {
                        std::cerr << "API error for " << symbol << ": " << response_json["status"].get<std::string>() << std::endl;
                        continue;
                    }
                    
                    // Process results
                    if (response_json.contains("results") && !response_json["results"].is_null()) {
                        const auto& quote = response_json["results"];
                        symbol_data.bid_price = quote["p"].get<double>();
                        symbol_data.ask_price = quote["P"].get<double>();
                        symbol_data.bid_ask_spread = symbol_data.ask_price - symbol_data.bid_price;
                        symbol_data.timestamp = quote["t"].get<uint64_t>();
                        
                        // Add to data
                        data.symbol_data[symbol] = symbol_data;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing quote data for " << symbol << ": " << e.what() << std::endl;
                }
            }
            
            // Set counts
            data.num_quotes_processed = data.symbol_data.size();
            
            return data;
        });
    }
    
    std::future<ParsedMarketData> fetchLatestTrades(const std::vector<std::string>& symbols) {
        return std::async(std::launch::async, [this, symbols]() {
            ParsedMarketData data;
            
            // Set timestamp
            data.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
            // Process each symbol individually
            for (const auto& symbol : symbols) {
                // Build URL for latest trade
                std::string url = base_url_ + "/v2/last/trade/" + symbol;
                url += "?apiKey=" + api_key_;
                
                // Make request
                std::string response_data;
                makeRequest(url, response_data);
                
                // Parse trade data
                ParsedMarketData::SymbolData symbol_data;
                symbol_data.symbol = symbol;
                
                try {
                    json response_json = json::parse(response_data);
                    
                    // Check status
                    if (response_json["status"] != "OK") {
                        std::cerr << "API error for " << symbol << ": " << response_json["status"].get<std::string>() << std::endl;
                        continue;
                    }
                    
                    // Process results
                    if (response_json.contains("results") && !response_json["results"].is_null()) {
                        const auto& trade = response_json["results"];
                        symbol_data.last_price = trade["p"].get<double>();
                        symbol_data.timestamp = trade["t"].get<uint64_t>();
                        
                        // Add to data
                        data.symbol_data[symbol] = symbol_data;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing trade data for " << symbol << ": " << e.what() << std::endl;
                }
            }
            
            // Set counts
            data.num_trades_processed = data.symbol_data.size();
            
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
    
    void setDataCallback(PolygonClient::DataCallback callback) {
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
                try {
                    auto data_future = fetchFullMarketSnapshot();
                    ParsedMarketData data = data_future.get();
                    
                    // Call callback if set
                    if (data_callback_ && !data.symbol_data.empty()) {
                        data_callback_(data);
                    }
                } catch (const std::exception& e) {
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
    
    void setApiKey(const std::string& api_key) {
        api_key_ = api_key;
        LOG_WARNING("Setting API key at runtime is not recommended. Use environment variables or config file instead.");
    }
    
private:
    // Make HTTP request using CURL
    void makeRequest(const std::string& url, std::string& response_data) {
        // Set up CURL
        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL");
        }
        
        // Set up request
        response_data.clear();
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        
        // Set timeout
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5);
        
        // Perform request
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            curl_easy_cleanup(curl);
            throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
        }
        
        // Clean up
        curl_easy_cleanup(curl);
    }
    
    // Parse snapshot response
    void parseSnapshotResponse(const std::string& response_data, ParsedMarketData& data) {
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
            ParsedMarketData::SymbolData symbol_data;
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
            if (ticker.contains("day") && !ticker["day"].is_null() && 
                ticker.contains("min") && !ticker["min"].is_null()) {
                // Use minute data for technical indicators
                json fake_results;
                fake_results.push_back({
                    {"o", symbol_data.open_price},
                    {"h", symbol_data.high_price},
                    {"l", symbol_data.low_price},
                    {"c", symbol_data.last_price},
                    {"v", symbol_data.volume},
                    {"vw", symbol_data.vwap},
                    {"t", symbol_data.timestamp}
                });
                
                calculateTechnicalIndicators(symbol_data, fake_results);
            } else {
                // Simplified indicators
                calculateSimplifiedIndicators(symbol_data);
            }
            
            // Add to data
            data.symbol_data[symbol_data.symbol] = symbol_data;
        }
        
        // Set counts
        data.num_trades_processed = data.symbol_data.size();
        data.num_quotes_processed = data.symbol_data.size();
    }
    
    // Calculate technical indicators using TA-Lib
    void calculateTechnicalIndicators(ParsedMarketData::SymbolData& data, const json& results) {
        // Ensure we have enough data points
        if (results.size() < 14) {
            calculateSimplifiedIndicators(data);
            return;
        }
        
        // Initialize TA-Lib
        TA_RetCode retCode = TA_Initialize();
        if (retCode != TA_SUCCESS) {
            std::cerr << "Failed to initialize TA-Lib" << std::endl;
            calculateSimplifiedIndicators(data);
            return;
        }
        
        // Extract price data
        std::vector<double> closes, highs, lows, opens, volumes;
        for (const auto& bar : results) {
            closes.push_back(bar["c"].get<double>());
            highs.push_back(bar["h"].get<double>());
            lows.push_back(bar["l"].get<double>());
            
            if (bar.contains("o")) {
                opens.push_back(bar["o"].get<double>());
            } else {
                opens.push_back(closes.back()); // Default to close if open not available
            }
            
            if (bar.contains("v")) {
                volumes.push_back(static_cast<double>(bar["v"].get<int64_t>()));
            } else {
                volumes.push_back(0.0); // Default to 0 if volume not available
            }
        }
        
        // Reverse so oldest is first (TA-Lib expects time-ordered data)
        std::reverse(closes.begin(), closes.end());
        std::reverse(highs.begin(), highs.end());
        std::reverse(lows.begin(), lows.end());
        std::reverse(opens.begin(), opens.end());
        std::reverse(volumes.begin(), volumes.end());
        
        int dataSize = static_cast<int>(closes.size());
        
        // Calculate RSI-14
        {
            double outReal[dataSize];
            int outBegIdx, outNbElement;
            
            retCode = TA_RSI(0, dataSize-1, closes.data(), 14,
                            &outBegIdx, &outNbElement, outReal);
            
            if (retCode == TA_SUCCESS && outNbElement > 0) {
                data.rsi_14 = outReal[outNbElement-1]; // Get the most recent value
            } else {
                data.rsi_14 = 50.0; // Default value
            }
        }
        
        // Calculate MACD
        {
            double outMACD[dataSize];
            double outMACDSignal[dataSize];
            double outMACDHist[dataSize];
            int outBegIdx, outNbElement;
            
            retCode = TA_MACD(0, dataSize-1, closes.data(), 12, 26, 9,
                             &outBegIdx, &outNbElement,
                             outMACD, outMACDSignal, outMACDHist);
            
            if (retCode == TA_SUCCESS && outNbElement > 0) {
                data.macd = outMACD[outNbElement-1];
                data.macd_signal = outMACDSignal[outNbElement-1];
                data.macd_histogram = outMACDHist[outNbElement-1];
            } else {
                data.macd = 0.0;
                data.macd_signal = 0.0;
                data.macd_histogram = 0.0;
            }
        }
        
        // Calculate Bollinger Bands
        {
            double outUpperBand[dataSize];
            double outMiddleBand[dataSize];
            double outLowerBand[dataSize];
            int outBegIdx, outNbElement;
            
            retCode = TA_BBANDS(0, dataSize-1, closes.data(), 20, 2.0, 2.0, TA_MAType_SMA,
                               &outBegIdx, &outNbElement,
                               outUpperBand, outMiddleBand, outLowerBand);
            
            if (retCode == TA_SUCCESS && outNbElement > 0) {
                data.bb_upper = outUpperBand[outNbElement-1];
                data.bb_middle = outMiddleBand[outNbElement-1];
                data.bb_lower = outLowerBand[outNbElement-1];
            } else {
                data.bb_middle = data.last_price;
                data.bb_upper = data.last_price * 1.02;
                data.bb_lower = data.last_price * 0.98;
            }
        }
        
        // Calculate ATR
        {
            double outReal[dataSize];
            int outBegIdx, outNbElement;
            
            retCode = TA_ATR(0, dataSize-1, highs.data(), lows.data(), closes.data(), 14,
                            &outBegIdx, &outNbElement, outReal);
            
            if (retCode == TA_SUCCESS && outNbElement > 0) {
                data.atr = outReal[outNbElement-1];
            } else {
                data.atr = (data.high_price - data.low_price) * 0.1; // Simple approximation
            }
        }
        
        // Calculate SMA for cross signals
        {
            double outSMA10[dataSize];
            double outSMA20[dataSize];
            int outBegIdx10, outNbElement10;
            int outBegIdx20, outNbElement20;
            
            retCode = TA_SMA(0, dataSize-1, closes.data(), 10,
                            &outBegIdx10, &outNbElement10, outSMA10);
                            
            TA_RetCode retCode2 = TA_SMA(0, dataSize-1, closes.data(), 20,
                                        &outBegIdx20, &outNbElement20, outSMA20);
            
            if (retCode == TA_SUCCESS && retCode2 == TA_SUCCESS &&
                outNbElement10 >= 2 && outNbElement20 >= 2) {
                
                bool current_cross_above = outSMA10[outNbElement10-1] > outSMA20[outNbElement20-1];
                bool prev_cross_above = outSMA10[outNbElement10-2] > outSMA20[outNbElement20-2];
                
                if (current_cross_above && !prev_cross_above) {
                    data.sma_cross_signal = 1.0; // Bullish cross
                } else if (!current_cross_above && prev_cross_above) {
                    data.sma_cross_signal = -1.0; // Bearish cross
                } else {
                    data.sma_cross_signal = 0.0; // No cross
                }
            } else {
                data.sma_cross_signal = 0.0;
            }
        }
        
        // Calculate ADX (Trend Strength)
        {
            double outReal[dataSize];
            int outBegIdx, outNbElement;
            
            retCode = TA_ADX(0, dataSize-1, highs.data(), lows.data(), closes.data(), 14,
                            &outBegIdx, &outNbElement, outReal);
            
            if (retCode == TA_SUCCESS && outNbElement > 0) {
                data.price_trend_strength = outReal[outNbElement-1] / 100.0; // Normalize to [0,1]
            } else {
                data.price_trend_strength = 0.5; // Default to neutral
            }
        }
        
        // Calculate additional indicators specifically for hybrid HFT architecture
        // Price changes
        if (dataSize >= 6) {
            data.price_change_1m = (closes.back() - closes[dataSize-2]) / closes[dataSize-2];
            data.price_change_5m = (closes.back() - closes[dataSize-6]) / closes[dataSize-6];
            data.momentum_1m = closes.back() - closes[dataSize-2];
        } else {
            data.price_change_1m = 0.0;
            data.price_change_5m = 0.0;
            data.momentum_1m = 0.0;
        }
        
        // Volume analysis
        if (!volumes.empty()) {
            double avg_volume = 0;
            for (auto v : volumes) avg_volume += v;
            avg_volume /= volumes.size();
            data.avg_volume = avg_volume;
            
            // Volume trend
            if (volumes.size() >= 2 && closes.size() >= 2) {
                double vol_trend = 0;
                for (size_t i = 1; i < std::min(volumes.size(), closes.size()); i++) {
                    vol_trend += (volumes[i] - volumes[i-1]) * (closes[i] - closes[i-1] > 0 ? 1 : -1);
                }
                data.volume_trend_strength = vol_trend / (volumes.size() - 1);
            } else {
                data.volume_trend_strength = 0.0;
            }
        } else {
            data.avg_volume = data.volume;
            data.volume_trend_strength = 0.0;
        }
        
        // Volatility calculation
        if (dataSize >= 21) {
            double outReal[dataSize];
            int outBegIdx, outNbElement;
            
            retCode = TA_STDDEV(0, dataSize-1, closes.data(), 20, 1.0,
                               &outBegIdx, &outNbElement, outReal);
            
            if (retCode == TA_SUCCESS && outNbElement > 0) {
                data.volatility_ratio = outReal[outNbElement-1] / closes.back() * 100.0; // As percentage of price
            } else {
                data.volatility_ratio = 1.0; // Default value
            }
        } else {
            data.volatility_ratio = 1.0; // Default value
        }
        
        // Cleanup
        TA_Shutdown();
    }
    
    // Calculate simplified technical indicators when there's not enough data
    void calculateSimplifiedIndicators(trading_system::data::ParsedMarketData::SymbolData& data) {
        // Basic technical indicators
        data.rsi_14 = 50.0;
        data.macd = 0.0;
        data.macd_signal = 0.0;
        data.macd_histogram = 0.0;
        data.bb_middle = data.last_price;
        data.bb_upper = data.last_price * 1.02;
        data.bb_lower = data.last_price * 0.98;
        data.atr = (data.high_price - data.low_price) * 0.1;
        
        // Volume metrics
        data.avg_volume = data.volume;
        data.volume_acceleration = 0.0;
        data.volume_spike = 0.0;
        data.volume_profile_imbalance = 0.0;
        
        // Price dynamics
        data.price_change_1m = 0.0;
        data.price_change_5m = 0.0;
        data.momentum_1m = 0.0;
        data.price_trend_strength = 0.5;  // Neutral
        data.volume_trend_strength = 0.0;
        data.volatility_ratio = 1.0;      // Average volatility
        data.volatility_change = 0.0;
        
        // Market context
        data.market_regime = 0.5;         // Neutral
        data.sector_performance = 0.0;
        data.relative_strength = 0.0;
        data.support_resistance_proximity = 0.5;  // Neutral
        data.sma_cross_signal = 0.0;      // No signal
        
        // Order book metrics
        data.bid_ask_imbalance = 0.0;
        data.bid_ask_spread_change = 0.0;
        data.trade_count = 0;
        data.avg_trade_size = 0.0;
        data.large_trade_ratio = 0.0;
    }
    
    // Configuration
    const common::Config& config_;
    
    // API settings
    std::string api_key_;
    std::string base_url_;
    
    // Thread management
    std::thread fetch_thread_;
    int thread_affinity_ = -1;
    std::atomic<bool> running_;
    std::mutex mutex_;
    std::condition_variable cv_;
    int fetch_interval_ms_ = 1000;
    
    // Callback
    PolygonClient::DataCallback data_callback_;
};

// PolygonClient implementation
PolygonClient::PolygonClient(const common::Config& config)
    : impl_(std::make_unique<Impl>(config)) {
}

PolygonClient::~PolygonClient() = default;

bool PolygonClient::initialize() {
    return impl_->initialize();
}

std::future<ParsedMarketData> PolygonClient::fetchFullMarketSnapshot() {
    return impl_->fetchFullMarketSnapshot();
}

std::future<ParsedMarketData> PolygonClient::fetchSymbolData(const std::vector<std::string>& symbols) {
    return impl_->fetchSymbolData(symbols);
}

std::future<ParsedMarketData> PolygonClient::fetchAggregatesSnapshot(
    const std::vector<std::string>& symbols,
    const std::string& timespan,
    int multiplier,
    int limit) {
    return impl_->fetchAggregatesSnapshot(symbols, timespan, multiplier, limit);
}

std::future<ParsedMarketData> PolygonClient::fetchEnhancedSnapshot(const std::vector<std::string>& symbols) {
    return impl_->fetchEnhancedSnapshot(symbols);
}

std::future<ParsedMarketData> PolygonClient::fetchLatestQuotes(const std::vector<std::string>& symbols) {
    return impl_->fetchLatestQuotes(symbols);
}

std::future<ParsedMarketData> PolygonClient::fetchLatestTrades(const std::vector<std::string>& symbols) {
    return impl_->fetchLatestTrades(symbols);
}

void PolygonClient::setThreadAffinity(int core_id) {
    impl_->setThreadAffinity(core_id);
}

void PolygonClient::setDataCallback(DataCallback callback) {
    impl_->setDataCallback(callback);
}

void PolygonClient::startPeriodicFetching(int interval_ms) {
    impl_->startPeriodicFetching(interval_ms);
}

void PolygonClient::stopPeriodicFetching() {
    impl_->stopPeriodicFetching();
}

void PolygonClient::setApiKey(const std::string& api_key) {
    impl_->setApiKey(api_key);
}

} // namespace data
} // namespace trading_system
