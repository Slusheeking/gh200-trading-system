/**
 * Market data structures
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>

namespace trading_system {
namespace data {

// Trade data
struct Trade {
    std::string symbol;
    double price;
    int64_t size;
    uint64_t timestamp;
    std::string exchange;
    std::string conditions;
};

// Quote data
struct Quote {
    std::string symbol;
    double bid_price;
    double ask_price;
    int64_t bid_size;
    int64_t ask_size;
    uint64_t timestamp;
    std::string exchange;
};

// Bar data
struct Bar {
    std::string symbol;
    double open;
    double high;
    double low;
    double close;
    int64_t volume;
    uint64_t timestamp;
};

// Market data container
class MarketData {
public:
    MarketData();
    ~MarketData();
    
    // Add data
    void addTrade(const Trade& trade);
    void addQuote(const Quote& quote);
    void addBar(const Bar& bar);
    
    // Get data
    const std::vector<Trade>& getTrades() const;
    const std::vector<Quote>& getQuotes() const;
    const std::vector<Bar>& getBars() const;
    
    // Clear data
    void clear();
    
    // Create pre-allocated market data container
    static std::unique_ptr<MarketData> createPreallocated(size_t capacity);
    
private:
    std::vector<Trade> trades_;
    std::vector<Quote> quotes_;
    std::vector<Bar> bars_;
    
    // Pre-allocation
    size_t capacity_;
    bool is_preallocated_;
};

// Parsed market data (output from CUDA parser)
struct ParsedMarketData {
    // Symbol data
    struct SymbolData {
        std::string symbol;
        double last_price;
        double bid_price;
        double ask_price;
        int64_t volume;
        double vwap;
        uint64_t timestamp;
        
        // Technical indicators (calculated by CUDA)
        double rsi_14;
        double macd;
        double macd_signal;
        double macd_histogram;
        double bb_upper;
        double bb_middle;
        double bb_lower;
        double atr;
    };
    
    // Map of symbol to data
    std::unordered_map<std::string, SymbolData> symbol_data;
    
    // Timestamp of parsing
    uint64_t timestamp;
    
    // Number of trades processed
    size_t num_trades_processed;
    
    // Number of quotes processed
    size_t num_quotes_processed;
};

} // namespace data
} // namespace trading_system