/**
 * Market data implementation
 */

#include "trading_system/data/market_data.h"

namespace trading_system {
namespace data {

MarketData::MarketData()
    : capacity_(0),
      is_preallocated_(false) {
}

MarketData::~MarketData() = default;

void MarketData::addTrade(const Trade& trade) {
    trades_.push_back(trade);
}

void MarketData::addQuote(const Quote& quote) {
    quotes_.push_back(quote);
}

void MarketData::addBar(const Bar& bar) {
    bars_.push_back(bar);
}

const std::vector<Trade>& MarketData::getTrades() const {
    return trades_;
}

const std::vector<Quote>& MarketData::getQuotes() const {
    return quotes_;
}

const std::vector<Bar>& MarketData::getBars() const {
    return bars_;
}

void MarketData::clear() {
    if (is_preallocated_) {
        // Clear without deallocating
        trades_.clear();
        quotes_.clear();
        bars_.clear();
    } else {
        // Clear and deallocate
        std::vector<Trade>().swap(trades_);
        std::vector<Quote>().swap(quotes_);
        std::vector<Bar>().swap(bars_);
    }
}

std::unique_ptr<MarketData> MarketData::createPreallocated(size_t capacity) {
    auto market_data = std::make_unique<MarketData>();
    
    // Set capacity
    market_data->capacity_ = capacity;
    market_data->is_preallocated_ = true;
    
    // Pre-allocate memory
    market_data->trades_.reserve(capacity);
    market_data->quotes_.reserve(capacity);
    market_data->bars_.reserve(capacity);
    
    return market_data;
}

} // namespace data
} // namespace trading_system