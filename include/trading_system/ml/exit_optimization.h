/**
 * Exit optimization model for trading positions
 */

#pragma once

#include <thread>
#include "trading_system/common/config.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "trading_system/ml/model.h"
#include "trading_system/ml/signals.h"
#include "trading_system/data/market_data.h"

namespace trading_system {
namespace ml {

/**
 * Exit optimization model for determining optimal exit points
 * Uses LSTM/GRU network for sequence modeling of price movements
 */
class ExitOptimizationModel {
public:
    ExitOptimizationModel();
    ~ExitOptimizationModel();
    
    // Initialize the model
    void initialize(const common::Config& config);
    
    // Optimize exits for active positions
    std::vector<Signal> optimizeExits(
        const std::vector<Signal>& active_positions,
        const data::ParsedMarketData& current_data
    );
    
    // Set thread affinity
    void setThreadAffinity(int core_id);
    
    // Configuration methods
    void setExitThreshold(double threshold);
    void setMaxHoldingTime(int minutes);
    void setCheckInterval(int seconds);
    
private:
    // ML model
    std::unique_ptr<Model> exit_model_;
    
    // Configuration
    double exit_threshold_ = 0.6;
    int max_holding_time_minutes_ = 240;
    int check_interval_seconds_ = 60;
    bool use_ml_exit_ = true;
    
    // Thread ID for affinity
    std::thread::id thread_id_;
    
    // Extract features for exit optimization
    std::vector<float> extractExitFeatures(
        const Signal& position,
        const data::ParsedMarketData& current_data
    );
    
    // Calculate time-based exit signals
    bool shouldExitBasedOnTime(
        const Signal& position,
        uint64_t current_timestamp
    );
    
    // Calculate profit-based exit signals
    bool shouldExitBasedOnProfit(
        const Signal& position,
        double current_price
    );
    
    // Calculate stop-loss exit signals
    bool shouldExitBasedOnStopLoss(
        const Signal& position,
        double current_price
    );
};

} // namespace ml
} // namespace trading_system