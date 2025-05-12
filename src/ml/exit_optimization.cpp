/**
 * Exit optimization model implementation
 */

#include <chrono>
#include <thread>
#include <string>
#include <stdexcept>
#include <iostream>
#include <algorithm>

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include "trading_system/ml/exit_optimization.h"
#include "trading_system/ml/model.h"

namespace trading_system {
namespace ml {

ExitOptimizationModel::ExitOptimizationModel() {
    std::cout << "Creating Exit Optimization Model" << std::endl;
}

ExitOptimizationModel::~ExitOptimizationModel() {
    std::cout << "Destroying Exit Optimization Model" << std::endl;
}

void ExitOptimizationModel::initialize(const common::Config& config) {
    try {
        // Load exit model
        const auto& exit_model_path = config.getMLConfig().model_paths.exit_model;
        if (!exit_model_path.empty()) {
            exit_model_ = std::make_unique<TensorRTModel>();
            exit_model_->load(exit_model_path);
            std::cout << "Loaded exit optimization model: " << exit_model_path << std::endl;
        } else {
            std::cerr << "Exit model path not specified in config" << std::endl;
        }
        
        // Load configuration
        exit_threshold_ = config.getTradingConfig().exit.exit_confidence_threshold;
        max_holding_time_minutes_ = config.getTradingConfig().exit.max_holding_time_minutes;
        check_interval_seconds_ = config.getTradingConfig().exit.check_exit_interval_seconds;
        use_ml_exit_ = config.getTradingConfig().exit.use_ml_exit;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing exit optimization model: " << e.what() << std::endl;
    }
}

std::vector<Signal> ExitOptimizationModel::optimizeExits(
    const std::vector<Signal>& active_positions,
    const data::ParsedMarketData& current_data) {
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Store exit signals
    std::vector<Signal> exit_signals;
    exit_signals.reserve(active_positions.size());
    
    // Process each active position
    for (const auto& position : active_positions) {
        // Check if symbol exists in the data
        auto it = current_data.symbol_data.find(position.symbol);
        if (it == current_data.symbol_data.end()) {
            continue;
        }
        
        // Get current price and timestamp
        double current_price = it->second.last_price;
        uint64_t current_timestamp = current_data.timestamp;
        
        // Check time-based exit
        bool time_exit = shouldExitBasedOnTime(position, current_timestamp);
        
        // Check profit-based exit
        bool profit_exit = shouldExitBasedOnProfit(position, current_price);
        
        // Check stop-loss exit
        bool stop_loss_exit = shouldExitBasedOnStopLoss(position, current_price);
        
        // ML-based exit
        bool ml_exit = false;
        double exit_probability = 0.0;
        
        if (use_ml_exit_ && exit_model_) {
            // Extract features for this position
            std::vector<float> features = extractExitFeatures(position, current_data);
            
            // Run inference
            std::vector<float> outputs = exit_model_->infer(features);
            
            // Skip if no outputs
            if (!outputs.empty()) {
                // Get exit probability
                exit_probability = outputs[0];
                
                // Check if probability exceeds threshold
                ml_exit = exit_probability > exit_threshold_;
            }
        }
        
        // Create exit signal if any exit condition is met
        if (time_exit || profit_exit || stop_loss_exit || ml_exit) {
            Signal exit_signal;
            exit_signal.symbol = position.symbol;
            exit_signal.type = SignalType::EXIT;
            exit_signal.confidence = exit_probability;
            exit_signal.price = current_price;
            exit_signal.timestamp = current_timestamp;
            
            // Set exit reason
            if (time_exit) {
                exit_signal.indicators["exit_reason"] = 1.0;  // Time-based exit
            } else if (profit_exit) {
                exit_signal.indicators["exit_reason"] = 2.0;  // Profit-based exit
            } else if (stop_loss_exit) {
                exit_signal.indicators["exit_reason"] = 3.0;  // Stop-loss exit
            } else if (ml_exit) {
                exit_signal.indicators["exit_reason"] = 4.0;  // ML-based exit
            }
            
            // Add to signals
            exit_signals.push_back(exit_signal);
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    std::cout << "Exit optimization completed in " << duration << " µs, "
              << "generated " << exit_signals.size() << " exit signals from " 
              << active_positions.size() << " active positions" << std::endl;
    
    return exit_signals;
}

void ExitOptimizationModel::setThreadAffinity(int core_id) {
    // Store thread ID
    thread_id_ = std::this_thread::get_id();
    
    // Set affinity
    common::pinThreadToCore(core_id);
}

void ExitOptimizationModel::setExitThreshold(double threshold) {
    exit_threshold_ = threshold;
}

void ExitOptimizationModel::setMaxHoldingTime(int minutes) {
    max_holding_time_minutes_ = minutes;
}

void ExitOptimizationModel::setCheckInterval(int seconds) {
    check_interval_seconds_ = seconds;
}

std::vector<float> ExitOptimizationModel::extractExitFeatures(
    const Signal& position,
    const data::ParsedMarketData& current_data) {
    
    // Find symbol data
    auto it = current_data.symbol_data.find(position.symbol);
    if (it == current_data.symbol_data.end()) {
        return {};
    }
    
    const auto& data = it->second;
    
    // Create feature vector for exit optimization
    std::vector<float> features;
    features.reserve(20);
    
    // Position-specific features
    features.push_back(static_cast<float>(data.last_price / position.price));  // Current P&L
    features.push_back(static_cast<float>((current_data.timestamp - position.timestamp) / 60000000000ULL));  // Duration in minutes
    features.push_back(static_cast<float>(position.confidence));  // Initial confidence
    
    // Market condition changes
    features.push_back(static_cast<float>(data.volatility_change));
    features.push_back(static_cast<float>(data.volume / data.avg_volume));
    features.push_back(static_cast<float>(data.bid_ask_spread_change));
    
    // Technical indicator changes
    features.push_back(static_cast<float>(data.rsi_14));
    features.push_back(static_cast<float>(data.macd));
    features.push_back(static_cast<float>(data.macd_histogram));
    features.push_back(static_cast<float>((data.last_price - data.bb_lower) / (data.bb_upper - data.bb_lower)));
    
    // Add more features as needed
    
    return features;
}

bool ExitOptimizationModel::shouldExitBasedOnTime(
    const Signal& position,
    uint64_t current_timestamp) {
    
    // Calculate position duration in minutes
    uint64_t duration_ns = current_timestamp - position.timestamp;
    uint64_t duration_minutes = duration_ns / 60000000000ULL;  // Convert ns to minutes
    
    // Check if position has exceeded max holding time
    return duration_minutes >= static_cast<uint64_t>(max_holding_time_minutes_);
}

bool ExitOptimizationModel::shouldExitBasedOnProfit(
    const Signal& position,
    double current_price) {
    
    // Calculate profit percentage
    double profit_pct = 0.0;
    
    if (position.type == SignalType::BUY) {
        profit_pct = (current_price - position.price) / position.price * 100.0;
    } else if (position.type == SignalType::SELL) {
        profit_pct = (position.price - current_price) / position.price * 100.0;
    }
    
    // Check if profit exceeds take profit level
    return profit_pct >= position.take_profit;
}

bool ExitOptimizationModel::shouldExitBasedOnStopLoss(
    const Signal& position,
    double current_price) {
    
    // Calculate loss percentage
    double loss_pct = 0.0;
    
    if (position.type == SignalType::BUY) {
        loss_pct = (position.price - current_price) / position.price * 100.0;
    } else if (position.type == SignalType::SELL) {
        loss_pct = (current_price - position.price) / position.price * 100.0;
    }
    
    // Check if loss exceeds stop loss level
    return loss_pct >= position.stop_loss;
}

} // namespace ml
} // namespace trading_system