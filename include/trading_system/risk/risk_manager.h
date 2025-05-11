/**
 * Risk management for trading decisions
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include "trading_system/common/config.h"
#include "trading_system/ml/signals.h"
#include "trading_system/risk/position.h"

namespace trading_system {
namespace risk {

class RiskManager {
public:
    RiskManager(const common::Config& config);
    ~RiskManager() = default;
    
    // Validate trading signals based on risk rules
    std::vector<ml::Signal> validateSignals(const std::vector<ml::Signal>& signals);
    
    // Set thread affinity
    void setThreadAffinity(int core_id);
    
    // Get current positions
    std::unordered_map<std::string, Position> getPositions() const;
    
    // Update position with execution data
    void updatePosition(const std::string& symbol, const Position& position);
    
    // Remove position
    void removePosition(const std::string& symbol);
    
private:
    // Risk configuration
    double max_position_size_pct_;
    std::string position_sizing_method_;
    double kelly_fraction_;
    double max_daily_drawdown_pct_;
    double max_total_risk_pct_;
    
    // Account state
    double account_value_;
    double daily_pnl_;
    double total_risk_;
    
    // Current positions
    std::unordered_map<std::string, Position> positions_;
    mutable std::mutex positions_mutex_;
    
    // Thread ID for affinity
    std::thread::id thread_id_;
    
    // Calculate position size based on risk
    double calculatePositionSize(const ml::Signal& signal);
    
    // Check if signal passes risk checks
    bool passesRiskChecks(const ml::Signal& signal, double position_size);
    
    // Calculate Kelly position size
    double calculateKellyPositionSize(const ml::Signal& signal);
    
    // Check portfolio constraints
    bool checkPortfolioConstraints(const ml::Signal& signal, double position_size);
};

} // namespace risk
} // namespace trading_system