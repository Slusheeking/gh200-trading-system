/**
 * Risk management implementation
 */

#include <chrono>
#include <numeric>
#include <algorithm>

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include "trading_system/risk/risk_manager.h"
#include "trading_system/ml/signals.h"

namespace trading_system {
namespace risk {

RiskManager::RiskManager(const common::Config& config)
    : max_position_size_pct_(config.getTradingConfig().risk.max_position_size_pct),
      position_sizing_method_(config.getTradingConfig().risk.position_sizing_method),
      kelly_fraction_(config.getTradingConfig().risk.kelly_fraction),
      max_daily_drawdown_pct_(config.getTradingConfig().risk.max_daily_drawdown_pct),
      max_total_risk_pct_(config.getTradingConfig().risk.max_position_size_pct * 
                         config.getTradingConfig().account.max_positions),
      account_value_(config.getTradingConfig().account.initial_capital),
      daily_pnl_(0.0),
      total_risk_(0.0) {
}

std::vector<ml::Signal> RiskManager::validateSignals(const std::vector<ml::Signal>& signals) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<ml::Signal> validated_signals;
    
    // Skip if no signals
    if (signals.empty()) {
        return validated_signals;
    }
    
    // Process each signal
    for (auto signal : signals) {
        // Calculate position size
        double position_size = calculatePositionSize(signal);
        
        // Skip if position size is too small
        if (position_size <= 0.0) {
            continue;
        }
        
        // Check if signal passes risk checks
        if (passesRiskChecks(signal, position_size)) {
            // Set position size
            signal.position_size = position_size;
            
            // Add to validated signals
            validated_signals.push_back(signal);
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    LOG_INFO("Risk validation completed in " + std::to_string(duration) + 
             " µs, validated " + std::to_string(validated_signals.size()) + 
             " of " + std::to_string(signals.size()) + " signals");
    
    return validated_signals;
}

void RiskManager::setThreadAffinity(int core_id) {
    // Store thread ID
    thread_id_ = std::this_thread::get_id();
    
    // Set affinity
    common::pinThreadToCore(core_id);
}

std::unordered_map<std::string, Position> RiskManager::getPositions() const {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    return positions_;
}

void RiskManager::updatePosition(const std::string& symbol, const Position& position) {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    
    // Update position
    positions_[symbol] = position;
    
    // Update daily P&L
    daily_pnl_ += position.unrealized_pnl - 
                 (positions_.count(symbol) ? positions_[symbol].unrealized_pnl : 0.0);
    
    // Update total risk
    total_risk_ = std::accumulate(
        positions_.begin(), positions_.end(), 0.0,
        [](double sum, const auto& p) {
            return sum + p.second.risk_amount;
        });
}

void RiskManager::removePosition(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    
    // Update daily P&L before removing
    if (positions_.count(symbol)) {
        daily_pnl_ += positions_[symbol].realized_pnl;
    }
    
    // Remove position
    positions_.erase(symbol);
    
    // Update total risk
    total_risk_ = std::accumulate(
        positions_.begin(), positions_.end(), 0.0,
        [](double sum, const auto& p) {
            return sum + p.second.risk_amount;
        });
}

double RiskManager::calculatePositionSize(const ml::Signal& signal) {
    // Use specified position sizing method
    if (position_sizing_method_ == "fixed") {
        // Fixed percentage of account
        return account_value_ * (max_position_size_pct_ / 100.0);
    } else if (position_sizing_method_ == "kelly") {
        // Kelly criterion
        return calculateKellyPositionSize(signal);
    } else if (position_sizing_method_ == "volatility") {
        // Volatility-based sizing
        double atr = signal.indicators.count("atr") ? 
                    signal.indicators.at("atr") : (signal.price * 0.01);
        
        // Size based on ATR
        double risk_per_share = atr * 2.0;  // 2x ATR for stop loss
        double max_risk_amount = account_value_ * (max_position_size_pct_ / 100.0) * 0.01;  // 1% risk
        
        return max_risk_amount / risk_per_share;
    } else {
        // Default to fixed percentage
        return account_value_ * (max_position_size_pct_ / 100.0);
    }
}

bool RiskManager::passesRiskChecks(const ml::Signal& signal, double position_size) {
    // Check daily drawdown
    if (daily_pnl_ < -account_value_ * (max_daily_drawdown_pct_ / 100.0)) {
        LOG_WARNING("Daily drawdown limit reached, rejecting signal for " + signal.symbol);
        return false;
    }
    
    // Check total risk
    double signal_risk = position_size * 0.01;  // Assume 1% risk per trade
    if (total_risk_ + signal_risk > account_value_ * (max_total_risk_pct_ / 100.0)) {
        LOG_WARNING("Total risk limit reached, rejecting signal for " + signal.symbol);
        return false;
    }
    
    // Check existing position
    {
        std::lock_guard<std::mutex> lock(positions_mutex_);
        if (positions_.count(signal.symbol)) {
            LOG_WARNING("Position already exists for " + signal.symbol + ", rejecting signal");
            return false;
        }
    }
    
    // Check portfolio constraints
    if (!checkPortfolioConstraints(signal, position_size)) {
        LOG_WARNING("Portfolio constraints not met, rejecting signal for " + signal.symbol);
        return false;
    }
    
    return true;
}

double RiskManager::calculateKellyPositionSize(const ml::Signal& signal) {
    // Kelly formula: f* = (bp - q) / b
    // where:
    // f* = fraction of bankroll to bet
    // b = odds received on the bet (profit/loss ratio)
    // p = probability of winning
    // q = probability of losing (1 - p)
    
    // Estimate probability of winning from signal confidence
    double p = signal.confidence;
    double q = 1.0 - p;
    
    // Estimate profit/loss ratio from take profit and stop loss
    double take_profit = signal.take_profit > 0.0 ? 
                        signal.take_profit : (signal.price * 1.03);  // Default 3% profit
    double stop_loss = signal.stop_loss > 0.0 ? 
                      signal.stop_loss : (signal.price * 0.98);  // Default 2% loss
    
    double profit = take_profit - signal.price;
    double loss = signal.price - stop_loss;
    
    double b = profit / loss;
    
    // Calculate Kelly fraction
    double f = (b * p - q) / b;
    
    // Apply Kelly fraction and cap at max position size
    f = std::max(0.0, f) * kelly_fraction_;
    f = std::min(f, max_position_size_pct_ / 100.0);
    
    return account_value_ * f;
}

bool RiskManager::checkPortfolioConstraints(const ml::Signal& signal, double position_size) {
    // This is a simplified implementation
    // In a real system, this would check sector exposure, correlation, etc.
    
    // Check maximum number of positions
    std::lock_guard<std::mutex> lock(positions_mutex_);
    if (positions_.size() >= 10) {  // Hardcoded limit for example
        return false;
    }
    
    return true;
}

} // namespace risk
} // namespace trading_system