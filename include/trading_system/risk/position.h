/**
 * Position data structures
 */

#pragma once

#include <string>
#include <cstdint>

namespace trading_system {
namespace risk {

// Position side
enum class PositionSide {
    LONG,
    SHORT
};

// Position data
struct Position {
    // Basic position info
    std::string symbol;
    double quantity;
    double entry_price;
    double current_price;
    PositionSide side;
    
    // P&L
    double unrealized_pnl;
    double realized_pnl;
    double unrealized_pnl_pct;
    
    // Risk management
    double stop_loss;
    double take_profit;
    double trailing_stop;
    double risk_amount;
    
    // Timestamps
    uint64_t entry_time;
    uint64_t updated_time;
    
    // Order info
    std::string order_id;
    
    // Constructor
    Position()
        : quantity(0.0),
          entry_price(0.0),
          current_price(0.0),
          side(PositionSide::LONG),
          unrealized_pnl(0.0),
          realized_pnl(0.0),
          unrealized_pnl_pct(0.0),
          stop_loss(0.0),
          take_profit(0.0),
          trailing_stop(0.0),
          risk_amount(0.0),
          entry_time(0),
          updated_time(0) {
    }
    
    // Calculate P&L
    void calculatePnL() {
        if (side == PositionSide::LONG) {
            unrealized_pnl = (current_price - entry_price) * quantity;
        } else {
            unrealized_pnl = (entry_price - current_price) * quantity;
        }
        
        unrealized_pnl_pct = entry_price > 0.0 ? 
                            (unrealized_pnl / (entry_price * quantity)) * 100.0 : 0.0;
    }
    
    // Update trailing stop
    void updateTrailingStop() {
        if (side == PositionSide::LONG && trailing_stop > 0.0) {
            // For long positions, trailing stop moves up
            double new_stop = current_price * (1.0 - trailing_stop / 100.0);
            if (new_stop > stop_loss) {
                stop_loss = new_stop;
            }
        } else if (side == PositionSide::SHORT && trailing_stop > 0.0) {
            // For short positions, trailing stop moves down
            double new_stop = current_price * (1.0 + trailing_stop / 100.0);
            if (new_stop < stop_loss || stop_loss == 0.0) {
                stop_loss = new_stop;
            }
        }
    }
    
    // Check if stop loss is triggered
    bool isStopLossTriggered() const {
        if (stop_loss <= 0.0) {
            return false;
        }
        
        if (side == PositionSide::LONG) {
            return current_price <= stop_loss;
        } else {
            return current_price >= stop_loss;
        }
    }
    
    // Check if take profit is triggered
    bool isTakeProfitTriggered() const {
        if (take_profit <= 0.0) {
            return false;
        }
        
        if (side == PositionSide::LONG) {
            return current_price >= take_profit;
        } else {
            return current_price <= take_profit;
        }
    }
};

// Convert position side to string
inline std::string positionSideToString(PositionSide side) {
    switch (side) {
        case PositionSide::LONG: return "long";
        case PositionSide::SHORT: return "short";
        default: return "unknown";
    }
}

// Convert string to position side
inline PositionSide stringToPositionSide(const std::string& str) {
    if (str == "long") {
        return PositionSide::LONG;
    } else if (str == "short") {
        return PositionSide::SHORT;
    } else {
        return PositionSide::LONG;  // Default to long
    }
}

} // namespace risk
} // namespace trading_system