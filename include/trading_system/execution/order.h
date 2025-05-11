/**
 * Order definitions
 */

#pragma once

#include <string>
#include <cstdint>

namespace trading_system {
namespace execution {

// Order side
enum class OrderSide {
    BUY,
    SELL
};

// Order type
enum class OrderType {
    MARKET,
    LIMIT,
    STOP,
    STOP_LIMIT,
    TRAILING_STOP
};

// Order class
enum class OrderClass {
    SIMPLE,
    BRACKET,
    OCO,
    OTO
};

// Time in force
enum class TimeInForce {
    DAY,
    GTC,
    IOC,
    FOK
};

// Order status
enum class OrderStatus {
    NEW,
    PARTIALLY_FILLED,
    FILLED,
    CANCELED,
    REJECTED,
    EXPIRED
};

// Order structure
struct Order {
    // Basic order info
    std::string symbol;
    OrderSide side;
    OrderType type;
    OrderClass order_class;
    TimeInForce time_in_force;
    
    // Quantity and price
    double quantity;
    double limit_price;
    double stop_price;
    
    // For bracket orders
    double take_profit_price;
    double stop_loss_price;
    double stop_loss_limit_price;
    
    // For trailing stop orders
    double trail_percent;
    double trail_price;
    
    // Client order ID
    std::string client_order_id;
    
    // Constructor
    Order()
        : side(OrderSide::BUY),
          type(OrderType::MARKET),
          order_class(OrderClass::SIMPLE),
          time_in_force(TimeInForce::DAY),
          quantity(0.0),
          limit_price(0.0),
          stop_price(0.0),
          take_profit_price(0.0),
          stop_loss_price(0.0),
          stop_loss_limit_price(0.0),
          trail_percent(0.0),
          trail_price(0.0) {
    }
};

// Bracket order structure
struct BracketOrder {
    Order entry;
    Order take_profit;
    Order stop_loss;
};

// Order response
struct OrderResponse {
    // Order ID
    std::string order_id;
    std::string client_order_id;
    
    // Order details
    std::string symbol;
    OrderSide side;
    OrderType type;
    OrderClass order_class;
    TimeInForce time_in_force;
    
    // Quantity and price
    double quantity;
    double filled_quantity;
    double limit_price;
    double stop_price;
    double filled_price;
    
    // Status
    OrderStatus status;
    std::string status_message;
    
    // Timestamps
    uint64_t created_at;
    uint64_t updated_at;
    uint64_t filled_at;
    
    // Related orders
    std::string take_profit_order_id;
    std::string stop_loss_order_id;
};

// Account information
struct AccountInfo {
    std::string account_id;
    std::string currency;
    double equity;
    double cash;
    double buying_power;
    double initial_margin;
    double maintenance_margin;
    double day_trade_buying_power;
    bool pattern_day_trader;
    bool trading_blocked;
    bool account_blocked;
    uint64_t created_at;
    
    // Constructor with default values
    AccountInfo()
        : equity(0.0),
          cash(0.0),
          buying_power(0.0),
          initial_margin(0.0),
          maintenance_margin(0.0),
          day_trade_buying_power(0.0),
          pattern_day_trader(false),
          trading_blocked(false),
          account_blocked(false),
          created_at(0) {
    }
};

// Convert order side to string
inline std::string orderSideToString(OrderSide side) {
    switch (side) {
        case OrderSide::BUY: return "buy";
        case OrderSide::SELL: return "sell";
        default: return "unknown";
    }
}

// Convert order type to string
inline std::string orderTypeToString(OrderType type) {
    switch (type) {
        case OrderType::MARKET: return "market";
        case OrderType::LIMIT: return "limit";
        case OrderType::STOP: return "stop";
        case OrderType::STOP_LIMIT: return "stop_limit";
        case OrderType::TRAILING_STOP: return "trailing_stop";
        default: return "unknown";
    }
}

// Convert order class to string
inline std::string orderClassToString(OrderClass order_class) {
    switch (order_class) {
        case OrderClass::SIMPLE: return "simple";
        case OrderClass::BRACKET: return "bracket";
        case OrderClass::OCO: return "oco";
        case OrderClass::OTO: return "oto";
        default: return "unknown";
    }
}

// Convert time in force to string
inline std::string timeInForceToString(TimeInForce time_in_force) {
    switch (time_in_force) {
        case TimeInForce::DAY: return "day";
        case TimeInForce::GTC: return "gtc";
        case TimeInForce::IOC: return "ioc";
        case TimeInForce::FOK: return "fok";
        default: return "unknown";
    }
}

// Convert order status to string
inline std::string orderStatusToString(OrderStatus status) {
    switch (status) {
        case OrderStatus::NEW: return "new";
        case OrderStatus::PARTIALLY_FILLED: return "partially_filled";
        case OrderStatus::FILLED: return "filled";
        case OrderStatus::CANCELED: return "canceled";
        case OrderStatus::REJECTED: return "rejected";
        case OrderStatus::EXPIRED: return "expired";
        default: return "unknown";
    }
}

} // namespace execution
} // namespace trading_system