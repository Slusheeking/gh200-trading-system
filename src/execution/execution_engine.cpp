/**
 * Execution engine implementation
 */

#include <chrono>
#include <thread>
#include <stdexcept>
#include <iostream>

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include "trading_system/execution/execution_engine.h"
#include "paper_trading_broker_client.h"

namespace trading_system {
namespace execution {

ExecutionEngine::ExecutionEngine(const common::Config& config)
    : default_order_type_(config.getTradingConfig().orders.default_order_type),
      use_bracket_orders_(config.getTradingConfig().orders.use_bracket_orders),
      time_in_force_(config.getTradingConfig().orders.time_in_force),
      default_stop_loss_pct_(config.getTradingConfig().risk.default_stop_loss_pct),
      default_take_profit_pct_(config.getTradingConfig().risk.default_take_profit_pct),
      use_trailing_stop_(config.getTradingConfig().risk.use_trailing_stop),
      trailing_stop_pct_(config.getTradingConfig().risk.trailing_stop_pct) {
    
    // Create broker client
    broker_client_ = createBrokerClient(config);
    
    if (!broker_client_) {
        throw std::runtime_error("Failed to create broker client");
    }
}

ExecutionEngine::~ExecutionEngine() = default;

void ExecutionEngine::executeTrades(const std::vector<ml::Signal>& validated_signals) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Skip if no signals
    if (validated_signals.empty()) {
        return;
    }
    
    // Execute each signal
    for (const auto& signal : validated_signals) {
        try {
            // Create order
            Order order;
            
            if (use_bracket_orders_) {
                // Create bracket order
                BracketOrder bracket_order = createBracketOrder(signal);
                
                // Submit bracket order
                OrderResponse response = broker_client_->submitOrder(bracket_order.entry);
                
                // Handle response
                handleOrderResponse(response);
            } else {
                // Create simple order
                order = createOrder(signal);
                
                // Submit order
                OrderResponse response = broker_client_->submitOrder(order);
                
                // Handle response
                handleOrderResponse(response);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error executing trade for " << signal.symbol << ": " << e.what() << std::endl;
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    std::cout << "Trade execution completed in " << duration
              << " µs for " << validated_signals.size() << " signals" << std::endl;
}

void ExecutionEngine::setThreadAffinity(int core_id) {
    // Store thread ID
    thread_id_ = std::this_thread::get_id();
    
    // Set affinity
    common::pinThreadToCore(core_id);
}

Order ExecutionEngine::createOrder(const ml::Signal& signal) {
    Order order;
    
    // Set basic order info
    order.symbol = signal.symbol;
    order.side = signal.type == ml::SignalType::BUY ? OrderSide::BUY : OrderSide::SELL;
    order.type = default_order_type_ == "market" ? OrderType::MARKET : OrderType::LIMIT;
    order.order_class = OrderClass::SIMPLE;
    order.time_in_force = time_in_force_ == "day" ? TimeInForce::DAY : 
                         (time_in_force_ == "gtc" ? TimeInForce::GTC : 
                         (time_in_force_ == "ioc" ? TimeInForce::IOC : TimeInForce::FOK));
    
    // Set quantity
    order.quantity = signal.position_size / signal.price;
    
    // Set limit price if applicable
    if (order.type == OrderType::LIMIT) {
        // For buy orders, limit price is slightly above current price
        // For sell orders, limit price is slightly below current price
        double slippage_factor = 0.001;  // 0.1% slippage
        order.limit_price = order.side == OrderSide::BUY ? 
                          signal.price * (1.0 + slippage_factor) : 
                          signal.price * (1.0 - slippage_factor);
    }
    
    // Generate client order ID
    order.client_order_id = signal.symbol + "_" + 
                           std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    
    return order;
}

BracketOrder ExecutionEngine::createBracketOrder(const ml::Signal& signal) {
    BracketOrder bracket_order;
    
    // Create entry order
    bracket_order.entry = createOrder(signal);
    bracket_order.entry.order_class = OrderClass::BRACKET;
    
    // Set take profit price
    double take_profit_pct = signal.take_profit > 0.0 ? 
                            (signal.take_profit - signal.price) / signal.price * 100.0 : 
                            default_take_profit_pct_;
    
    bracket_order.entry.take_profit_price = bracket_order.entry.side == OrderSide::BUY ? 
                                          signal.price * (1.0 + take_profit_pct / 100.0) : 
                                          signal.price * (1.0 - take_profit_pct / 100.0);
    
    // Set stop loss price
    double stop_loss_pct = signal.stop_loss > 0.0 ? 
                          (signal.price - signal.stop_loss) / signal.price * 100.0 : 
                          default_stop_loss_pct_;
    
    bracket_order.entry.stop_loss_price = bracket_order.entry.side == OrderSide::BUY ? 
                                        signal.price * (1.0 - stop_loss_pct / 100.0) : 
                                        signal.price * (1.0 + stop_loss_pct / 100.0);
    
    // Set trailing stop if enabled
    if (use_trailing_stop_) {
        bracket_order.entry.trail_percent = trailing_stop_pct_;
    }
    
    return bracket_order;
}

void ExecutionEngine::handleOrderResponse(const OrderResponse& response) {
    // Log order response
    std::cout << "Order " << response.order_id << " for " << response.symbol
              << " submitted with status " << orderStatusToString(response.status) << std::endl;
    
    // Handle different statuses
    switch (response.status) {
        case OrderStatus::NEW:
        case OrderStatus::PARTIALLY_FILLED:
            // Order accepted, nothing to do
            break;
            
        case OrderStatus::FILLED:
            // Order filled, update position
            std::cout << "Order " << response.order_id << " filled at "
                     << response.filled_price << std::endl;
            break;
            
        case OrderStatus::REJECTED:
            // Order rejected, log error
            std::cerr << "Order " << response.order_id << " rejected: "
                     << response.status_message << std::endl;
            break;
            
        case OrderStatus::CANCELED:
        case OrderStatus::EXPIRED:
            // Order canceled or expired, log warning
            std::cout << "Order " << response.order_id << " "
                     << orderStatusToString(response.status) << ": "
                     << response.status_message << std::endl;
            break;
    }
}

std::unique_ptr<BrokerClient> ExecutionEngine::createBrokerClient(const common::Config& config) {
    // Create broker client based on configuration
    std::string broker_type = "paper"; // Default to paper trading
    
    // In a real implementation, we would get the broker type from config
    // For now, just create a paper trading broker client
    return std::make_unique<PaperTradingBrokerClient>(config);
}

std::string ExecutionEngine::orderStatusToString(OrderStatus status) {
    switch (status) {
        case OrderStatus::NEW:
            return "NEW";
        case OrderStatus::PARTIALLY_FILLED:
            return "PARTIALLY_FILLED";
        case OrderStatus::FILLED:
            return "FILLED";
        case OrderStatus::CANCELED:
            return "CANCELED";
        case OrderStatus::REJECTED:
            return "REJECTED";
        case OrderStatus::EXPIRED:
            return "EXPIRED";
        default:
            return "UNKNOWN";
    }
}

} // namespace execution
} // namespace trading_system