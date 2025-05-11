/**
 * Paper trading broker client implementation
 */

#include <iostream>
#include <random>
#include <chrono>
#include <unordered_map>

#include "paper_trading_broker_client.h"

namespace trading_system {
namespace execution {

PaperTradingBrokerClient::PaperTradingBrokerClient(const common::Config& config)
    : config_(config) {
    
    // Initialize account info
    account_info_.account_id = "paper_trading";
    account_info_.buying_power = 1000000.0;  // $1M buying power
    account_info_.cash = 1000000.0;          // $1M cash
    account_info_.equity = 1000000.0;        // $1M equity
    account_info_.initial_margin = 0.0;      // No margin used
}

OrderResponse PaperTradingBrokerClient::submitOrder(const Order& order) {
    // Generate order ID
    std::string order_id = generateOrderId();
    
    // Store order
    orders_[order_id] = order;
    
    // Create response
    OrderResponse response;
    response.order_id = order_id;
    response.symbol = order.symbol;
    response.status = OrderStatus::NEW;
    response.status_message = "Order accepted";
    
    // Simulate fill (in a real system, this would happen asynchronously)
    if (order.type == OrderType::MARKET) {
        // Market orders fill immediately
        response.status = OrderStatus::FILLED;
        response.filled_price = order.limit_price > 0.0 ? order.limit_price : 100.0;  // Use limit price if available
        response.filled_quantity = order.quantity;
        
        // Update position
        updatePosition(order, response);
    }
    
    return response;
}

OrderResponse PaperTradingBrokerClient::getOrder(const std::string& order_id) {
    // Check if order exists
    if (orders_.find(order_id) == orders_.end()) {
        OrderResponse response;
        response.order_id = order_id;
        response.status = OrderStatus::REJECTED;
        response.status_message = "Order not found";
        return response;
    }
    
    // Get order
    const Order& order = orders_[order_id];
    
    // Create response
    OrderResponse response;
    response.order_id = order_id;
    response.symbol = order.symbol;
    response.status = OrderStatus::FILLED;  // Assume all orders are filled for simplicity
    response.status_message = "Order filled";
    response.filled_price = order.limit_price > 0.0 ? order.limit_price : 100.0;
    response.filled_quantity = order.quantity;
    
    return response;
}

std::vector<OrderResponse> PaperTradingBrokerClient::getOpenOrders() {
    std::vector<OrderResponse> responses;
    
    // Convert all orders to responses
    for (const auto& pair : orders_) {
        const std::string& order_id = pair.first;
        const Order& order = pair.second;
        
        OrderResponse response;
        response.order_id = order_id;
        response.symbol = order.symbol;
        response.status = OrderStatus::NEW;  // Assume all orders are open
        response.status_message = "Order open";
        
        responses.push_back(response);
    }
    
    return responses;
}

bool PaperTradingBrokerClient::cancelOrder(const std::string& order_id) {
    // Check if order exists
    if (orders_.find(order_id) == orders_.end()) {
        return false;
    }
    
    // Remove order
    orders_.erase(order_id);
    
    return true;
}

std::vector<risk::Position> PaperTradingBrokerClient::getPositions() {
    std::vector<risk::Position> result;
    
    // Convert positions map to vector
    for (const auto& pair : positions_) {
        result.push_back(pair.second);
    }
    
    return result;
}

AccountInfo PaperTradingBrokerClient::getAccountInfo() {
    return account_info_;
}

std::string PaperTradingBrokerClient::generateOrderId() {
    // Generate random order ID
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 999999);
    
    // Use current time and random number
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch().count();
    
    return "order_" + std::to_string(value) + "_" + std::to_string(dis(gen));
}

void PaperTradingBrokerClient::updatePosition(const Order& order, const OrderResponse& response) {
    // Check if position exists
    if (positions_.find(order.symbol) == positions_.end()) {
        // Create new position
        risk::Position position;
        position.symbol = order.symbol;
        position.quantity = 0;
        position.entry_price = 0.0;
        position.current_price = 0.0;
        position.unrealized_pnl = 0.0;
        position.realized_pnl = 0.0;
        position.side = risk::PositionSide::LONG;
        
        positions_[order.symbol] = position;
    }
    
    // Get position
    risk::Position& position = positions_[order.symbol];
    
    // Update position
    if (order.side == OrderSide::BUY) {
        // Buy order
        double cost = response.filled_price * response.filled_quantity;
        double total_cost = position.entry_price * position.quantity + cost;
        position.quantity += response.filled_quantity;
        position.entry_price = total_cost / position.quantity;
        position.current_price = response.filled_price;
    } else {
        // Sell order
        double revenue = response.filled_price * response.filled_quantity;
        double cost = position.entry_price * response.filled_quantity;
        position.realized_pnl += revenue - cost;
        position.quantity -= response.filled_quantity;
        position.current_price = response.filled_price;
        
        // If position is closed, reset entry price
        if (position.quantity <= 0) {
            position.entry_price = 0.0;
            position.quantity = 0;
        }
    }
    
    // Update account info
    if (order.side == OrderSide::BUY) {
        account_info_.cash -= response.filled_price * response.filled_quantity;
    } else {
        account_info_.cash += response.filled_price * response.filled_quantity;
    }
    
    // Update equity
    account_info_.equity = account_info_.cash;
    for (const auto& pair : positions_) {
        const risk::Position& pos = pair.second;
        if (pos.quantity > 0) {
            // Assume current price is the same as entry price for simplicity
            account_info_.equity += pos.quantity * pos.entry_price;
        }
    }
}

} // namespace execution
} // namespace trading_system