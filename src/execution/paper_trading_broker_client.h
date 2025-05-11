/**
 * Paper trading broker client implementation
 */

#pragma once

#include "trading_system/execution/broker_client.h"
#include "trading_system/common/config.h"

namespace trading_system {
namespace execution {

class PaperTradingBrokerClient : public BrokerClient {
public:
    PaperTradingBrokerClient(const common::Config& config);
    ~PaperTradingBrokerClient() override = default;
    
    // Submit order to broker
    OrderResponse submitOrder(const Order& order) override;
    
    // Get order status
    OrderResponse getOrder(const std::string& order_id) override;
    
    // Get all open orders
    std::vector<OrderResponse> getOpenOrders() override;
    
    // Cancel order
    bool cancelOrder(const std::string& order_id) override;
    
    // Get positions
    std::vector<risk::Position> getPositions() override;
    
    // Get account info
    AccountInfo getAccountInfo() override;
    
private:
    // Configuration
    const common::Config& config_;
    
    // Simulated orders
    std::unordered_map<std::string, Order> orders_;
    
    // Simulated positions
    std::unordered_map<std::string, risk::Position> positions_;
    
    // Simulated account
    AccountInfo account_info_;
    
    // Generate order ID
    std::string generateOrderId();
    
    // Update position based on order execution
    void updatePosition(const Order& order, const OrderResponse& response);
};

} // namespace execution
} // namespace trading_system