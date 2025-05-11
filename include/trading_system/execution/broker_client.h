/**
 * Broker client interface
 */

#pragma once

#include <string>
#include <vector>

#include "trading_system/common/config.h"
#include "trading_system/execution/order.h"
#include "trading_system/risk/position.h"

namespace trading_system {
namespace execution {

// Broker client interface
class BrokerClient {
public:
    virtual ~BrokerClient() = default;
    
    // Submit order
    virtual OrderResponse submitOrder(const Order& order) = 0;
    
    // Get order by ID
    virtual OrderResponse getOrder(const std::string& order_id) = 0;
    
    // Get open orders
    virtual std::vector<OrderResponse> getOpenOrders() = 0;
    
    // Cancel order
    virtual bool cancelOrder(const std::string& order_id) = 0;
    
    // Get positions
    virtual std::vector<risk::Position> getPositions() = 0;
    
    // Get account information
    virtual AccountInfo getAccountInfo() = 0;
    
    // Factory method to create broker client
    static std::unique_ptr<BrokerClient> create(const common::Config& config);
};

// Factory function to create broker client
std::unique_ptr<BrokerClient> createBrokerClient(const common::Config& config);

} // namespace execution
} // namespace trading_system