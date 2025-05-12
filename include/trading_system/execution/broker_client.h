/**
 * Broker client interface
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h" // Added logging include
#include "trading_system/execution/order.h"
#include "trading_system/risk/position.h"

using json = nlohmann::json; // Added using directive

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
};

// Alpaca broker client implementation
class AlpacaBrokerClient : public BrokerClient {
public:
    AlpacaBrokerClient(const common::Config& config);
    ~AlpacaBrokerClient() override;

    OrderResponse submitOrder(const Order& order) override;
    OrderResponse getOrder(const std::string& order_id) override;
    std::vector<OrderResponse> getOpenOrders() override;
    bool cancelOrder(const std::string& order_id) override;
    std::vector<risk::Position> getPositions() override;
    AccountInfo getAccountInfo() override;

private:
    // API credentials
    std::string api_key_;
    std::string api_secret_;
    std::string base_url_;

    // CURL handle
    CURL* curl_;

    // Response buffer
    std::string response_buffer_;

    // Convert order type to string
    std::string orderTypeToString(OrderType type);

    // Convert time in force to string
    std::string timeInForceToString(TimeInForce tif);

    // Send HTTP request
    std::string sendRequest(const std::string& method, const std::string& endpoint, const std::string& data);

    // CURL write callback
    static size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata);

    // Parse order response
    OrderResponse parseOrderResponse(const std::string& response);

    // Parse orders response
    std::vector<OrderResponse> parseOrdersResponse(const std::string& response);

    // Parse positions response
    std::vector<risk::Position> parsePositionsResponse(const std::string& response);

    // Parse account info response
    AccountInfo parseAccountInfoResponse(const std::string& response);
};


// Factory function to create broker client
std::unique_ptr<BrokerClient> createBrokerClient(const common::Config& config);

} // namespace execution
} // namespace trading_system