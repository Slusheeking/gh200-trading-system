/**
 * Broker client implementation for Alpaca
 */

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include "trading_system/execution/broker_client.h"
#include "trading_system/execution/order.h"

using json = nlohmann::json;

namespace trading_system {
namespace execution {

// Alpaca broker client implementation
class AlpacaBrokerClient : public BrokerClient {
public:
    AlpacaBrokerClient(const common::Config& config)
        : BrokerClient(),
          curl_(nullptr) {
        
        // Get Alpaca configuration
        const auto& alpaca_config = config.getDataSourceConfig("alpaca");
        
        // Store API credentials
        api_key_ = alpaca_config.api_key;
        api_secret_ = alpaca_config.api_secret;
        base_url_ = alpaca_config.base_url;
        
        // Initialize CURL
        curl_ = curl_easy_init();
        if (!curl_) {
            throw std::runtime_error("Failed to initialize CURL");
        }
        
        // Set common CURL options
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 10L);  // 10 second timeout
        
        // Set headers
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("APCA-API-KEY-ID: " + api_key_).c_str());
        headers = curl_slist_append(headers, ("APCA-API-SECRET-KEY: " + api_secret_).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    }
    
    ~AlpacaBrokerClient() override {
        // Clean up CURL
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
    }
    
    OrderResponse submitOrder(const Order& order) override {
        // Create order JSON
        json j = {
            {"symbol", order.symbol},
            {"qty", order.quantity},
            {"side", order.side == OrderSide::BUY ? "buy" : "sell"},
            {"type", orderTypeToString(order.type)},
            {"time_in_force", timeInForceToString(order.time_in_force)}
        };
        
        // Add limit price if applicable
        if (order.type == OrderType::LIMIT) {
            j["limit_price"] = order.limit_price;
        }
        
        // Handle bracket orders
        if (order.order_class == OrderClass::BRACKET) {
            j["order_class"] = "bracket";
            
            // Add take profit
            if (order.take_profit_price > 0) {
                j["take_profit"] = {
                    {"limit_price", order.take_profit_price}
                };
            }
            
            // Add stop loss
            if (order.stop_loss_price > 0) {
                j["stop_loss"] = {
                    {"stop_price", order.stop_loss_price}
                };
                
                // Add limit price for stop loss if specified
                if (order.stop_loss_limit_price > 0) {
                    j["stop_loss"]["limit_price"] = order.stop_loss_limit_price;
                }
            }
        }
        
        // Convert to string
        std::string json_str = j.dump();
        
        // Send request
        std::string response = sendRequest("POST", "/v2/orders", json_str);
        
        // Parse response
        return parseOrderResponse(response);
    }
    
    OrderResponse getOrder(const std::string& order_id) override {
        // Send request
        std::string response = sendRequest("GET", "/v2/orders/" + order_id, "");
        
        // Parse response
        return parseOrderResponse(response);
    }
    
    std::vector<OrderResponse> getOpenOrders() override {
        // Send request
        std::string response = sendRequest("GET", "/v2/orders?status=open", "");
        
        // Parse response
        return parseOrdersResponse(response);
    }
    
    bool cancelOrder(const std::string& order_id) override {
        // Send request
        std::string response = sendRequest("DELETE", "/v2/orders/" + order_id, "");
        
        // Check if successful (empty response means success)
        return response.empty();
    }
    
    std::vector<risk::Position> getPositions() override {
        // Send request
        std::string response = sendRequest("GET", "/v2/positions", "");
        
        // Parse response
        return parsePositionsResponse(response);
    }
    
    AccountInfo getAccountInfo() override {
        // Send request
        std::string response = sendRequest("GET", "/v2/account", "");
        
        // Parse response
        return parseAccountInfoResponse(response);
    }
    
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
    std::string orderTypeToString(OrderType type) {
        switch (type) {
            case OrderType::MARKET: return "market";
            case OrderType::LIMIT: return "limit";
            case OrderType::STOP: return "stop";
            case OrderType::STOP_LIMIT: return "stop_limit";
            default: return "market";
        }
    }
    
    // Convert time in force to string
    std::string timeInForceToString(TimeInForce tif) {
        switch (tif) {
            case TimeInForce::DAY: return "day";
            case TimeInForce::GTC: return "gtc";
            case TimeInForce::IOC: return "ioc";
            case TimeInForce::FOK: return "fok";
            default: return "day";
        }
    }
    
    // Send HTTP request
    std::string sendRequest(const std::string& method, const std::string& endpoint, const std::string& data) {
        // Clear response buffer
        response_buffer_.clear();
        
        // Set URL
        std::string url = base_url_ + endpoint;
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        
        // Set method
        if (method == "GET") {
            curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
        } else if (method == "POST") {
            curl_easy_setopt(curl_, CURLOPT_POST, 1L);
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data.c_str());
        } else if (method == "DELETE") {
            curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
        }
        
        // Set write callback
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_buffer_);
        
        // Perform request
        CURLcode res = curl_easy_perform(curl_);
        if (res != CURLE_OK) {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
            return "";
        }
        
        // Check response code
        long response_code;
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response_code);
        if (response_code >= 400) {
            std::cerr << "HTTP error: " << response_code << " - " << response_buffer_ << std::endl;
            return "";
        }
        
        return response_buffer_;
    }
    
    // CURL write callback
    static size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
        std::string* response = static_cast<std::string*>(userdata);
        response->append(ptr, size * nmemb);
        return size * nmemb;
    }
    
    // Parse order response
    OrderResponse parseOrderResponse(const std::string& response) {
        OrderResponse result;
        
        try {
            json j = json::parse(response);
            
            result.order_id = j["id"];
            result.client_order_id = j["client_order_id"];
            result.symbol = j["symbol"];
            result.status = j["status"];
            result.created_at = j["created_at"];
            
            // Parse filled quantity and price if available
            if (j.contains("filled_qty")) {
                result.filled_quantity = std::stod(j["filled_qty"].get<std::string>());
            }
            
            if (j.contains("filled_avg_price")) {
                result.filled_price = std::stod(j["filled_avg_price"].get<std::string>());
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing order response: " << e.what() << std::endl;
        }
        
        return result;
    }
    
    // Parse orders response
    std::vector<OrderResponse> parseOrdersResponse(const std::string& response) {
        std::vector<OrderResponse> results;
        
        try {
            json j = json::parse(response);
            
            for (const auto& order_json : j) {
                OrderResponse order;
                
                order.order_id = order_json["id"];
                order.client_order_id = order_json["client_order_id"];
                order.symbol = order_json["symbol"];
                order.status = order_json["status"];
                order.created_at = order_json["created_at"];
                
                // Parse filled quantity and price if available
                if (order_json.contains("filled_qty")) {
                    order.filled_quantity = std::stod(order_json["filled_qty"].get<std::string>());
                }
                
                if (order_json.contains("filled_avg_price")) {
                    order.filled_price = std::stod(order_json["filled_avg_price"].get<std::string>());
                }
                
                results.push_back(order);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing orders response: " << e.what() << std::endl;
        }
        
        return results;
    }
    
    // Parse positions response
    std::vector<risk::Position> parsePositionsResponse(const std::string& response) {
        std::vector<risk::Position> results;
        
        try {
            json j = json::parse(response);
            
            for (const auto& pos_json : j) {
                risk::Position position;
                
                position.symbol = pos_json["symbol"];
                position.quantity = std::stod(pos_json["qty"].get<std::string>());
                position.entry_price = std::stod(pos_json["avg_entry_price"].get<std::string>());
                position.current_price = std::stod(pos_json["current_price"].get<std::string>());
                position.unrealized_pnl = std::stod(pos_json["unrealized_pl"].get<std::string>());
                
                results.push_back(position);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing positions response: " << e.what() << std::endl;
        }
        
        return results;
    }
    
    // Parse account info response
    AccountInfo parseAccountInfoResponse(const std::string& response) {
        AccountInfo result;
        
        try {
            json j = json::parse(response);
            
            result.account_id = j["id"];
            result.cash = std::stod(j["cash"].get<std::string>());
            // Store portfolio value in equity since AccountInfo doesn't have portfolio_value
            result.equity = std::stod(j["portfolio_value"].get<std::string>());
            result.buying_power = std::stod(j["buying_power"].get<std::string>());
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing account info response: " << e.what() << std::endl;
        }
        
        return result;
    }
};

// Factory function to create broker client
std::unique_ptr<BrokerClient> createBrokerClient(const common::Config& config) {
    // Get broker type from config
    std::string broker_type = "alpaca";  // Default to Alpaca
    
    // Create appropriate broker client
    if (broker_type == "alpaca") {
        return std::make_unique<AlpacaBrokerClient>(config);
    } else {
        throw std::runtime_error("Unsupported broker type: " + broker_type);
    }
}

} // namespace execution
} // namespace trading_system