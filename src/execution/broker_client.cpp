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

// Factory function to create broker client
std::unique_ptr<BrokerClient> createBrokerClient(const common::Config& config) {
    // Always create Alpaca broker client
    return std::make_unique<AlpacaBrokerClient>(config);
}

} // namespace execution
} // namespace trading_system