/**
 * Order execution engine
 */

#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <string>
#include <chrono>
#include <atomic>
#include "trading_system/common/config.h"
#include "trading_system/ml/signals.h"
#include "trading_system/execution/broker_client.h"

namespace trading_system {
namespace execution {

class ExecutionEngine {
public:
    ExecutionEngine(const common::Config& config);
    ~ExecutionEngine();
    
    // Execute trades based on validated signals
    void executeTrades(const std::vector<ml::Signal>& validated_signals);
    
    // Set thread affinity
    void setThreadAffinity(int core_id);
    
private:
    // Broker client
    std::unique_ptr<BrokerClient> broker_client_;
    
    // Thread ID for affinity
    std::thread::id thread_id_;
    
    // Configuration
    std::string default_order_type_;
    bool use_bracket_orders_;
    std::string time_in_force_;
    
    // Performance monitoring
    std::atomic<uint64_t> total_execution_time_ns_;
    std::atomic<uint64_t> execution_count_;
    
    // Create order from signal
    Order createOrder(const ml::Signal& signal);
    
    // Submit order to broker
    void submitOrder(const Order& order);
};

} // namespace execution
} // namespace trading_system