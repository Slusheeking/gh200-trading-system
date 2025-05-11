/**
 * Metrics publisher interface
 */

#pragma once

#include <memory>

namespace trading_system {
namespace monitoring {

// Metrics publisher class
class MetricsPublisher {
public:
    MetricsPublisher();
    ~MetricsPublisher();
    
    // Start publishing metrics
    void start();
    
    // Stop publishing metrics
    void stop();
    
    // Update system metrics
    void updateSystemMetrics(double cpu_usage, double memory_usage, double gpu_usage);
    
    // Update latency metrics
    void updateLatencyMetrics(double data_ingestion, double ml_inference, 
                             double risk_check, double execution, double end_to_end);
    
    // Update trading metrics
    void updateTradingMetrics(int positions, int signals, int trades, double pnl);
    
private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace monitoring
} // namespace trading_system