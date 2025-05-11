/**
 * Metrics publisher for GH200 Trading System
 * Publishes metrics from the trading system to a shared file
 */

#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include "../../include/third_party/json.hpp"
#include "../../include/trading_system/monitoring/metrics_publisher.h"

namespace trading_system {
namespace monitoring {

using json = nlohmann::json;

// Metrics file path (shared with Node.js collector)
const std::string METRICS_FILE = "/tmp/trading_metrics.json";

// Metrics publisher implementation
class MetricsPublisher::Impl {
public:
    Impl()
        : running_(false),
          publish_interval_ms_(100) {
    }
    
    ~Impl() {
        stop();
    }
    
    // Start publishing metrics
    void start() {
        if (running_) {
            return;
        }
        
        running_ = true;
        
        // Start publisher thread
        publisher_thread_ = std::thread([this]() {
            while (running_) {
                publishMetrics();
                std::this_thread::sleep_for(std::chrono::milliseconds(publish_interval_ms_));
            }
        });
    }
    
    // Stop publishing metrics
    void stop() {
        if (!running_) {
            return;
        }
        
        running_ = false;
        
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
    }
    
    // Update system metrics
    void updateSystemMetrics(double cpu_usage, double memory_usage, double gpu_usage) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        metrics_["system"]["cpu_usage"] = cpu_usage;
        metrics_["system"]["memory_usage"] = memory_usage;
        metrics_["system"]["gpu_usage"] = gpu_usage;
    }
    
    // Update latency metrics
    void updateLatencyMetrics(double data_ingestion, double ml_inference, 
                             double risk_check, double execution, double end_to_end) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        metrics_["latency"]["data_ingestion"] = data_ingestion;
        metrics_["latency"]["ml_inference"] = ml_inference;
        metrics_["latency"]["risk_check"] = risk_check;
        metrics_["latency"]["execution"] = execution;
        metrics_["latency"]["end_to_end"] = end_to_end;
    }
    
    // Update trading metrics
    void updateTradingMetrics(int positions, int signals, int trades, double pnl) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        metrics_["trading"]["positions"] = positions;
        metrics_["trading"]["signals"] = signals;
        metrics_["trading"]["trades"] = trades;
        metrics_["trading"]["pnl"] = pnl;
    }
    
private:
    // Publish metrics to file
    void publishMetrics() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        try {
            // Write metrics to file
            std::ofstream file(METRICS_FILE);
            file << metrics_.dump();
            file.close();
        } catch (const std::exception& e) {
            // Log error
            std::cerr << "Error publishing metrics: " << e.what() << std::endl;
        }
    }
    
    // Thread control
    std::atomic<bool> running_;
    std::thread publisher_thread_;
    
    // Metrics data
    json metrics_;
    
    // Synchronization
    std::mutex mutex_;
    
    // Configuration
    int publish_interval_ms_;
};

// MetricsPublisher implementation
MetricsPublisher::MetricsPublisher()
    : impl_(std::make_unique<Impl>()) {
}

MetricsPublisher::~MetricsPublisher() = default;

void MetricsPublisher::start() {
    impl_->start();
}

void MetricsPublisher::stop() {
    impl_->stop();
}

void MetricsPublisher::updateSystemMetrics(double cpu_usage, double memory_usage, double gpu_usage) {
    impl_->updateSystemMetrics(cpu_usage, memory_usage, gpu_usage);
}

void MetricsPublisher::updateLatencyMetrics(double data_ingestion, double ml_inference, 
                                          double risk_check, double execution, double end_to_end) {
    impl_->updateLatencyMetrics(data_ingestion, ml_inference, risk_check, execution, end_to_end);
}

void MetricsPublisher::updateTradingMetrics(int positions, int signals, int trades, double pnl) {
    impl_->updateTradingMetrics(positions, signals, trades, pnl);
}

} // namespace monitoring
} // namespace trading_system