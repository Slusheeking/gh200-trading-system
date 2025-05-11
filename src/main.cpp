/**
 * Main entry point for the GH200 Trading System
 */

#include <iostream>
#include <chrono>
#include <atomic>
#include <csignal>
#include <string>
#include <thread>
#include <boost/program_options.hpp>

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include "trading_system/data/market_data.h"
#include "trading_system/data/websocket_client.h"
#include "trading_system/cuda/parser.h"
#include "trading_system/ml/inference.h"
#include "trading_system/risk/risk_manager.h"
#include "trading_system/execution/execution_engine.h"

namespace po = boost::program_options;
using namespace trading_system;

// Global signal handler
std::atomic<bool> running{true};

void signalHandler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    running = false;
}

int main(int argc, char** argv) {
    try {
        // Parse command line options
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("config", po::value<std::string>()->default_value("config/system.yaml"), "system configuration file")
            ("trading-config", po::value<std::string>()->default_value("config/trading.yaml"), "trading configuration file")
            ("log-level", po::value<std::string>()->default_value("info"), "log level (debug, info, warning, error)")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        // Register signal handler
        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        
        // Initialize zero-allocation logging
        common::ZeroAllocLogger logger("trading_system");
        logger.setLevel(vm["log-level"].as<std::string>());
        logger.log(LogLevel::INFO, "Starting GH200 Trading System");
        
        // Load configuration
        common::Config config(vm["config"].as<std::string>(), vm["trading-config"].as<std::string>());
        logger.log(LogLevel::INFO, "Configuration loaded");
        
        // Pin main thread to core 2
        common::pinThreadToCore(config.getHardwareConfig().cpu_cores.main_thread);
        
        // Initialize components
        data::WebSocketClient wsClient(config);
        cuda::Parser parser(config);
        ml::InferenceEngine inferenceEngine(config);
        risk::RiskManager riskManager(config);
        execution::ExecutionEngine executionEngine(config);
        
        // Set thread affinities
        wsClient.setThreadAffinity(config.getHardwareConfig().cpu_cores.websocket);
        inferenceEngine.setThreadAffinity(config.getHardwareConfig().cpu_cores.inference);
        riskManager.setThreadAffinity(config.getHardwareConfig().cpu_cores.risk);
        executionEngine.setThreadAffinity(config.getHardwareConfig().cpu_cores.execution);
        
        logger.log(LogLevel::INFO, "All components initialized");
        
        // Connect WebSocket
        wsClient.connect();
        logger.log(LogLevel::INFO, "WebSocket connected");
        
        // Pre-allocate memory for market data
        auto marketData = data::MarketData::createPreallocated(config.getPerformanceConfig().websocket_parser_batch_size);
        
        // Performance monitoring
        uint64_t cycle_count = 0;
        uint64_t total_latency_ns = 0;
        auto last_stats_time = std::chrono::high_resolution_clock::now();
        
        // Main loop
        while (running) {
            auto cycle_start = std::chrono::high_resolution_clock::now();
            
            // Process market data
            wsClient.getLatestData(marketData);
            
            // Parse data with CUDA
            auto parsedData = parser.parse(marketData);
            
            // Run inference
            auto signals = inferenceEngine.infer(parsedData);
            
            // Apply risk management
            auto validatedSignals = riskManager.validateSignals(signals);
            
            // Execute trades
            executionEngine.executeTrades(validatedSignals);
            
            // Calculate cycle latency
            auto cycle_end = std::chrono::high_resolution_clock::now();
            auto cycle_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(cycle_end - cycle_start).count();
            
            // Update statistics
            cycle_count++;
            total_latency_ns += cycle_latency;
            
            // Log performance stats periodically
            auto now = std::chrono::high_resolution_clock::now();
            auto stats_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count();
            
            if (stats_elapsed >= config.getLoggingConfig().latency_log_interval_s) {
                double avg_latency_us = (total_latency_ns / static_cast<double>(cycle_count)) / 1000.0;
                logger.log(LogLevel::INFO, "Performance: " + std::to_string(cycle_count) + 
                          " cycles, avg latency: " + std::to_string(avg_latency_us) + " µs");
                
                // Reset counters
                cycle_count = 0;
                total_latency_ns = 0;
                last_stats_time = now;
            }
            
            // Check if latency exceeds threshold
            if (cycle_latency / 1000 > config.getPerformanceConfig().max_e2e_latency_us) {
                logger.log(LogLevel::WARNING, "High latency detected: " + 
                          std::to_string(cycle_latency / 1000) + " µs");
            }
        }
        
        // Cleanup
        wsClient.disconnect();
        logger.log(LogLevel::INFO, "WebSocket disconnected");
        
        logger.log(LogLevel::INFO, "Trading system shutdown complete");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}