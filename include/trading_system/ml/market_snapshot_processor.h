/**
 * Market Snapshot Processor
 * Processes market data snapshots for the trading system
 * Extracts features and prepares data for ML models
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include "trading_system/common/config.h"
#include "trading_system/data/market_data.h"

namespace trading_system {
namespace ml {

// Forward declarations
class Model;

/**
 * Market Snapshot Processor
 * Processes market data snapshots and extracts features for ML models
 * Uses CUDA acceleration for feature extraction and technical indicators
 */
class MarketSnapshotProcessor {
public:
    // Feature extraction modes
    enum class FeatureMode {
        FAST_PATH,      // Fast path for market scanning
        ACCURATE_PATH,  // Accurate path for signal generation
        EXIT_OPTIMIZATION // Exit optimization for position management
    };

    // Constructor
    MarketSnapshotProcessor(const common::Config& config);
    
    // Destructor
    ~MarketSnapshotProcessor();
    
    // Process a market data snapshot and extract features
    std::unordered_map<std::string, std::vector<float>> processSnapshot(
        const data::ParsedMarketData& snapshot,
        FeatureMode mode = FeatureMode::FAST_PATH
    );
    
    // Process a batch of market data snapshots (for better GPU utilization)
    std::vector<std::unordered_map<std::string, std::vector<float>>> processBatch(
        const std::vector<data::ParsedMarketData>& snapshots,
        FeatureMode mode = FeatureMode::FAST_PATH
    );
    
    // Extract exit features for active positions
    std::unordered_map<std::string, std::vector<float>> extractExitFeatures(
        const data::ParsedMarketData& snapshot,
        const std::vector<std::pair<std::string, std::vector<float>>>& positions
    );
    
    // Calculate technical indicators for a symbol
    std::unordered_map<std::string, float> calculateTechnicalIndicators(
        const std::string& symbol,
        const std::vector<data::Bar>& historical_bars
    );
    
    // Set GPU device to use
    void setGpuDevice(int device_id);
    
    // Enable/disable GPU acceleration
    void setUseGpu(bool use_gpu);

private:
    // Configuration
    common::Config config_;
    bool use_gpu_;
    int gpu_device_id_;
    
    // CUDA resources
    cudaStream_t cuda_stream_;
    void* d_features_buffer_;
    void* d_indicators_buffer_;
    size_t d_features_buffer_size_;
    size_t d_indicators_buffer_size_;
    
    // Historical data cache
    struct HistoricalData {
        std::vector<float> prices;
        std::vector<float> volumes;
        std::vector<float> timestamps;
        size_t max_size;
    };
    std::unordered_map<std::string, HistoricalData> historical_data_;
    std::mutex historical_data_mutex_;
    
    // Market context data
    float market_regime_;
    std::unordered_map<std::string, float> sector_performance_;
    float volatility_index_;
    
    // Feature extraction methods
    std::unordered_map<std::string, std::vector<float>> extractFastPathFeatures(
        const data::ParsedMarketData& snapshot
    );
    
    std::unordered_map<std::string, std::vector<float>> extractAccuratePathFeatures(
        const data::ParsedMarketData& snapshot
    );
    
    // Technical indicator calculation methods
    float calculateRSI(const std::vector<float>& prices, int period);
    std::tuple<float, float, float> calculateMACD(
        const std::vector<float>& prices,
        int fast_period,
        int slow_period,
        int signal_period
    );
    std::tuple<float, float, float> calculateBollingerBands(
        const std::vector<float>& prices,
        int period,
        float std_dev
    );
    float calculateATR(
        const std::vector<float>& high_prices,
        const std::vector<float>& low_prices,
        const std::vector<float>& close_prices,
        int period
    );
    float calculateSMA(const std::vector<float>& prices, int period);
    float calculateSMACrossSignal(const std::vector<float>& prices);
    
    // CUDA kernel launch methods
    void launchFeatureExtractionKernel(
        const data::ParsedMarketData& snapshot,
        FeatureMode mode
    );
    
    void launchTechnicalIndicatorKernel(
        const std::vector<float>& prices,
        const std::vector<float>& volumes,
        void* d_output_buffer,
        size_t output_size
    );
    
    // Initialize CUDA resources
    void initCudaResources();
    
    // Free CUDA resources
    void freeCudaResources();
    
    // Update historical data cache
    void updateHistoricalData(const data::ParsedMarketData& snapshot);
};

} // namespace ml
} // namespace trading_system