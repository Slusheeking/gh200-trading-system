/**
 * Market Snapshot Processor implementation
 */

#include "trading_system/ml/market_snapshot_processor.h"
#include "trading_system/ml/model.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <chrono>

namespace trading_system {
namespace ml {

// Constructor
MarketSnapshotProcessor::MarketSnapshotProcessor(const common::Config& config)
    : config_(config),
      use_gpu_(true),
      gpu_device_id_(0),
      d_features_buffer_(nullptr),
      d_indicators_buffer_(nullptr),
      d_features_buffer_size_(0),
      d_indicators_buffer_size_(0),
      market_regime_(0.0f),
      volatility_index_(0.0f) {
    
    // Initialize CUDA resources if GPU is enabled
    if (use_gpu_) {
        initCudaResources();
    }
    
    std::cout << "Market Snapshot Processor initialized" << std::endl;
}

// Destructor
MarketSnapshotProcessor::~MarketSnapshotProcessor() {
    // Free CUDA resources
    if (use_gpu_) {
        freeCudaResources();
    }
}

// Process a market data snapshot and extract features
std::unordered_map<std::string, std::vector<float>> MarketSnapshotProcessor::processSnapshot(
    const data::ParsedMarketData& snapshot,
    FeatureMode mode
) {
    // Update historical data
    updateHistoricalData(snapshot);
    
    // Extract features based on mode
    std::unordered_map<std::string, std::vector<float>> features;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    switch (mode) {
        case FeatureMode::FAST_PATH:
            features = extractFastPathFeatures(snapshot);
            break;
        case FeatureMode::ACCURATE_PATH:
            features = extractAccuratePathFeatures(snapshot);
            break;
        case FeatureMode::EXIT_OPTIMIZATION:
            // Exit optimization requires position data, so this is handled separately
            std::cerr << "Exit optimization requires position data. Use extractExitFeatures instead." << std::endl;
            break;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    std::cout << "Feature extraction for " 
              << (mode == FeatureMode::FAST_PATH ? "fast path" : "accurate path")
              << " completed in " << duration / 1000.0 << " ms" << std::endl;
    
    return features;
}

// Process a batch of market data snapshots (for better GPU utilization)
std::vector<std::unordered_map<std::string, std::vector<float>>> MarketSnapshotProcessor::processBatch(
    const std::vector<data::ParsedMarketData>& snapshots,
    FeatureMode mode
) {
    std::vector<std::unordered_map<std::string, std::vector<float>>> results;
    results.reserve(snapshots.size());
    
    // Process each snapshot
    for (const auto& snapshot : snapshots) {
        // Update historical data
        updateHistoricalData(snapshot);
    }
    
    // Process each snapshot individually for now
    // TODO: Implement true batch processing for better GPU utilization
    for (const auto& snapshot : snapshots) {
        results.push_back(processSnapshot(snapshot, mode));
    }
    
    return results;
}

// Extract exit features for active positions
std::unordered_map<std::string, std::vector<float>> MarketSnapshotProcessor::extractExitFeatures(
    const data::ParsedMarketData& snapshot,
    const std::vector<std::pair<std::string, std::vector<float>>>& positions
) {
    std::unordered_map<std::string, std::vector<float>> features;
    
    for (const auto& position : positions) {
        const std::string& symbol = position.first;
        const std::vector<float>& position_data = position.second;
        
        // Skip if symbol not in snapshot
        if (snapshot.symbol_data.find(symbol) == snapshot.symbol_data.end()) {
            continue;
        }
        
        const auto& data = snapshot.symbol_data.at(symbol);
        
        // Create feature vector
        std::vector<float> position_features;
        
        // Position-specific features
        float current_price = data.last_price;
        float entry_price = position_data[0]; // Assuming first element is entry price
        bool is_long = position_data[1] > 0;  // Assuming second element indicates position type (>0 for long)
        
        // Current P&L
        float pnl = is_long ? (current_price - entry_price) / entry_price
                           : (entry_price - current_price) / entry_price;
        position_features.push_back(pnl);
        
        // Position duration in minutes (assuming third element is timestamp)
        uint64_t position_timestamp = static_cast<uint64_t>(position_data[2]);
        float duration_minutes = (snapshot.timestamp - position_timestamp) / 60000000000.0f; // Convert ns to minutes
        position_features.push_back(duration_minutes);
        
        // Initial confidence (assuming fourth element is confidence)
        position_features.push_back(position_data[3]);
        
        // Market condition changes
        position_features.push_back(data.volatility_change);
        position_features.push_back(data.volume / std::max(data.avg_volume, 1.0));
        position_features.push_back(data.bid_ask_spread_change);
        
        // Technical indicators
        position_features.push_back(data.rsi_14);
        position_features.push_back(data.macd);
        position_features.push_back(data.macd_histogram);
        
        // Bollinger band position
        float bb_range = data.bb_upper - data.bb_lower;
        if (bb_range > 0) {
            position_features.push_back((data.last_price - data.bb_lower) / bb_range);
        } else {
            position_features.push_back(0.5f);
        }
        
        // Store features
        features[symbol] = position_features;
    }
    
    return features;
}

// Calculate technical indicators for a symbol
std::unordered_map<std::string, float> MarketSnapshotProcessor::calculateTechnicalIndicators(
    const std::string& symbol,
    const std::vector<data::Bar>& historical_bars
) {
    std::unordered_map<std::string, float> indicators;
    
    // Extract price data
    std::vector<float> prices;
    std::vector<float> high_prices;
    std::vector<float> low_prices;
    std::vector<float> volumes;
    
    for (const auto& bar : historical_bars) {
        prices.push_back(bar.close);
        high_prices.push_back(bar.high);
        low_prices.push_back(bar.low);
        volumes.push_back(static_cast<float>(bar.volume));
    }
    
    // Calculate RSI
    indicators["rsi_14"] = calculateRSI(prices, 14);
    indicators["rsi_9"] = calculateRSI(prices, 9);
    
    // Calculate MACD
    auto [macd, signal, hist] = calculateMACD(prices, 12, 26, 9);
    indicators["macd"] = macd;
    indicators["macd_signal"] = signal;
    indicators["macd_histogram"] = hist;
    
    // Calculate Bollinger Bands
    auto [upper, middle, lower] = calculateBollingerBands(prices, 20, 2.0f);
    indicators["bb_upper"] = upper;
    indicators["bb_middle"] = middle;
    indicators["bb_lower"] = lower;
    
    // Calculate ATR
    indicators["atr"] = calculateATR(high_prices, low_prices, prices, 14);
    
    // Calculate SMAs
    indicators["sma_10"] = calculateSMA(prices, 10);
    indicators["sma_20"] = calculateSMA(prices, 20);
    indicators["sma_50"] = calculateSMA(prices, 50);
    indicators["sma_200"] = calculateSMA(prices, 200);
    
    // Calculate SMA cross signal
    indicators["sma_cross_signal"] = calculateSMACrossSignal(prices);
    
    return indicators;
}

// Set GPU device to use
void MarketSnapshotProcessor::setGpuDevice(int device_id) {
    if (device_id == gpu_device_id_) {
        return;
    }
    
    // Free existing resources
    if (use_gpu_) {
        freeCudaResources();
    }
    
    // Set new device
    gpu_device_id_ = device_id;
    
    // Reinitialize resources
    if (use_gpu_) {
        initCudaResources();
    }
}

// Enable/disable GPU acceleration
void MarketSnapshotProcessor::setUseGpu(bool use_gpu) {
    if (use_gpu == use_gpu_) {
        return;
    }
    
    // Free existing resources if GPU was enabled
    if (use_gpu_) {
        freeCudaResources();
    }
    
    // Update flag
    use_gpu_ = use_gpu;
    
    // Initialize resources if GPU is now enabled
    if (use_gpu_) {
        initCudaResources();
    }
}

// Extract fast path features
std::unordered_map<std::string, std::vector<float>> MarketSnapshotProcessor::extractFastPathFeatures(
    const data::ParsedMarketData& snapshot
) {
    std::unordered_map<std::string, std::vector<float>> features;
    
    // CPU implementation
    for (const auto& [symbol, data] : snapshot.symbol_data) {
        // Skip if missing required data
        if (data.last_price == 0.0 || data.open_price == 0.0) {
            continue;
        }
        
        // Create feature vector
        std::vector<float> symbol_features;
        
        // Price-based features
        symbol_features.push_back(data.last_price / data.open_price);
        symbol_features.push_back(data.last_price / data.prev_close);
        symbol_features.push_back(data.last_price / data.vwap);
        
        // Day high/low position
        float day_range = data.high_price - data.low_price;
        if (day_range > 0) {
            float high_pos = (data.last_price - data.low_price) / day_range;
            float low_pos = (data.high_price - data.last_price) / day_range;
            symbol_features.push_back(high_pos);
            symbol_features.push_back(low_pos);
        } else {
            symbol_features.push_back(0.5f);
            symbol_features.push_back(0.5f);
        }
        
        // Price velocity
        symbol_features.push_back(data.price_change_1m);
        symbol_features.push_back(data.price_change_5m);
        
        // Volume-based features
        symbol_features.push_back(data.volume / std::max(data.avg_volume, 1.0));
        symbol_features.push_back(data.volume_acceleration);
        symbol_features.push_back(data.volume_spike);
        symbol_features.push_back(data.volume_profile_imbalance);
        
        // Technical indicators
        symbol_features.push_back(data.rsi_14);
        symbol_features.push_back(data.macd);
        
        // Bollinger band position
        float bb_range = data.bb_upper - data.bb_lower;
        if (bb_range > 0) {
            float bb_pos = (data.last_price - data.bb_lower) / bb_range;
            symbol_features.push_back(bb_pos);
        } else {
            symbol_features.push_back(0.5f);
        }
        
        // ATR normalized
        symbol_features.push_back(data.atr / data.last_price);
        
        // Additional features
        symbol_features.push_back(data.momentum_1m);
        symbol_features.push_back(data.sma_cross_signal);
        symbol_features.push_back(data.support_resistance_proximity);
        symbol_features.push_back(data.volatility_ratio);
        
        // Store features
        features[symbol] = symbol_features;
    }
    
    return features;
}

// Extract accurate path features
std::unordered_map<std::string, std::vector<float>> MarketSnapshotProcessor::extractAccuratePathFeatures(
    const data::ParsedMarketData& snapshot
) {
    // Get fast path features first
    auto fast_path_features = extractFastPathFeatures(snapshot);
    
    // Add additional features for accurate path
    std::unordered_map<std::string, std::vector<float>> features;
    
    for (const auto& [symbol, base_features] : fast_path_features) {
        // Skip if symbol not in snapshot
        if (snapshot.symbol_data.find(symbol) == snapshot.symbol_data.end()) {
            continue;
        }
        
        const auto& data = snapshot.symbol_data.at(symbol);
        
        // Create extended feature vector
        std::vector<float> symbol_features = base_features;  // Start with fast path features
        
        // Order book metrics
        symbol_features.push_back(data.bid_ask_spread / data.last_price);
        symbol_features.push_back(data.bid_ask_imbalance);
        symbol_features.push_back(static_cast<float>(data.trade_count));
        symbol_features.push_back(data.avg_trade_size);
        symbol_features.push_back(data.large_trade_ratio);
        
        // Trend strength
        symbol_features.push_back(data.price_trend_strength);
        symbol_features.push_back(data.volume_trend_strength);
        
        // Market context
        symbol_features.push_back(data.market_regime);
        
        // Sector performance
        symbol_features.push_back(data.sector_performance);
        
        // Relative strength
        symbol_features.push_back(data.relative_strength);
        
        // Store features
        features[symbol] = symbol_features;
    }
    
    return features;
}

// Calculate RSI
float MarketSnapshotProcessor::calculateRSI(const std::vector<float>& prices, int period) {
    if (prices.size() < period + 1) {
        return 50.0f;
    }
    
    // Calculate price changes
    std::vector<float> deltas(prices.size() - 1);
    for (size_t i = 0; i < deltas.size(); ++i) {
        deltas[i] = prices[i + 1] - prices[i];
    }
    
    // Calculate gains and losses
    std::vector<float> gains(deltas.size());
    std::vector<float> losses(deltas.size());
    
    for (size_t i = 0; i < deltas.size(); ++i) {
        gains[i] = deltas[i] > 0 ? deltas[i] : 0;
        losses[i] = deltas[i] < 0 ? -deltas[i] : 0;
    }
    
    // Calculate average gain and loss
    float avg_gain = std::accumulate(gains.end() - period, gains.end(), 0.0f) / period;
    float avg_loss = std::accumulate(losses.end() - period, losses.end(), 0.0f) / period;
    
    // Calculate RS and RSI
    if (avg_loss == 0) {
        return 100.0f;
    }
    
    float rs = avg_gain / avg_loss;
    float rsi = 100.0f - (100.0f / (1.0f + rs));
    
    return rsi;
}

// Calculate MACD
std::tuple<float, float, float> MarketSnapshotProcessor::calculateMACD(
    const std::vector<float>& prices,
    int fast_period,
    int slow_period,
    int signal_period
) {
    if (prices.size() < slow_period + signal_period) {
        return {0.0f, 0.0f, 0.0f};
    }
    
    // Calculate EMAs
    std::vector<float> fast_ema(prices.size());
    std::vector<float> slow_ema(prices.size());
    
    // Initialize EMAs
    fast_ema[0] = prices[0];
    slow_ema[0] = prices[0];
    
    // Calculate EMAs
    float fast_alpha = 2.0f / (fast_period + 1.0f);
    float slow_alpha = 2.0f / (slow_period + 1.0f);
    
    for (size_t i = 1; i < prices.size(); ++i) {
        fast_ema[i] = prices[i] * fast_alpha + fast_ema[i - 1] * (1.0f - fast_alpha);
        slow_ema[i] = prices[i] * slow_alpha + slow_ema[i - 1] * (1.0f - slow_alpha);
    }
    
    // Calculate MACD line
    std::vector<float> macd_line(prices.size());
    for (size_t i = 0; i < prices.size(); ++i) {
        macd_line[i] = fast_ema[i] - slow_ema[i];
    }
    
    // Calculate signal line
    std::vector<float> signal_line(prices.size());
    signal_line[0] = macd_line[0];
    
    float signal_alpha = 2.0f / (signal_period + 1.0f);
    for (size_t i = 1; i < prices.size(); ++i) {
        signal_line[i] = macd_line[i] * signal_alpha + signal_line[i - 1] * (1.0f - signal_alpha);
    }
    
    // Calculate histogram
    float histogram = macd_line.back() - signal_line.back();
    
    return {macd_line.back(), signal_line.back(), histogram};
}

// Calculate Bollinger Bands
std::tuple<float, float, float> MarketSnapshotProcessor::calculateBollingerBands(
    const std::vector<float>& prices,
    int period,
    float std_dev
) {
    if (prices.size() < period) {
        float price = prices.back();
        return {price * 1.02f, price, price * 0.98f};
    }
    
    // Calculate SMA
    float sum = std::accumulate(prices.end() - period, prices.end(), 0.0f);
    float sma = sum / period;
    
    // Calculate standard deviation
    float variance_sum = 0.0f;
    for (auto it = prices.end() - period; it != prices.end(); ++it) {
        float diff = *it - sma;
        variance_sum += diff * diff;
    }
    float std_deviation = std::sqrt(variance_sum / period);
    
    // Calculate bands
    float upper = sma + std_dev * std_deviation;
    float lower = sma - std_dev * std_deviation;
    
    return {upper, sma, lower};
}

// Calculate ATR
float MarketSnapshotProcessor::calculateATR(
    const std::vector<float>& high_prices,
    const std::vector<float>& low_prices,
    const std::vector<float>& close_prices,
    int period
) {
    if (high_prices.size() < 2 || low_prices.size() < 2 || close_prices.size() < 2) {
        return close_prices.back() * 0.02f;  // Default to 2% of price
    }
    
    // Calculate true ranges
    std::vector<float> true_ranges(close_prices.size() - 1);
    
    for (size_t i = 1; i < close_prices.size(); ++i) {
        float tr1 = high_prices[i] - low_prices[i];
        float tr2 = std::abs(high_prices[i] - close_prices[i - 1]);
        float tr3 = std::abs(low_prices[i] - close_prices[i - 1]);
        
        true_ranges[i - 1] = std::max({tr1, tr2, tr3});
    }
    
    // Calculate ATR
    if (true_ranges.size() < period) {
        return std::accumulate(true_ranges.begin(), true_ranges.end(), 0.0f) / true_ranges.size();
    }
    
    return std::accumulate(true_ranges.end() - period, true_ranges.end(), 0.0f) / period;
}

// Calculate SMA
float MarketSnapshotProcessor::calculateSMA(const std::vector<float>& prices, int period) {
    if (prices.size() < period) {
        return prices.back();
    }
    
    float sum = std::accumulate(prices.end() - period, prices.end(), 0.0f);
    return sum / period;
}

// Calculate SMA cross signal
float MarketSnapshotProcessor::calculateSMACrossSignal(const std::vector<float>& prices) {
    if (prices.size() < 50) {
        return 0.0f;
    }
    
    // Calculate fast and slow SMAs
    std::vector<float> fast_sma(prices.size());
    std::vector<float> slow_sma(prices.size());
    
    // Calculate SMAs
    for (size_t i = 9; i < prices.size(); ++i) {
        float fast_sum = 0.0f;
        for (size_t j = i - 9; j <= i; ++j) {
            fast_sum += prices[j];
        }
        fast_sma[i] = fast_sum / 10.0f;
    }
    
    for (size_t i = 19; i < prices.size(); ++i) {
        float slow_sum = 0.0f;
        for (size_t j = i - 19; j <= i; ++j) {
            slow_sum += prices[j];
        }
        slow_sma[i] = slow_sum / 20.0f;
    }
    
    // Check for crosses
    if (fast_sma[prices.size() - 2] < slow_sma[prices.size() - 2] &&
        fast_sma[prices.size() - 1] > slow_sma[prices.size() - 1]) {
        return 1.0f;  // Bullish cross
    }
    
    if (fast_sma[prices.size() - 2] > slow_sma[prices.size() - 2] &&
        fast_sma[prices.size() - 1] < slow_sma[prices.size() - 1]) {
        return -1.0f;  // Bearish cross
    }
    
    return 0.0f;  // No cross
}

// Initialize CUDA resources
void MarketSnapshotProcessor::initCudaResources() {
    if (!use_gpu_) {
        return;
    }
    
    try {
        // Set device
        cudaError_t error = cudaSetDevice(gpu_device_id_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
            use_gpu_ = false;
            return;
        }
        
        // Create CUDA stream
        error = cudaStreamCreate(&cuda_stream_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
            use_gpu_ = false;
            return;
        }
        
        // Allocate device memory for features
        d_features_buffer_size_ = 1024 * 1024 * sizeof(float);  // 1M floats
        error = cudaMalloc(&d_features_buffer_, d_features_buffer_size_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for features: " << cudaGetErrorString(error) << std::endl;
            use_gpu_ = false;
            return;
        }
        
        // Allocate device memory for indicators
        d_indicators_buffer_size_ = 1024 * 1024 * sizeof(float);  // 1M floats
        error = cudaMalloc(&d_indicators_buffer_, d_indicators_buffer_size_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for indicators: " << cudaGetErrorString(error) << std::endl;
            use_gpu_ = false;
            return;
        }
        
        std::cout << "CUDA resources initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception during CUDA initialization: " << e.what() << std::endl;
        use_gpu_ = false;
    }
}

// Free CUDA resources
void MarketSnapshotProcessor::freeCudaResources() {
    if (!use_gpu_) {
        return;
    }
    
    // Free device memory
    if (d_features_buffer_) {
        cudaFree(d_features_buffer_);
        d_features_buffer_ = nullptr;
    }
    
    if (d_indicators_buffer_) {
        cudaFree(d_indicators_buffer_);
        d_indicators_buffer_ = nullptr;
    }
    
    // Destroy CUDA stream
    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = 0;
    }
}

// Update historical data cache
void MarketSnapshotProcessor::updateHistoricalData(const data::ParsedMarketData& snapshot) {
    std::lock_guard<std::mutex> lock(historical_data_mutex_);
    
    for (const auto& [symbol, data] : snapshot.symbol_data) {
        // Initialize if not exists
        if (historical_data_.find(symbol) == historical_data_.end()) {
            historical_data_[symbol] = {
                .prices = {},
                .volumes = {},
                .timestamps = {},
                .max_size = 100
            };
        }
        
        auto& hist_data = historical_data_[symbol];
        
        // Add new data
        hist_data.prices.push_back(data.last_price);
        hist_data.volumes.push_back(data.volume);
        hist_data.timestamps.push_back(snapshot.timestamp);
        
        // Trim if too large
        if (hist_data.prices.size() > hist_data.max_size) {
            hist_data.prices.erase(hist_data.prices.begin());
            hist_data.volumes.erase(hist_data.volumes.begin());
            hist_data.timestamps.erase(hist_data.timestamps.begin());
        }
    }
}

// Launch feature extraction kernel
void MarketSnapshotProcessor::launchFeatureExtractionKernel(
    const data::ParsedMarketData& snapshot,
    FeatureMode mode
) {
    // This is a placeholder for actual CUDA kernel launch
    // In a real implementation, this would copy data to GPU, launch kernels, and copy results back
    
    if (!use_gpu_) {
        return;
    }
    
    // TODO: Implement actual CUDA kernel launch
    std::cout << "GPU feature extraction not yet implemented, falling back to CPU" << std::endl;
}

// Launch technical indicator kernel
void MarketSnapshotProcessor::launchTechnicalIndicatorKernel(
    const std::vector<float>& prices,
    const std::vector<float>& volumes,
    void* d_output_buffer,
    size_t output_size
) {
    // This is a placeholder for actual CUDA kernel launch
    // In a real implementation, this would copy data to GPU, launch kernels, and copy results back
    
    if (!use_gpu_) {
        return;
    }
    
    // TODO: Implement actual CUDA kernel launch
    std::cout << "GPU technical indicator calculation not yet implemented, falling back to CPU" << std::endl;
}

} // namespace ml
} // namespace trading_system