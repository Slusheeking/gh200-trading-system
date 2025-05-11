/**
 * ML inference engine implementation
 */

#include <chrono>
#include <thread>
#include <string>
#include <stdexcept>
#include <iostream>

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include "trading_system/ml/inference.h"
#include "trading_system/ml/model.h"

namespace trading_system {
namespace ml {

InferenceEngine::InferenceEngine(const common::Config& config)
    : batch_size_(config.getMLConfig().inference.batch_size),
      use_fp16_(config.getMLConfig().inference.use_fp16),
      max_batch_latency_ms_(config.getMLConfig().inference.max_batch_latency_ms) {
    
    // Load models
    loadModels(config);
}

InferenceEngine::~InferenceEngine() = default;

std::vector<Signal> InferenceEngine::infer(const data::ParsedMarketData& parsed_data) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Extract features from market data
    std::vector<float> features = extractFeatures(parsed_data);
    
    // Skip if no features
    if (features.empty()) {
        return {};
    }
    
    // Run pattern recognition model
    std::vector<float> pattern_outputs;
    if (pattern_recognition_model_) {
        pattern_outputs = pattern_recognition_model_->infer(features);
    }
    
    // Run ranking model
    std::vector<float> ranking_outputs;
    if (ranking_model_) {
        ranking_outputs = ranking_model_->infer(features);
    }
    
    // Generate signals
    std::vector<Signal> signals = generateSignals(
        pattern_outputs, ranking_outputs, parsed_data);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    std::cout << "ML inference completed in " << duration << " µs" << std::endl;
    
    return signals;
}

void InferenceEngine::setThreadAffinity(int core_id) {
    // Store thread ID
    thread_id_ = std::this_thread::get_id();
    
    // Set affinity
    common::pinThreadToCore(core_id);
}

void InferenceEngine::loadModels(const common::Config& config) {
    try {
        // Load pattern recognition model
        const auto& pattern_path = config.getMLConfig().model_paths.pattern_recognition;
        if (!pattern_path.empty()) {
            pattern_recognition_model_ = std::make_unique<PyTorchModel>();
            pattern_recognition_model_->load(pattern_path);
            std::cout << "Loaded pattern recognition model: " << pattern_path << std::endl;
        }
        
        // Load exit optimization model
        const auto& exit_path = config.getMLConfig().model_paths.exit_optimization;
        if (!exit_path.empty()) {
            exit_optimization_model_ = std::make_unique<PyTorchModel>();
            exit_optimization_model_->load(exit_path);
            std::cout << "Loaded exit optimization model: " << exit_path << std::endl;
        }
        
        // Load ranking model
        const auto& ranking_path = config.getMLConfig().model_paths.ranking;
        if (!ranking_path.empty()) {
            ranking_model_ = std::make_unique<PyTorchModel>();
            ranking_model_->load(ranking_path);
            std::cout << "Loaded ranking model: " << ranking_path << std::endl;
        }
        
        // Load sentiment model
        const auto& sentiment_path = config.getMLConfig().model_paths.sentiment;
        if (!sentiment_path.empty()) {
            sentiment_model_ = std::make_unique<PyTorchModel>();
            sentiment_model_->load(sentiment_path);
            std::cout << "Loaded sentiment model: " << sentiment_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading models: " << e.what() << std::endl;
    }
}

std::vector<float> InferenceEngine::extractFeatures(const data::ParsedMarketData& parsed_data) {
    // This is a simplified implementation
    // In a real system, this would extract features from market data
    
    std::vector<float> features;
    
    // Reserve space for features
    features.reserve(parsed_data.symbol_data.size() * 10);  // 10 features per symbol
    
    // Extract features for each symbol
    for (const auto& [symbol, data] : parsed_data.symbol_data) {
        // Add price features
        features.push_back(static_cast<float>(data.last_price));
        features.push_back(static_cast<float>(data.bid_price));
        features.push_back(static_cast<float>(data.ask_price));
        
        // Add volume features
        features.push_back(static_cast<float>(data.volume));
        
        // Add technical indicators
        features.push_back(static_cast<float>(data.rsi_14));
        features.push_back(static_cast<float>(data.macd));
        features.push_back(static_cast<float>(data.macd_signal));
        features.push_back(static_cast<float>(data.macd_histogram));
        features.push_back(static_cast<float>(data.bb_upper / data.last_price));
        features.push_back(static_cast<float>(data.bb_lower / data.last_price));
    }
    
    return features;
}

std::vector<Signal> InferenceEngine::generateSignals(
    const std::vector<float>& pattern_outputs,
    const std::vector<float>& ranking_outputs,
    const data::ParsedMarketData& parsed_data) {
    
    std::vector<Signal> signals;
    
    // Skip if no outputs
    if (pattern_outputs.empty() || ranking_outputs.empty()) {
        return signals;
    }
    
    // Generate signals for each symbol
    size_t i = 0;
    for (const auto& [symbol, data] : parsed_data.symbol_data) {
        // Skip if out of bounds
        if (i >= pattern_outputs.size() / 2) {
            break;
        }
        
        // Get pattern type and confidence
        float pattern_type_value = pattern_outputs[i * 2];
        float pattern_confidence = pattern_outputs[i * 2 + 1];
        
        // Get ranking score
        float ranking_score = ranking_outputs[i];
        
        // Create signal if confidence is high enough
        if (pattern_confidence > 0.7 && ranking_score > 0.5) {
            Signal signal;
            signal.symbol = symbol;
            signal.type = pattern_type_value > 0.5 ? SignalType::BUY : SignalType::SELL;
            signal.confidence = ranking_score;
            signal.price = data.last_price;
            signal.timestamp = data.timestamp;
            
            // Set pattern information
            signal.pattern = static_cast<PatternType>(static_cast<int>(pattern_type_value * 10));
            signal.pattern_confidence = pattern_confidence;
            
            // Set technical indicators
            signal.indicators["rsi_14"] = data.rsi_14;
            signal.indicators["macd"] = data.macd;
            signal.indicators["macd_histogram"] = data.macd_histogram;
            signal.indicators["bb_position"] = (data.last_price - data.bb_lower) / 
                                              (data.bb_upper - data.bb_lower);
            
            // Add to signals
            signals.push_back(signal);
        }
        
        i++;
    }
    
    return signals;
}

} // namespace ml
} // namespace trading_system