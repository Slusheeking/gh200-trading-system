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
#include "trading_system/ml/gbdt_model.h"
#include "trading_system/ml/axial_attention_model.h" // Added Axial Attention model
#include "trading_system/ml/lstm_gru_model.h"      // Added LSTM/GRU model
#include <algorithm>
#include <numeric>

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
        // Legacy models - now using LightGBM for better performance
        // Load pattern recognition model
        const auto& pattern_path = config.getMLConfig().model_paths.pattern_recognition;
        if (!pattern_path.empty()) {
            pattern_recognition_model_ = Model::create("lightgbm");
            pattern_recognition_model_->load(pattern_path);
            std::cout << "Loaded pattern recognition model: " << pattern_path << std::endl;
        }
        
        // Load exit optimization model
        const auto& exit_path = config.getMLConfig().model_paths.exit_optimization;
        if (!exit_path.empty()) {
            exit_optimization_model_ = Model::create("lightgbm");
            exit_optimization_model_->load(exit_path);
            std::cout << "Loaded exit optimization model: " << exit_path << std::endl;
        }
        
        // Load ranking model
        const auto& ranking_path = config.getMLConfig().model_paths.ranking;
        if (!ranking_path.empty()) {
            ranking_model_ = Model::create("lightgbm");
            ranking_model_->load(ranking_path);
            std::cout << "Loaded ranking model: " << ranking_path << std::endl;
        }
        
        // Load sentiment model
        const auto& sentiment_path = config.getMLConfig().model_paths.sentiment;
        if (!sentiment_path.empty()) {
            sentiment_model_ = Model::create("lightgbm");
            sentiment_model_->load(sentiment_path);
            std::cout << "Loaded sentiment model: " << sentiment_path << std::endl;
        }
        
        // Hybrid approach models
        
        // Load fast path model (Market Scanner)
        const auto& fast_path_model = config.getMLConfig().model_paths.fast_path;
        if (!fast_path_model.empty()) {
            fast_path_model_ = std::make_unique<GBDTModel>();
            fast_path_model_->load(fast_path_model);
            std::cout << "Loaded fast path model: " << fast_path_model << std::endl;
            
            // Configure GBDT model according to GH200_Trading_System_Architecture.md
            fast_path_model_->setNumTrees(150);
            fast_path_model_->setMaxDepth(8);
            fast_path_model_->setLearningRate(0.05);
            fast_path_model_->setFeatureFraction(0.8);
            fast_path_model_->setBaggingFraction(0.7);
            
            // Set feature names - expanded to include all features from architecture doc
            std::vector<std::string> feature_names = {
                // Price momentum features
                "price_rel_open", "price_rel_close", "price_rel_vwap",
                "price_velocity_1m", "price_velocity_5m", "price_velocity_15m",
                
                // Volatility features
                "day_high_pos", "day_low_pos", "atr_normalized", "volatility_ratio",
                
                // Volume features
                "volume_rel_avg", "volume_accel", "volume_spike", "volume_profile_imbalance",
                
                // Technical indicators
                "rsi_14", "macd", "bb_position", "momentum_1m",
                
                // Pattern-based features
                "sma_cross_signal", "support_resistance_proximity",
                
                // Market context
                "sector_performance", "market_regime"
            };
            fast_path_model_->setFeatureNames(feature_names);
            
            // Set LightGBM-specific parameters
            if (auto* lgbm_model = dynamic_cast<LightGBMModel*>(fast_path_model_.get())) {
                lgbm_model->setNumThreads(4);  // Use 4 threads for inference
                lgbm_model->setPredictionType("raw");  // Use raw scores
            }
        }
        
        // Load accurate path model (Signal Generator) - Direct TensorRT
        const auto& accurate_path_model_path = config.getMLConfig().model_paths.accurate_path;
        if (!accurate_path_model_path.empty()) {
            accurate_path_model_ = Model::create("axial_attention");
            
            // Configure Axial Attention model according to GH200_Trading_System_Architecture.md
            if (auto* aa_model = dynamic_cast<AxialAttentionModel*>(accurate_path_model_.get())) {
                aa_model->setNumHeads(4);
                aa_model->setHeadDimension(64);
                aa_model->setNumLayers(6);
                aa_model->setSequenceLength(100);
                aa_model->setDropout(0.1);
                // FP16 precision is enabled by default in the model constructor
            }
            
            accurate_path_model_->load(accurate_path_model_path);
            std::cout << "Loaded accurate path model: " << accurate_path_model_path << std::endl;
        }
        
        // Load exit model (Exit Optimization) - Direct TensorRT
        const auto& exit_model_path = config.getMLConfig().model_paths.exit_model;
        if (!exit_model_path.empty()) {
            exit_model_ = Model::create("lstm_gru");
            
            // Configure LSTM/GRU model according to GH200_Trading_System_Architecture.md
            if (auto* lstm_model = dynamic_cast<LstmGruModel*>(exit_model_.get())) {
                lstm_model->setNumLayers(3);
                lstm_model->setHiddenSize(128);
                lstm_model->setBidirectional(true);
                lstm_model->setAttentionEnabled(true);
                // FP16 precision is enabled by default in the model constructor
            }
            
            exit_model_->load(exit_model_path);
            std::cout << "Loaded exit model: " << exit_model_path << std::endl;
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

std::vector<std::string> InferenceEngine::fastPathInfer(const data::ParsedMarketData& full_market_data) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check if fast path model is loaded
    if (!fast_path_model_) {
        std::cerr << "Fast path model not loaded" << std::endl;
        return {};
    }
    
    // Store scores for each symbol
    std::vector<std::pair<std::string, float>> symbol_scores;
    symbol_scores.reserve(full_market_data.symbol_data.size());
    
    // Process each symbol in batches
    std::vector<std::vector<float>> feature_batch;
    std::vector<std::string> symbol_batch;
    
    // Prepare batches
    for (const auto& [symbol, data] : full_market_data.symbol_data) {
        // Extract features for this symbol
        std::vector<float> features = extractFastPathFeatures(full_market_data, symbol);
        
        // Add to batch
        feature_batch.push_back(features);
        symbol_batch.push_back(symbol);
        
        // Process batch if full
        if (feature_batch.size() >= static_cast<size_t>(batch_size_)) {
            // Run batch inference
            auto batch_results = fast_path_model_->inferBatch(feature_batch);
            
            // Store results
            for (size_t i = 0; i < batch_results.size(); ++i) {
                // Get score (assuming single output value per symbol)
                float score = batch_results[i][0];
                symbol_scores.emplace_back(symbol_batch[i], score);
            }
            
            // Clear batch
            feature_batch.clear();
            symbol_batch.clear();
        }
    }
    
    // Process remaining symbols
    if (!feature_batch.empty()) {
        auto batch_results = fast_path_model_->inferBatch(feature_batch);
        
        for (size_t i = 0; i < batch_results.size(); ++i) {
            float score = batch_results[i][0];
            symbol_scores.emplace_back(symbol_batch[i], score);
        }
    }
    
    // Filter candidates based on scores
    float threshold = 0.65f;  // Minimum confidence threshold
    int max_candidates = 300; // Maximum number of candidates
    std::vector<std::string> candidates = filterCandidates(symbol_scores, threshold, max_candidates);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    
    // Track performance
    trackInferenceLatency("fast_path", duration);
    
    std::cout << "Fast path inference completed in " << (duration / 1000) << " µs, "
              << "found " << candidates.size() << " candidates out of "
              << full_market_data.symbol_data.size() << " symbols" << std::endl;
    
    return candidates;
}

std::vector<Signal> InferenceEngine::accuratePathInfer(
    const data::ParsedMarketData& selected_data,
    const std::vector<std::string>& candidates) {
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check if accurate path model is loaded
    if (!accurate_path_model_) {
        std::cerr << "Accurate path model not loaded" << std::endl;
        return {};
    }
    
    // Store signals
    std::vector<Signal> signals;
    signals.reserve(candidates.size());
    
    // Process each candidate
    for (const auto& symbol : candidates) {
        // Check if symbol exists in the data
        auto it = selected_data.symbol_data.find(symbol);
        if (it == selected_data.symbol_data.end()) {
            continue;
        }
        
        // Extract features for this symbol
        std::vector<float> features = extractAccuratePathFeatures(selected_data, symbol);
        
        // Run inference
        std::vector<float> outputs = accurate_path_model_->infer(features);
        
        // Skip if no outputs
        if (outputs.empty()) {
            continue;
        }
        
        // Parse outputs (assuming format: [signal_type, confidence, target_price])
        float signal_type_value = outputs[0];
        float confidence = outputs[1];
        float target_price = outputs[2];
        
        // Create signal if confidence is high enough
        if (confidence > 0.7) {
            Signal signal;
            signal.symbol = symbol;
            signal.type = signal_type_value > 0.5 ? SignalType::BUY : SignalType::SELL;
            signal.confidence = confidence;
            signal.price = it->second.last_price;
            signal.timestamp = it->second.timestamp;
            
            // Set pattern information (simplified)
            signal.pattern = static_cast<PatternType>(static_cast<int>(signal_type_value * 10));
            signal.pattern_confidence = confidence;
            
            // Set technical indicators
            signal.indicators["rsi_14"] = it->second.rsi_14;
            signal.indicators["macd"] = it->second.macd;
            signal.indicators["macd_histogram"] = it->second.macd_histogram;
            signal.indicators["bb_position"] = (it->second.last_price - it->second.bb_lower) /
                                             (it->second.bb_upper - it->second.bb_lower);
            
            // Set position sizing and targets
            signal.position_size = 1.0; // Will be adjusted by risk manager
            signal.take_profit = target_price;
            signal.stop_loss = it->second.last_price * 0.98; // Simple 2% stop loss
            
            // Add to signals
            signals.push_back(signal);
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    
    // Track performance
    trackInferenceLatency("accurate_path", duration);
    
    std::cout << "Accurate path inference completed in " << (duration / 1000) << " µs, "
              << "generated " << signals.size() << " signals from "
              << candidates.size() << " candidates" << std::endl;
    
    return signals;
}

std::vector<Signal> InferenceEngine::optimizeExits(
    const std::vector<Signal>& active_positions,
    const data::ParsedMarketData& current_data) {
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check if exit model is loaded
    if (!exit_model_) {
        std::cerr << "Exit model not loaded" << std::endl;
        return {};
    }
    
    // Store exit signals
    std::vector<Signal> exit_signals;
    exit_signals.reserve(active_positions.size());
    
    // Process each active position
    for (const auto& position : active_positions) {
        // Check if symbol exists in the data
        auto it = current_data.symbol_data.find(position.symbol);
        if (it == current_data.symbol_data.end()) {
            continue;
        }
        
        // Extract features for this position
        std::vector<float> features = extractExitFeatures(position, current_data);
        
        // Run inference
        std::vector<float> outputs = exit_model_->infer(features);
        
        // Skip if no outputs
        if (outputs.empty()) {
            continue;
        }
        
        // Parse outputs (assuming format: [exit_probability, optimal_exit_price])
        float exit_probability = outputs[0];
        // Unused: float optimal_exit_price = outputs[1];
        
        // Create exit signal if probability is high enough
        if (exit_probability > 0.7) {
            Signal exit_signal;
            exit_signal.symbol = position.symbol;
            exit_signal.type = SignalType::EXIT;
            exit_signal.confidence = exit_probability;
            exit_signal.price = it->second.last_price;
            exit_signal.timestamp = it->second.timestamp;
            
            // Set technical indicators
            exit_signal.indicators = position.indicators;
            
            // Add to signals
            exit_signals.push_back(exit_signal);
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    
    // Track performance
    trackInferenceLatency("exit_optimization", duration);
    
    std::cout << "Exit optimization completed in " << (duration / 1000) << " µs, "
              << "generated " << exit_signals.size() << " exit signals from "
              << active_positions.size() << " active positions" << std::endl;
    
    return exit_signals;
}

std::vector<float> InferenceEngine::extractFastPathFeatures(
    const data::ParsedMarketData& parsed_data, const std::string& symbol) {
    
    // Find symbol data
    auto it = parsed_data.symbol_data.find(symbol);
    if (it == parsed_data.symbol_data.end()) {
        return {};
    }
    
    const auto& data = it->second;
    
    // Create feature vector (expanded to match architecture document)
    std::vector<float> features;
    features.reserve(22); // Expanded feature set
    
    // Price momentum features
    features.push_back(static_cast<float>(data.last_price / data.open_price));
    features.push_back(static_cast<float>(data.last_price / data.prev_close));
    features.push_back(static_cast<float>(data.last_price / data.vwap));
    features.push_back(static_cast<float>(data.price_change_1m));
    features.push_back(static_cast<float>(data.price_change_5m));
    // Approximate 15m price change as 3x the 5m change (since we don't have direct 15m data)
    features.push_back(static_cast<float>(data.price_change_5m * 3.0));
    
    // Volatility features
    float day_range = data.high_price - data.low_price;
    features.push_back(static_cast<float>((data.last_price - data.low_price) / (day_range > 0 ? day_range : 1.0)));
    features.push_back(static_cast<float>((data.high_price - data.last_price) / (day_range > 0 ? day_range : 1.0)));
    features.push_back(static_cast<float>(data.atr / data.last_price));
    features.push_back(static_cast<float>(data.volatility_ratio));
    
    // Volume-based features
    features.push_back(static_cast<float>(data.volume / data.avg_volume));
    features.push_back(static_cast<float>(data.volume_acceleration));
    features.push_back(static_cast<float>(data.volume_spike));
    features.push_back(static_cast<float>(data.volume_profile_imbalance));
    
    // Technical indicators
    features.push_back(static_cast<float>(data.rsi_14));
    features.push_back(static_cast<float>(data.macd));
    features.push_back(static_cast<float>((data.last_price - data.bb_lower) / (data.bb_upper - data.bb_lower)));
    features.push_back(static_cast<float>(data.momentum_1m));
    
    // Pattern-based features
    features.push_back(static_cast<float>(data.sma_cross_signal));
    features.push_back(static_cast<float>(data.support_resistance_proximity));
    
    // Market context
    features.push_back(static_cast<float>(data.sector_performance));
    features.push_back(static_cast<float>(data.market_regime));
    
    // Ensure we have exactly the number of features expected by the model
    if (features.size() != 22) {
        std::cout << "Warning: Feature vector size mismatch. Expected 22, got "
                  << features.size() << " for symbol " << symbol << std::endl;
    }
    
    return features;
}

std::vector<float> InferenceEngine::extractAccuratePathFeatures(
    const data::ParsedMarketData& parsed_data, const std::string& symbol) {
    
    // Find symbol data
    auto it = parsed_data.symbol_data.find(symbol);
    if (it == parsed_data.symbol_data.end()) {
        return {};
    }
    
    const auto& data = it->second;
    
    // For accurate path, we would use more detailed features including:
    // - Price sequence data
    // - Order book data (if available)
    // - More technical indicators
    
    // This is a simplified implementation
    std::vector<float> features;
    features.reserve(30);  // More features for accurate path
    
    // Include all fast path features
    auto fast_features = extractFastPathFeatures(parsed_data, symbol);
    features.insert(features.end(), fast_features.begin(), fast_features.end());
    
    // Add additional features for accurate path
    // (In a real implementation, these would be more sophisticated)
    features.push_back(static_cast<float>(data.bid_ask_spread / data.last_price));
    features.push_back(static_cast<float>(data.bid_ask_imbalance));
    features.push_back(static_cast<float>(data.trade_count));
    features.push_back(static_cast<float>(data.avg_trade_size));
    features.push_back(static_cast<float>(data.large_trade_ratio));
    features.push_back(static_cast<float>(data.price_trend_strength));
    features.push_back(static_cast<float>(data.volume_trend_strength));
    features.push_back(static_cast<float>(data.market_regime));
    features.push_back(static_cast<float>(data.sector_performance));
    features.push_back(static_cast<float>(data.relative_strength));
    
    return features;
}

std::vector<float> InferenceEngine::extractExitFeatures(
    const Signal& position, const data::ParsedMarketData& current_data) {
    
    // Find symbol data
    auto it = current_data.symbol_data.find(position.symbol);
    if (it == current_data.symbol_data.end()) {
        return {};
    }
    
    const auto& data = it->second;
    
    // Create feature vector for exit optimization
    std::vector<float> features;
    features.reserve(20);
    
    // Position-specific features
    features.push_back(static_cast<float>(data.last_price / position.price));  // Current P&L
    features.push_back(static_cast<float>((current_data.timestamp - position.timestamp) / 60000000000ULL));  // Duration in minutes
    features.push_back(static_cast<float>(position.confidence));  // Initial confidence
    
    // Market condition changes
    features.push_back(static_cast<float>(data.volatility_change));
    features.push_back(static_cast<float>(data.volume / data.avg_volume));
    features.push_back(static_cast<float>(data.bid_ask_spread_change));
    
    // Technical indicator changes
    features.push_back(static_cast<float>(data.rsi_14));
    features.push_back(static_cast<float>(data.macd));
    features.push_back(static_cast<float>(data.macd_histogram));
    features.push_back(static_cast<float>((data.last_price - data.bb_lower) / (data.bb_upper - data.bb_lower)));
    
    // Add more features as needed
    
    return features;
}

std::vector<std::string> InferenceEngine::filterCandidates(
    const std::vector<std::pair<std::string, float>>& scores,
    float threshold,
    int max_candidates) {
    
    // Sort scores in descending order
    std::vector<std::pair<std::string, float>> sorted_scores = scores;
    std::sort(sorted_scores.begin(), sorted_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Filter by threshold and limit
    std::vector<std::string> candidates;
    candidates.reserve(std::min(static_cast<size_t>(max_candidates), sorted_scores.size()));
    
    for (const auto& [symbol, score] : sorted_scores) {
        if (score >= threshold) {
            candidates.push_back(symbol);
            
            if (candidates.size() >= static_cast<size_t>(max_candidates)) {
                break;
            }
        }
    }
    
    return candidates;
}

void InferenceEngine::trackInferenceLatency(const std::string& stage, long duration_ns) {
    // In a real implementation, this would track latency metrics
    // For now, just log to console
    std::cout << "Inference latency for " << stage << ": "
              << (duration_ns / 1000) << " µs" << std::endl;
}

} // namespace ml
} // namespace trading_system