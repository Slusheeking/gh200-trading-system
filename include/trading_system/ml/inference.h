/**
 * ML inference engine for trading signals
 */

#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include "trading_system/common/config.h"
#include "trading_system/data/market_data.h"
#include "trading_system/ml/signals.h"
#include "trading_system/ml/model.h"

namespace trading_system {
namespace ml {

class InferenceEngine {
public:
    InferenceEngine(const common::Config& config);
    ~InferenceEngine();
    
    // Run inference on parsed market data
    std::vector<Signal> infer(const data::ParsedMarketData& parsed_data);
    
    // Set thread affinity
    void setThreadAffinity(int core_id);
    
private:
    // ML models
    std::unique_ptr<Model> pattern_recognition_model_;
    std::unique_ptr<Model> exit_optimization_model_;
    std::unique_ptr<Model> ranking_model_;
    std::unique_ptr<Model> sentiment_model_;
    
    // Configuration
    int batch_size_;
    bool use_fp16_;
    int max_batch_latency_ms_;
    
    // Thread ID for affinity
    std::thread::id thread_id_;
    
    // Load models
    void loadModels(const common::Config& config);
    
    // Extract features from market data
    std::vector<float> extractFeatures(const data::ParsedMarketData& parsed_data);
    
    // Generate signals from model outputs
    std::vector<Signal> generateSignals(
        const std::vector<float>& pattern_outputs,
        const std::vector<float>& ranking_outputs,
        const data::ParsedMarketData& parsed_data
    );
};

} // namespace ml
} // namespace trading_system