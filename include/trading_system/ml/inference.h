/**
 * ML inference engine for trading signals
 */

#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <unordered_set>
#include <mutex>
#include "trading_system/common/config.h"
#include "trading_system/data/market_data.h"
#include "trading_system/ml/signals.h"
#include "trading_system/ml/model.h"
#include "trading_system/ml/gbdt_model.h"

namespace trading_system {
namespace ml {

class InferenceEngine {
public:
    InferenceEngine(const common::Config& config);
    ~InferenceEngine();
    
    // Run inference on parsed market data (legacy method)
    std::vector<Signal> infer(const data::ParsedMarketData& parsed_data);
    
    // Hybrid approach methods
    
    // Fast path inference for full market scan
    std::vector<std::string> fastPathInfer(const data::ParsedMarketData& full_market_data);
    
    // Accurate path inference for selected symbols
    std::vector<Signal> accuratePathInfer(const data::ParsedMarketData& selected_data,
                                         const std::vector<std::string>& candidates);
    
    // Exit optimization for active positions
    std::vector<Signal> optimizeExits(const std::vector<Signal>& active_positions,
                                     const data::ParsedMarketData& current_data);
    
    // Set thread affinity
    void setThreadAffinity(int core_id);
    
private:
    // ML models - Legacy
    std::unique_ptr<Model> pattern_recognition_model_;
    std::unique_ptr<Model> exit_optimization_model_;
    std::unique_ptr<Model> ranking_model_;
    std::unique_ptr<Model> sentiment_model_;
    
    // ML models - Hybrid approach
    std::unique_ptr<GBDTModel> fast_path_model_;        // Market scanner (LightGBM)
    std::unique_ptr<Model> accurate_path_model_;        // Signal generator (Axial Attention)
    std::unique_ptr<Model> exit_model_;                 // Exit optimization (LSTM/GRU)
    
    // Candidate tracking
    std::unordered_set<std::string> current_candidates_; // Currently tracked symbols
    std::mutex candidates_mutex_;                       // Mutex for thread safety
    
    // Configuration
    int batch_size_;
    bool use_fp16_;
    int max_batch_latency_ms_;
    
    // Thread ID for affinity
    std::thread::id thread_id_;
    
    // Load models
    void loadModels(const common::Config& config);
    
    // Feature extraction methods
    std::vector<float> extractFeatures(const data::ParsedMarketData& parsed_data);
    std::vector<float> extractFastPathFeatures(const data::ParsedMarketData& parsed_data, const std::string& symbol);
    std::vector<float> extractAccuratePathFeatures(const data::ParsedMarketData& parsed_data, const std::string& symbol);
    std::vector<float> extractExitFeatures(const Signal& position, const data::ParsedMarketData& current_data);
    
    // Signal generation methods
    std::vector<Signal> generateSignals(
        const std::vector<float>& pattern_outputs,
        const std::vector<float>& ranking_outputs,
        const data::ParsedMarketData& parsed_data
    );
    
    // Filtering methods
    std::vector<std::string> filterCandidates(
        const std::vector<std::pair<std::string, float>>& scores,
        float threshold,
        int max_candidates
    );
    
    // Performance tracking
    void trackInferenceLatency(const std::string& stage, long duration_ns);
};

} // namespace ml
} // namespace trading_system