/**
 * LSTM/GRU model implementation for exit optimization
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "trading_system/ml/model.h"

namespace trading_system {
namespace ml {

/**
 * LSTM/GRU model implementation for exit optimization
 * Uses TensorRT for optimized inference
 */
class LstmGruModel : public Model {
public:
    LstmGruModel();
    ~LstmGruModel() override;
    
    // Load model from file
    void load(const std::string& model_path) override;
    
    // Run inference
    std::vector<float> infer(const std::vector<float>& features) override;
    
    // Run batch inference
    std::vector<std::vector<float>> inferBatch(
        const std::vector<std::vector<float>>& features_batch) override;
    
    // Get model name
    std::string getName() const override;
    
    // Get input shape
    std::vector<int> getInputShape() const override;
    
    // Get output shape
    std::vector<int> getOutputShape() const override;
    
    // LSTM/GRU specific methods
    void setNumLayers(int num_layers);
    void setHiddenSize(int hidden_size);
    void setBidirectional(bool bidirectional);
    void setAttentionEnabled(bool enable_attention);
    
private:
    // Implementation details
    void* engine_ = nullptr;       // TensorRT engine
    void* context_ = nullptr;      // TensorRT execution context
    void* cuda_stream_ = nullptr;  // CUDA stream
    
    std::string model_path_;
    int num_layers_ = 3;
    int hidden_size_ = 128;
    bool bidirectional_ = true;
    bool attention_enabled_ = true;
    bool use_fp16_ = true;
    
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    bool is_loaded_ = false;
    
    // Helper methods
    void initializeEngine();
    void configureModel();
    
    // Reinforcement learning state
    std::unordered_map<std::string, std::vector<float>> position_states_;
};

} // namespace ml
} // namespace trading_system