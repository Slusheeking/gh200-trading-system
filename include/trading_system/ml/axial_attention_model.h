/**
 * Axial Attention model implementation for accurate path inference
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
 * Axial Attention model implementation
 * Uses TensorRT for optimized inference
 */
class AxialAttentionModel : public Model {
public:
    AxialAttentionModel();
    ~AxialAttentionModel() override;
    
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
    
    // Axial Attention specific methods
    void setNumHeads(int num_heads);
    void setHeadDimension(int head_dim);
    void setNumLayers(int num_layers);
    void setSequenceLength(int seq_length);
    void setDropout(float dropout);
    
private:
    // Implementation details
    void* engine_ = nullptr;       // TensorRT engine
    void* context_ = nullptr;      // TensorRT execution context
    void* cuda_stream_ = nullptr;  // CUDA stream
    
    std::string model_path_;
    int num_heads_ = 4;
    int head_dim_ = 64;
    int num_layers_ = 6;
    int seq_length_ = 100;
    float dropout_ = 0.1;
    bool use_fp16_ = true;
    
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    bool is_loaded_ = false;
    
    // Helper methods
    void initializeEngine();
    void configureModel();
};

} // namespace ml
} // namespace trading_system