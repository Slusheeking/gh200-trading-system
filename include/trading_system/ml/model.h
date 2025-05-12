/**
 * ML model interface
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace trading_system {
namespace ml {

// Model interface
class Model {
public:
    virtual ~Model() = default;
    
    // Load model from file
    virtual void load(const std::string& model_path) = 0;
    
    // Run inference
    virtual std::vector<float> infer(const std::vector<float>& features) = 0;
    
    // Run batch inference
    virtual std::vector<std::vector<float>> inferBatch(
        const std::vector<std::vector<float>>& features_batch) = 0;
    
    // Get model name
    virtual std::string getName() const = 0;
    
    // Get input shape
    virtual std::vector<int> getInputShape() const = 0;
    
    // Get output shape
    virtual std::vector<int> getOutputShape() const = 0;
    
    // Factory method to create model based on type
    static std::unique_ptr<Model> create(const std::string& model_type);
};

// LightGBM model implementation
class LightGBMModel : public Model {
public:
    LightGBMModel();
    ~LightGBMModel() override;
    
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
    
    // LightGBM specific methods
    void setNumThreads(int num_threads);
    void setPredictionType(const std::string& pred_type); // "raw", "probability"
    
private:
    // Implementation details
    int num_threads_ = 4;
    std::string prediction_type_ = "probability";
    int num_features_ = 0;
    int num_classes_ = 0;
    void* booster_ = nullptr; // LightGBM booster handle
};

// TensorRT model implementation (direct integration without ONNX)
class TensorRTModel : public Model {
public:
    TensorRTModel();
    ~TensorRTModel() override;
    
    // Load model from file (supports .plan or .engine files)
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
    
    // TensorRT specific methods
    void setOptimizationLevel(int level);
    void enableFP16(bool enable);
    void enableFP8(bool enable);  // Added FP8 precision support
    void setMaxBatchSize(int batch_size);
    void setMaxWorkspaceSize(size_t size);
    
    // Create engine from serialized plan file
    void loadEngine(const std::string& engine_path);
    
    // Save engine to file
    void saveEngine(const std::string& engine_path);
    
private:
    // Implementation details
    int optimization_level_ = 2;
    bool use_fp16_ = true;
    bool use_fp8_ = false;  // FP8 precision flag
    int max_batch_size_ = 64;
    size_t max_workspace_size_ = 4ULL * 1024 * 1024 * 1024; // 4GB
    std::string engine_cache_path_;
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    void* runtime_ = nullptr;      // TensorRT runtime
    void* engine_ = nullptr;       // TensorRT engine
    void* context_ = nullptr;      // TensorRT execution context
    void* cuda_stream_ = nullptr;  // CUDA stream
};

} // namespace ml
} // namespace trading_system