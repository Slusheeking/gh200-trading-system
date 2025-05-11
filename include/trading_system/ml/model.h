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

// PyTorch model implementation
class PyTorchModel : public Model {
public:
    PyTorchModel();
    ~PyTorchModel() override;
    
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
    
private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ONNX model implementation
class ONNXModel : public Model {
public:
    ONNXModel();
    ~ONNXModel() override;
    
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
    
private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ml
} // namespace trading_system