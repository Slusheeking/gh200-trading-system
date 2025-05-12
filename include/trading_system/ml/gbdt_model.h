/**
 * GBDT model implementation for fast path inference
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
 * Gradient Boosting Decision Tree model implementation
 * Uses LightGBM directly for fast inference
 */
class GBDTModel : public Model {
public:
    GBDTModel();
    ~GBDTModel() override;
    
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
    
    // GBDT specific methods
    void setNumTrees(int num_trees);
    void setMaxDepth(int max_depth);
    void setFeatureNames(const std::vector<std::string>& feature_names);
    void setNumThreads(int num_threads);
    void setLearningRate(float learning_rate);
    void setFeatureFraction(float feature_fraction);
    void setBaggingFraction(float bagging_fraction);
    
    // Feature importance
    std::unordered_map<std::string, float> getFeatureImportance() const;
    
private:
    // Implementation details
    void* booster_ = nullptr; // LightGBM booster handle
    std::string model_path_;
    int num_trees_ = 150;
    int max_depth_ = 8;
    int num_threads_ = 4;
    float learning_rate_ = 0.05;
    float feature_fraction_ = 0.8;
    float bagging_fraction_ = 0.7;
    std::vector<std::string> feature_names_;
    std::unordered_map<std::string, float> feature_importance_;
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    bool is_loaded_ = false;
    
    // Helper methods
    void loadBooster(const std::string& model_path);
    void extractModelMetadata();
};

} // namespace ml
} // namespace trading_system
