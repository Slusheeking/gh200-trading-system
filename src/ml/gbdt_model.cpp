/**
 * GBDT model implementation for fast path inference
 * Direct LightGBM integration without ONNX
 */

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include "trading_system/ml/gbdt_model.h"
#include "trading_system/ml/model.h"

// LightGBM C API
extern "C" {
    #include <LightGBM/c_api.h>
}

namespace fs = std::filesystem;

namespace trading_system {
namespace ml {

GBDTModel::GBDTModel() : booster_(nullptr) {
    std::cout << "Creating GBDT model" << std::endl;
    input_shape_ = {1, 20};  // Default: batch size, 20 features
    output_shape_ = {1, 1};  // Default: batch size, 1 output (binary classification)
    
    // Initialize with architecture parameters from GH200_Trading_System_Architecture.md
    num_trees_ = 150;
    max_depth_ = 8;
    learning_rate_ = 0.05;
    feature_fraction_ = 0.8;
    bagging_fraction_ = 0.7;
}

GBDTModel::~GBDTModel() {
    std::cout << "Destroying GBDT model" << std::endl;
    
    // Free LightGBM booster if allocated
    if (booster_ != nullptr) {
        LGBM_BoosterFree(booster_);
        booster_ = nullptr;
    }
}

void GBDTModel::load(const std::string& model_path) {
    std::cout << "Loading GBDT model from " << model_path << std::endl;
    
    // Store the model path
    model_path_ = model_path;
    
    // Check if file exists
    if (!fs::exists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path);
    }
    
    // Load the LightGBM model
    loadBooster(model_path);
    
    // Extract model metadata
    extractModelMetadata();
    
    is_loaded_ = true;
}

void GBDTModel::loadBooster(const std::string& model_path) {
    // Free existing booster if any
    if (booster_ != nullptr) {
        LGBM_BoosterFree(booster_);
        booster_ = nullptr;
    }
    
    // Load model from file
    int out_num_iterations = 0;
    int result = LGBM_BoosterCreateFromModelfile(
        model_path.c_str(),
        &out_num_iterations,
        &booster_
    );
    
    if (result != 0 || booster_ == nullptr) {
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to load LightGBM model: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    std::cout << "Loaded LightGBM model with " << out_num_iterations << " iterations" << std::endl;
    
    // Apply model parameters
    if (booster_ != nullptr) {
        // Set number of threads
        std::string thread_param = "num_threads=" + std::to_string(num_threads_);
        LGBM_BoosterResetParameter(booster_, thread_param.c_str());
        
        // Set learning rate
        std::string lr_param = "learning_rate=" + std::to_string(learning_rate_);
        LGBM_BoosterResetParameter(booster_, lr_param.c_str());
        
        // Set feature fraction
        std::string ff_param = "feature_fraction=" + std::to_string(feature_fraction_);
        LGBM_BoosterResetParameter(booster_, ff_param.c_str());
        
        // Set bagging fraction
        std::string bf_param = "bagging_fraction=" + std::to_string(bagging_fraction_);
        LGBM_BoosterResetParameter(booster_, bf_param.c_str());
        
        std::cout << "Applied GBDT model parameters: trees=" << num_trees_
                  << ", depth=" << max_depth_
                  << ", learning_rate=" << learning_rate_
                  << ", feature_fraction=" << feature_fraction_
                  << ", bagging_fraction=" << bagging_fraction_ << std::endl;
    }
}

void GBDTModel::extractModelMetadata() {
    if (booster_ == nullptr) {
        throw std::runtime_error("Booster not initialized");
    }
    
    // Get number of features
    int out_len = 0;
    int result = LGBM_BoosterGetNumFeature(booster_, &out_len);
    if (result != 0) {
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to get number of features: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    // Update input shape
    input_shape_[1] = out_len;
    
    // Get feature names if not already set
    if (feature_names_.empty() && out_len > 0) {
        feature_names_.resize(out_len);
        for (int i = 0; i < out_len; i++) {
            feature_names_[i] = "feature_" + std::to_string(i);
        }
    }
    
    // Get feature importance
    std::vector<double> importance(out_len);
    int64_t out_buffer_len = out_len;
    result = LGBM_BoosterFeatureImportance(
        booster_,
        0,  // Importance type: 0 for split, 1 for gain
        0,  // Iteration: 0 for all
        importance.data(),
        &out_buffer_len
    );
    
    if (result != 0) {
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to get feature importance: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    // Store feature importance
    feature_importance_.clear();
    for (size_t i = 0; i < feature_names_.size() && i < importance.size(); i++) {
        feature_importance_[feature_names_[i]] = static_cast<float>(importance[i]);
    }
}

std::vector<float> GBDTModel::infer(const std::vector<float>& features) {
    if (!is_loaded_ || booster_ == nullptr) {
        throw std::runtime_error("Model not loaded");
    }
    
    // Check input size
    if (features.size() != static_cast<size_t>(input_shape_[1])) {
        throw std::runtime_error("Input size mismatch: expected " +
                                std::to_string(input_shape_[1]) +
                                ", got " + std::to_string(features.size()));
    }
    
    // Create data matrix
    void* data_handle = nullptr;
    int result = LGBM_DatasetCreateFromMat(
        features.data(),
        C_API_DTYPE_FLOAT32,
        1,  // Number of rows
        features.size(),  // Number of columns
        1,  // Row-major (CSR) format
        nullptr,  // Reference dataset
        &data_handle
    );
    
    if (result != 0 || data_handle == nullptr) {
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to create dataset: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    // Run prediction
    int64_t out_len = 0;
    result = LGBM_BoosterPredictForMatSingleRow(
        booster_,
        features.data(),
        C_API_DTYPE_FLOAT32,
        output_shape_[1],  // Number of output classes
        1,  // Prediction type: 0 for normal, 1 for raw score, 2 for leaf index
        0,  // Start iteration: 0 for all
        0,  // Number of iterations: 0 for all
        "",  // Parameter string
        &out_len,
        nullptr  // Get output length first
    );
    
    if (result != 0) {
        LGBM_DatasetFree(data_handle);
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to get prediction length: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    // Allocate output buffer
    std::vector<double> out_result(out_len);
    
    // Get prediction
    result = LGBM_BoosterPredictForMatSingleRow(
        booster_,
        features.data(),
        C_API_DTYPE_FLOAT32,
        output_shape_[1],  // Number of output classes
        1,  // Prediction type: 0 for normal, 1 for raw score, 2 for leaf index
        0,  // Start iteration: 0 for all
        0,  // Number of iterations: 0 for all
        "",  // Parameter string
        &out_len,
        out_result.data()
    );
    
    // Free data matrix
    LGBM_DatasetFree(data_handle);
    
    if (result != 0) {
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to get prediction: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    // Convert to float
    std::vector<float> output(out_len);
    for (size_t i = 0; i < out_len; i++) {
        output[i] = static_cast<float>(out_result[i]);
    }
    
    return output;
}

std::vector<std::vector<float>> GBDTModel::inferBatch(
    const std::vector<std::vector<float>>& features_batch) {
    
    if (!is_loaded_ || booster_ == nullptr) {
        throw std::runtime_error("Model not loaded");
    }
    
    // Check batch size
    if (features_batch.empty()) {
        return {};
    }
    
    // Check input dimensions
    size_t num_features = features_batch[0].size();
    for (size_t i = 1; i < features_batch.size(); i++) {
        if (features_batch[i].size() != num_features) {
            throw std::runtime_error("Inconsistent feature dimensions in batch");
        }
    }
    
    // Flatten features for batch processing
    std::vector<float> flat_features;
    flat_features.reserve(features_batch.size() * num_features);
    
    for (const auto& features : features_batch) {
        flat_features.insert(flat_features.end(), features.begin(), features.end());
    }
    
    // Create data matrix
    void* data_handle = nullptr;
    int result = LGBM_DatasetCreateFromMat(
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        features_batch.size(),  // Number of rows
        num_features,  // Number of columns
        1,  // Row-major (CSR) format
        nullptr,  // Reference dataset
        &data_handle
    );
    
    if (result != 0 || data_handle == nullptr) {
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to create dataset: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    // Run prediction
    int64_t out_len = 0;
    result = LGBM_BoosterPredictForMat(
        booster_,
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        features_batch.size(),  // Number of rows
        num_features,  // Number of columns
        1,  // Row-major (CSR) format
        1,  // Prediction type: 0 for normal, 1 for raw score, 2 for leaf index
        0,  // Start iteration: 0 for all
        0,  // Number of iterations: 0 for all
        "",  // Parameter string
        &out_len,
        nullptr  // Get output length first
    );
    
    if (result != 0) {
        LGBM_DatasetFree(data_handle);
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to get prediction length: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    // Allocate output buffer
    std::vector<double> out_result(out_len);
    
    // Get prediction
    result = LGBM_BoosterPredictForMat(
        booster_,
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        features_batch.size(),  // Number of rows
        num_features,  // Number of columns
        1,  // Row-major (CSR) format
        1,  // Prediction type: 0 for normal, 1 for raw score, 2 for leaf index
        0,  // Start iteration: 0 for all
        0,  // Number of iterations: 0 for all
        "",  // Parameter string
        &out_len,
        out_result.data()
    );
    
    // Free data matrix
    LGBM_DatasetFree(data_handle);
    
    if (result != 0) {
        const char* err_msg = LGBM_GetLastError();
        throw std::runtime_error("Failed to get prediction: " +
                                std::string(err_msg ? err_msg : "unknown error"));
    }
    
    // Reshape output
    size_t output_size = out_len / features_batch.size();
    std::vector<std::vector<float>> output(features_batch.size());
    
    for (size_t i = 0; i < features_batch.size(); i++) {
        output[i].resize(output_size);
        for (size_t j = 0; j < output_size; j++) {
            output[i][j] = static_cast<float>(out_result[i * output_size + j]);
        }
    }
    
    return output;
}

std::string GBDTModel::getName() const {
    return "GBDT Model (LightGBM - Fast Path)";
}

std::vector<int> GBDTModel::getInputShape() const {
    return input_shape_;
}

std::vector<int> GBDTModel::getOutputShape() const {
    return output_shape_;
}

void GBDTModel::setNumTrees(int num_trees) {
    num_trees_ = num_trees;
}

void GBDTModel::setMaxDepth(int max_depth) {
    max_depth_ = max_depth;
}

void GBDTModel::setFeatureNames(const std::vector<std::string>& feature_names) {
    feature_names_ = feature_names;
}

void GBDTModel::setNumThreads(int num_threads) {
    num_threads_ = num_threads;
    
    // Set number of threads for LightGBM
    if (booster_ != nullptr) {
        std::string param = "num_threads=" + std::to_string(num_threads);
        LGBM_BoosterResetParameter(booster_, param.c_str());
    }
}

void GBDTModel::setLearningRate(float learning_rate) {
    learning_rate_ = learning_rate;
    
    // Set learning rate for LightGBM
    if (booster_ != nullptr) {
        std::string param = "learning_rate=" + std::to_string(learning_rate);
        LGBM_BoosterResetParameter(booster_, param.c_str());
    }
}

void GBDTModel::setFeatureFraction(float feature_fraction) {
    feature_fraction_ = feature_fraction;
    
    // Set feature fraction for LightGBM
    if (booster_ != nullptr) {
        std::string param = "feature_fraction=" + std::to_string(feature_fraction);
        LGBM_BoosterResetParameter(booster_, param.c_str());
    }
}

void GBDTModel::setBaggingFraction(float bagging_fraction) {
    bagging_fraction_ = bagging_fraction;
    
    // Set bagging fraction for LightGBM
    if (booster_ != nullptr) {
        std::string param = "bagging_fraction=" + std::to_string(bagging_fraction);
        LGBM_BoosterResetParameter(booster_, param.c_str());
    }
}

std::unordered_map<std::string, float> GBDTModel::getFeatureImportance() const {
    return feature_importance_;
}

} // namespace ml
} // namespace trading_system