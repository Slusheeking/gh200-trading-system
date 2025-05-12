/**
 * Model Trainer for Trading System
 * Trains and exports models for the hybrid HFT architecture
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "trading_system/common/config.h"
#include "trading_system/ml/model.h"
#include "trading_system/ml/gbdt_model.h"
#include "trading_system/ml/axial_attention_model.h"
#include "trading_system/ml/lstm_gru_model.h"

namespace trading_system {
namespace ml {

/**
 * @class ModelTrainer
 * @brief Trains and exports models for the trading system
 */
class ModelTrainer {
public:
    /**
     * @brief Constructor
     * @param config System configuration
     */
    ModelTrainer(const common::Config& config);

    /**
     * @brief Destructor
     */
    ~ModelTrainer();

    /**
     * @brief Train a fast path model (GBDT/LightGBM)
     * @param data_path Path to training data
     * @param output_path Path to save the model
     * @return True if training was successful
     */
    bool trainFastPathModel(const std::string& data_path, const std::string& output_path);

    /**
     * @brief Train an accurate path model (Axial Attention)
     * @param data_path Path to training data
     * @param output_path Path to save the model
     * @return True if training was successful
     */
    bool trainAccuratePathModel(const std::string& data_path, const std::string& output_path);

    /**
     * @brief Train an exit optimization model (LSTM/GRU)
     * @param data_path Path to training data
     * @param output_path Path to save the model
     * @return True if training was successful
     */
    bool trainExitOptimizationModel(const std::string& data_path, const std::string& output_path);

    /**
     * @brief Load training data from CSV
     * @param data_path Path to CSV file
     * @param feature_cols Feature column names
     * @param label_cols Label column names
     * @param test_split Test split ratio (0.0-1.0)
     * @return True if data was loaded successfully
     */
    bool loadTrainingData(const std::string& data_path, 
                         const std::vector<std::string>& feature_cols,
                         const std::vector<std::string>& label_cols,
                         float test_split = 0.2);

    /**
     * @brief Export model to file
     * @param model Model to export
     * @param output_path Path to save the model
     * @return True if export was successful
     */
    bool exportModel(std::shared_ptr<Model> model, const std::string& output_path);

    /**
     * @brief Evaluate model performance
     * @param model Model to evaluate
     * @param metrics Output metrics
     * @return True if evaluation was successful
     */
    bool evaluateModel(std::shared_ptr<Model> model, 
                      std::unordered_map<std::string, float>& metrics);

    /**
     * @brief Set GPU device to use for training
     * @param device_id GPU device ID
     */
    void setGpuDevice(int device_id);

    /**
     * @brief Enable or disable FP16 precision
     * @param enable True to enable FP16, false to use FP32
     */
    void enableFP16(bool enable);

private:
    // Configuration
    common::Config config_;
    bool use_gpu_;
    bool use_fp16_;
    int gpu_device_id_;
    
    // Training data
    std::vector<std::vector<float>> train_features_;
    std::vector<std::vector<float>> train_labels_;
    std::vector<std::vector<float>> test_features_;
    std::vector<std::vector<float>> test_labels_;
    std::vector<std::string> feature_names_;
    
    // Model parameters
    std::unordered_map<std::string, std::unordered_map<std::string, float>> model_params_;
    
    // Helper methods
    /**
     * @brief Initialize model parameters from configuration
     */
    void initializeModelParameters();
    
    bool loadCsvData(const std::string& file_path,
                     std::vector<std::vector<float>>& features,
                     std::vector<std::vector<float>>& labels,
                     const std::vector<std::string>& feature_cols,
                     const std::vector<std::string>& label_cols);
    
    void splitTrainTest(const std::vector<std::vector<float>>& features,
                       const std::vector<std::vector<float>>& labels,
                       std::vector<std::vector<float>>& train_features,
                       std::vector<std::vector<float>>& train_labels,
                       std::vector<std::vector<float>>& test_features,
                       std::vector<std::vector<float>>& test_labels,
                       float test_split);
};

} // namespace ml
} // namespace trading_system