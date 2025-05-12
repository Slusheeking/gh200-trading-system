/**
 * Model Trainer Main
 * Command-line application to train models for the trading system
 */

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <filesystem>
#include "trading_system/common/config.h"
#include "trading_system/ml/model_trainer.h"

namespace fs = std::filesystem;

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model-type TYPE     Model type to train (fast_path, accurate_path, exit_optimization)" << std::endl;
    std::cout << "  --data-path PATH      Path to training data directory" << std::endl;
    std::cout << "  --output-dir PATH     Output directory for trained models" << std::endl;
    std::cout << "  --config PATH         Path to configuration file (default: config/system.yaml)" << std::endl;
    std::cout << "  --gpu-id ID           GPU device ID to use (default: 0)" << std::endl;
    std::cout << "  --use-fp16            Use FP16 precision for training (default: true)" << std::endl;
    std::cout << "  --help                Display this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default values
    std::string model_type;
    std::string data_path;
    std::string output_dir = "models";
    std::string config_path = "config/system.yaml";
    int gpu_id = 0;
    bool use_fp16 = true;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--model-type" && i + 1 < argc) {
            model_type = argv[++i];
        } else if (arg == "--data-path" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--gpu-id" && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
        } else if (arg == "--use-fp16") {
            use_fp16 = true;
        } else if (arg == "--no-fp16") {
            use_fp16 = false;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Check required arguments
    if (model_type.empty()) {
        std::cerr << "Error: Model type is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    if (data_path.empty()) {
        std::cerr << "Error: Data path is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // Validate model type
    if (model_type != "fast_path" && model_type != "accurate_path" && model_type != "exit_optimization") {
        std::cerr << "Error: Invalid model type: " << model_type << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        // Load configuration
        std::cout << "Loading configuration from " << config_path << std::endl;
        trading_system::common::Config config(config_path);
        
        // Create model trainer
        std::cout << "Initializing model trainer..." << std::endl;
        trading_system::ml::ModelTrainer trainer(config);
        
        // Set GPU device
        trainer.setGpuDevice(gpu_id);
        
        // Set precision
        trainer.enableFP16(use_fp16);
        
        // Create output directory if it doesn't exist
        fs::create_directories(output_dir);
        
        // Train model
        bool success = false;
        
        if (model_type == "fast_path") {
            std::cout << "Training Fast Path model..." << std::endl;
            std::string model_path = output_dir + "/fast_path_model.txt";
            success = trainer.trainFastPathModel(data_path, model_path);
        } else if (model_type == "accurate_path") {
            std::cout << "Training Accurate Path model..." << std::endl;
            std::string model_path = output_dir + "/accurate_path_model.pt";
            success = trainer.trainAccuratePathModel(data_path, model_path);
        } else if (model_type == "exit_optimization") {
            std::cout << "Training Exit Optimization model..." << std::endl;
            std::string model_path = output_dir + "/exit_optimization_model.pt";
            success = trainer.trainExitOptimizationModel(data_path, model_path);
        }
        
        if (success) {
            std::cout << "Model training completed successfully!" << std::endl;
            return 0;
        } else {
            std::cerr << "Model training failed" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}