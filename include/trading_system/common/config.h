/**
 * Configuration management for trading system
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <yaml-cpp/yaml.h>

namespace trading_system {
namespace common {

// Hardware configuration
struct HardwareConfig {
    std::string device;
    int memory_limit_mb;
    int num_cuda_streams;
    bool use_tensor_cores;
    
    struct CpuCores {
        int main_thread;
        int websocket;
        int inference;
        int risk;
        int execution;
        int monitoring;
    } cpu_cores;
    
    bool use_huge_pages;
    bool preallocate_memory;
    int preallocate_size_mb;
    
    int receive_buffer_size;
    int send_buffer_size;
    bool tcp_nodelay;
};

// Data source configuration
struct DataSourceConfig {
    bool enabled;
    std::string api_key;
    std::string api_secret;
    std::string base_url;
    std::string websocket_url;
    int reconnect_interval_ms;
    std::string subscription_type;
};

// ML configuration
struct MLConfig {
    // Path to model configuration file
    std::string config_file;
    
    struct ModelPaths {
        // Legacy models
        std::string pattern_recognition;
        std::string exit_optimization;
        std::string ranking;
        std::string sentiment;
        
        // Hybrid approach models
        std::string fast_path;       // Market scanner (GBDT/LightGBM)
        std::string accurate_path;   // Signal generator (Axial Attention)
        std::string exit_model;      // Exit optimization (LSTM/GRU)
        
        // Engine files (optimized)
        std::string fast_path_engine;
        std::string accurate_path_engine;
        std::string exit_model_engine;
    } model_paths;
    
    struct Inference {
        int batch_size;
        bool use_fp16;
        int max_batch_latency_ms;
        int inference_threads;
        
        // Hybrid approach settings
        float fast_path_threshold;
        int max_candidates;
        float accurate_path_threshold;
        bool use_tensorrt;
        std::string tensorrt_cache_path;
        
        // Model-specific settings
        struct GBDT {
            int num_threads;
            std::string prediction_type;
        } gbdt;
        
        struct AxialAttention {
            int num_heads;
            int head_dim;
            int num_layers;
            int seq_length;
        } axial_attention;
        
        struct LstmGru {
            int num_layers;
            int hidden_size;
            bool bidirectional;
            bool attention_enabled;
        } lstm_gru;
    } inference;
    
    struct Features {
        int lookback_periods;
        bool use_cuda_extraction;
        std::vector<std::string> indicators;
        std::vector<std::string> fast_path_features;
    } features;
};

// Performance configuration
struct PerformanceConfig {
    int max_e2e_latency_us;
    int max_inference_latency_us;
    int max_execution_latency_us;
    
    bool latency_monitoring;
    int latency_log_interval_s;
    bool performance_profiling;
    
    bool use_lock_free_queues;
    bool use_zero_copy;
    int websocket_parser_batch_size;
};

// Logging configuration
struct LoggingConfig {
    std::string level;
    std::string file;
    int max_file_size_mb;
    int max_files;
    std::string log_format;
    
    bool log_latency_stats;
    bool log_memory_usage;
    
    bool use_zero_alloc_logging;
    int preallocated_log_buffer_size;
    int flush_interval_ms;
    
    int latency_log_interval_s;
};

// Trading configuration
struct TradingConfig {
    struct Account {
        double initial_capital;
        int max_positions;
        std::string base_currency;
        std::string account_type;
        double leverage;
    } account;
    
    struct Risk {
        double max_position_size_pct;
        std::string position_sizing_method;
        double kelly_fraction;
        double max_daily_drawdown_pct;
        bool use_stop_loss;
        double default_stop_loss_pct;
        bool use_take_profit;
        double default_take_profit_pct;
        bool use_trailing_stop;
        double trailing_stop_pct;
    } risk;
    
    struct Orders {
        std::string default_order_type;
        bool use_bracket_orders;
        std::string time_in_force;
        std::string broker_type; // Add broker_type here
    } orders;
    
    struct Strategies {
        bool pattern_recognition_enabled;
        double confidence_threshold;
        std::vector<std::string> patterns;
    } strategies;
    
    struct Exit {
        int max_holding_time_minutes;
        int check_exit_interval_seconds;
        bool use_ml_exit;
        double exit_confidence_threshold;
        bool close_positions_eod;
        std::string eod_exit_time;
    } exit;
};

class Config {
public:
    Config(const std::string& config_path);
    ~Config() = default;
    
    const HardwareConfig& getHardwareConfig() const { return hardware_config_; }
    const DataSourceConfig& getDataSourceConfig(const std::string& source) const;
    const MLConfig& getMLConfig() const { return ml_config_; }
    const PerformanceConfig& getPerformanceConfig() const { return performance_config_; }
    const LoggingConfig& getLoggingConfig() const { return logging_config_; }
    const TradingConfig& getTradingConfig() const { return trading_config_; }
    
    // Load model-specific configuration from ml_config_.config_file
    void loadModelConfig();
    
private:
    void loadConfig(const std::string& path);
    std::string expandEnvVars(const std::string& value);
    
    HardwareConfig hardware_config_;
    std::unordered_map<std::string, DataSourceConfig> data_sources_;
    MLConfig ml_config_;
    PerformanceConfig performance_config_;
    LoggingConfig logging_config_;
    TradingConfig trading_config_;
};

// Thread affinity utility
void pinThreadToCore(int core_id);

} // namespace common
} // namespace trading_system