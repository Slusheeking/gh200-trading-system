/**
 * Configuration management implementation
 */

#include <regex>
#include <stdexcept>

#include <yaml-cpp/yaml.h>
#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include <cstdlib>
#include <pthread.h>

namespace trading_system {
namespace common {

Config::Config(const std::string& system_config_path, const std::string& trading_config_path) {
    // Load system configuration
    loadSystemConfig(system_config_path);
    
    // Load trading configuration
    loadTradingConfig(trading_config_path);
    
    LOG_INFO("Configuration loaded from " + system_config_path + " and " + trading_config_path);
}

const DataSourceConfig& Config::getDataSourceConfig(const std::string& source) const {
    auto it = data_sources_.find(source);
    if (it == data_sources_.end()) {
        static DataSourceConfig empty_config;
        return empty_config;
    }
    
    return it->second;
}

void Config::loadSystemConfig(const std::string& path) {
    try {
        // Load YAML file
        YAML::Node config = YAML::LoadFile(path);
        
        // Parse hardware configuration
        if (config["hardware"]) {
            auto hw = config["hardware"];
            hardware_config_.device = hw["device"].as<std::string>("cuda:0");
            hardware_config_.memory_limit_mb = hw["memory_limit_mb"].as<int>(32000);
            hardware_config_.num_cuda_streams = hw["num_cuda_streams"].as<int>(4);
            hardware_config_.use_tensor_cores = hw["use_tensor_cores"].as<bool>(true);
            
            // Parse CPU cores
            if (hw["cpu_cores"]) {
                hardware_config_.cpu_cores.main_thread = hw["cpu_cores"]["main_thread"].as<int>(2);
                hardware_config_.cpu_cores.websocket = hw["cpu_cores"]["websocket"].as<int>(3);
                hardware_config_.cpu_cores.inference = hw["cpu_cores"]["inference"].as<int>(4);
                hardware_config_.cpu_cores.risk = hw["cpu_cores"]["risk"].as<int>(5);
                hardware_config_.cpu_cores.execution = hw["cpu_cores"]["execution"].as<int>(6);
                hardware_config_.cpu_cores.monitoring = hw["cpu_cores"]["monitoring"].as<int>(7);
            }
            
            hardware_config_.use_huge_pages = hw["use_huge_pages"].as<bool>(true);
            hardware_config_.preallocate_memory = hw["preallocate_memory"].as<bool>(true);
            hardware_config_.preallocate_size_mb = hw["preallocate_size_mb"].as<int>(16000);
            
            hardware_config_.receive_buffer_size = hw["receive_buffer_size"].as<int>(16777216);
            hardware_config_.send_buffer_size = hw["send_buffer_size"].as<int>(8388608);
            hardware_config_.tcp_nodelay = hw["tcp_nodelay"].as<bool>(true);
        }
        
        // Parse data sources
        if (config["data_sources"]) {
            auto ds = config["data_sources"];
            
            // Parse Polygon configuration
            if (ds["polygon"]) {
                auto polygon = ds["polygon"];
                DataSourceConfig polygon_config;
                polygon_config.enabled = polygon["enabled"].as<bool>(true);
                polygon_config.api_key = expandEnvVars(polygon["api_key"].as<std::string>(""));
                polygon_config.websocket_url = polygon["websocket_url"].as<std::string>("wss://socket.polygon.io/stocks");
                polygon_config.base_url = polygon["rest_url"].as<std::string>("https://api.polygon.io/v2");
                polygon_config.reconnect_interval_ms = polygon["reconnect_interval_ms"].as<int>(5000);
                polygon_config.subscription_type = polygon["subscription_type"].as<std::string>("T.*");
                
                data_sources_["polygon"] = polygon_config;
            }
            
            // Parse Alpaca configuration
            if (ds["alpaca"]) {
                auto alpaca = ds["alpaca"];
                DataSourceConfig alpaca_config;
                alpaca_config.enabled = alpaca["enabled"].as<bool>(true);
                alpaca_config.api_key = expandEnvVars(alpaca["api_key"].as<std::string>(""));
                alpaca_config.api_secret = expandEnvVars(alpaca["api_secret"].as<std::string>(""));
                alpaca_config.base_url = alpaca["base_url"].as<std::string>("https://paper-api.alpaca.markets");
                alpaca_config.websocket_url = alpaca["websocket_url"].as<std::string>("wss://stream.data.alpaca.markets/v2/sip");
                
                data_sources_["alpaca"] = alpaca_config;
            }
        }
        
        // Parse ML configuration
        if (config["ml"]) {
            auto ml = config["ml"];
            
            // Parse model paths
            if (ml["model_paths"]) {
                auto paths = ml["model_paths"];
                ml_config_.model_paths.pattern_recognition = paths["pattern_recognition"].as<std::string>("");
                ml_config_.model_paths.exit_optimization = paths["exit_optimization"].as<std::string>("");
                ml_config_.model_paths.ranking = paths["ranking"].as<std::string>("");
                ml_config_.model_paths.sentiment = paths["sentiment"].as<std::string>("");
            }
            
            // Parse inference settings
            if (ml["inference"]) {
                auto inference = ml["inference"];
                ml_config_.inference.batch_size = inference["batch_size"].as<int>(64);
                ml_config_.inference.use_fp16 = inference["use_fp16"].as<bool>(true);
                ml_config_.inference.max_batch_latency_ms = inference["max_batch_latency_ms"].as<int>(5);
                ml_config_.inference.inference_threads = inference["inference_threads"].as<int>(2);
            }
            
            // Parse feature settings
            if (ml["features"]) {
                auto features = ml["features"];
                ml_config_.features.lookback_periods = features["lookback_periods"].as<int>(100);
                ml_config_.features.use_cuda_extraction = features["use_cuda_extraction"].as<bool>(true);
                
                // Parse indicators
                if (features["indicators"]) {
                    for (const auto& indicator : features["indicators"]) {
                        ml_config_.features.indicators.push_back(indicator.as<std::string>());
                    }
                }
            }
        }
        
        // Parse performance configuration
        if (config["performance"]) {
            auto perf = config["performance"];
            performance_config_.max_e2e_latency_us = perf["max_e2e_latency_us"].as<int>(1000);
            performance_config_.max_inference_latency_us = perf["max_inference_latency_us"].as<int>(500);
            performance_config_.max_execution_latency_us = perf["max_execution_latency_us"].as<int>(200);
            
            performance_config_.latency_monitoring = perf["latency_monitoring"].as<bool>(true);
            performance_config_.latency_log_interval_s = perf["latency_log_interval_s"].as<int>(60);
            performance_config_.performance_profiling = perf["performance_profiling"].as<bool>(true);
            
            performance_config_.use_lock_free_queues = perf["use_lock_free_queues"].as<bool>(true);
            performance_config_.use_zero_copy = perf["use_zero_copy"].as<bool>(true);
            performance_config_.websocket_parser_batch_size = perf["websocket_parser_batch_size"].as<int>(1000);
        }
        
        // Parse logging configuration
        if (config["logging"]) {
            auto log = config["logging"];
            logging_config_.level = log["level"].as<std::string>("INFO");
            logging_config_.file = log["file"].as<std::string>("logs/trading_system.log");
            logging_config_.max_file_size_mb = log["max_file_size_mb"].as<int>(100);
            logging_config_.max_files = log["max_files"].as<int>(10);
            logging_config_.log_format = log["log_format"].as<std::string>("[%(asctime)s] [%(levelname)s] [%(thread)d] %(message)s");
            
            logging_config_.log_latency_stats = log["log_latency_stats"].as<bool>(true);
            logging_config_.log_memory_usage = log["log_memory_usage"].as<bool>(true);
            
            logging_config_.use_zero_alloc_logging = log["use_zero_alloc_logging"].as<bool>(true);
            logging_config_.preallocated_log_buffer_size = log["preallocated_log_buffer_size"].as<int>(8192);
            logging_config_.flush_interval_ms = log["flush_interval_ms"].as<int>(1000);
            
            logging_config_.latency_log_interval_s = log["latency_log_interval_s"].as<int>(60);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading system config: " + std::string(e.what()));
    }
}

void Config::loadTradingConfig(const std::string& path) {
    try {
        // Load YAML file
        YAML::Node config = YAML::LoadFile(path);
        
        // Parse account settings
        if (config["account"]) {
            auto account = config["account"];
            trading_config_.account.initial_capital = account["initial_capital"].as<double>(25000.0);
            trading_config_.account.max_positions = account["max_positions"].as<int>(10);
            trading_config_.account.base_currency = account["base_currency"].as<std::string>("USD");
            trading_config_.account.account_type = account["account_type"].as<std::string>("margin");
            trading_config_.account.leverage = account["leverage"].as<double>(1.0);
        }
        
        // Parse risk settings
        if (config["risk"]) {
            auto risk = config["risk"];
            trading_config_.risk.max_position_size_pct = risk["max_position_size_pct"].as<double>(5.0);
            trading_config_.risk.position_sizing_method = risk["position_sizing_method"].as<std::string>("kelly");
            trading_config_.risk.kelly_fraction = risk["kelly_fraction"].as<double>(0.5);
            trading_config_.risk.max_daily_drawdown_pct = risk["max_daily_drawdown_pct"].as<double>(3.0);
            
            trading_config_.risk.use_stop_loss = risk["use_stop_loss"].as<bool>(true);
            trading_config_.risk.default_stop_loss_pct = risk["default_stop_loss_pct"].as<double>(2.0);
            
            trading_config_.risk.use_take_profit = risk["use_take_profit"].as<bool>(true);
            trading_config_.risk.default_take_profit_pct = risk["default_take_profit_pct"].as<double>(3.0);
            
            trading_config_.risk.use_trailing_stop = risk["use_trailing_stop"].as<bool>(true);
            trading_config_.risk.trailing_stop_pct = risk["trailing_stop_pct"].as<double>(1.5);
        }
        
        // Parse order settings
        if (config["orders"]) {
            auto orders = config["orders"];
            trading_config_.orders.default_order_type = orders["default_order_type"].as<std::string>("market");
            trading_config_.orders.use_bracket_orders = orders["use_bracket_orders"].as<bool>(true);
            trading_config_.orders.time_in_force = orders["time_in_force"].as<std::string>("day");
            trading_config_.orders.broker = "alpaca";  // Default broker
        }
        
        // Parse strategy settings
        if (config["strategies"]) {
            auto strategies = config["strategies"];
            
            // Parse pattern recognition strategy
            if (strategies["pattern_recognition"]) {
                auto pattern = strategies["pattern_recognition"];
                trading_config_.strategies.pattern_recognition_enabled = pattern["enabled"].as<bool>(true);
                trading_config_.strategies.confidence_threshold = pattern["confidence_threshold"].as<double>(0.7);
                
                // Parse patterns
                if (pattern["patterns"]) {
                    for (const auto& p : pattern["patterns"]) {
                        trading_config_.strategies.patterns.push_back(p.as<std::string>());
                    }
                }
            }
        }
        
        // Parse exit settings
        if (config["exit"]) {
            auto exit = config["exit"];
            trading_config_.exit.max_holding_time_minutes = exit["max_holding_time_minutes"].as<int>(240);
            trading_config_.exit.check_exit_interval_seconds = exit["check_exit_interval_seconds"].as<int>(60);
            trading_config_.exit.use_ml_exit = exit["use_ml_exit"].as<bool>(true);
            trading_config_.exit.exit_confidence_threshold = exit["exit_confidence_threshold"].as<double>(0.6);
            trading_config_.exit.close_positions_eod = exit["close_positions_eod"].as<bool>(true);
            trading_config_.exit.eod_exit_time = exit["eod_exit_time"].as<std::string>("15:45");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading trading config: " + std::string(e.what()));
    }
}

std::string Config::expandEnvVars(const std::string& value) {
    std::string result = value;
    std::regex env_var_pattern("\\${([^}]+)}");
    
    std::smatch match;
    while (std::regex_search(result, match, env_var_pattern)) {
        std::string env_var_name = match[1].str();
        const char* env_var_value = std::getenv(env_var_name.c_str());
        
        std::string replacement = env_var_value ? env_var_value : "";
        result.replace(match[0].first - result.begin(), 
                      match[0].length(), 
                      replacement);
    }
    
    return result;
}

void pinThreadToCore(int core_id) {
    if (core_id < 0) {
        return;
    }
    
#ifdef _WIN32
    // Windows implementation
    DWORD_PTR mask = 1ULL << core_id;
    SetThreadAffinityMask(GetCurrentThread(), mask);
#else
    // Linux/Unix implementation
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
#endif
}

} // namespace common
} // namespace trading_system