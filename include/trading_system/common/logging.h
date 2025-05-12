/**
 * Zero-allocation logging for trading system
 */

#pragma once

#include <string>
#include <fstream>
#include <thread>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>

namespace trading_system {
namespace common {

// Log levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

// Log entry structure (pre-allocated)
struct LogEntry {
    uint64_t timestamp;       // Nanoseconds since epoch
    LogLevel level;
    uint32_t thread_id;
    char message[1024];       // Fixed-size message buffer
};

// Zero-allocation logger
class ZeroAllocLogger {
public:
    ZeroAllocLogger(const std::string& name, LogLevel level = LogLevel::INFO);
    ~ZeroAllocLogger();
    
    // Log a message with no allocations
    template <typename... Args>
    void log(LogLevel level, const char* format, Args... args);
    
    // Set log level
    void setLevel(const std::string& level_str);
    void setLevel(LogLevel level);
    
    // Flush logs to disk
    void flush();
    
private:
    // Pre-allocated ring buffer
    static constexpr size_t BUFFER_SIZE = 8192;
    std::array<LogEntry, BUFFER_SIZE> buffer_;
    std::atomic<size_t> write_index_{0};
    
    // Logger name
    std::string name_;
    
    // Current log level
    std::atomic<LogLevel> level_;
    
    // Output file
    std::ofstream file_;
    
    // Background thread for flushing
    std::thread flush_thread_;
    std::atomic<bool> running_{true};
    
    // Convert log level string to enum
    LogLevel stringToLogLevel(const std::string& level_str);
    
    // Convert log level to string
    const char* logLevelToString(LogLevel level);
    
    // Background flush function
    void flushThreadFunc();
    
    // Get current timestamp in nanoseconds
    uint64_t getCurrentNanoTime();
};

// Global logger instance
extern ZeroAllocLogger g_logger;

// Convenience macros
#define LOG_DEBUG(format, ...) g_logger.log(LogLevel::DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) g_logger.log(LogLevel::INFO, format, ##__VA_ARGS__)
#define LOG_WARNING(format, ...) g_logger.log(LogLevel::WARNING, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) g_logger.log(LogLevel::ERROR, format, ##__VA_ARGS__)
#define LOG_CRITICAL(format, ...) g_logger.log(LogLevel::CRITICAL, format, ##__VA_ARGS__)

} // namespace common
} // namespace trading_system