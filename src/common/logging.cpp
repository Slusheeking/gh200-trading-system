/**
 * Zero-allocation logging implementation
 */

#include <chrono>

#include <thread>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>
#include "trading_system/common/logging.h"

namespace trading_system {
namespace common {

// Global logger instance
ZeroAllocLogger g_logger("trading_system");

ZeroAllocLogger::ZeroAllocLogger(const std::string& name, LogLevel level)
    : name_(name),
      level_(level),
      write_index_(0),
      running_(true) {
    
    // Open log file
    file_.open(name + ".log", std::ios::out | std::ios::app);
    
    // Start flush thread
    flush_thread_ = std::thread(&ZeroAllocLogger::flushThreadFunc, this);
}

ZeroAllocLogger::~ZeroAllocLogger() {
    // Stop flush thread
    running_ = false;
    
    // Wait for thread to finish
    if (flush_thread_.joinable()) {
        flush_thread_.join();
    }
    
    // Flush remaining logs
    flush();
    
    // Close file
    if (file_.is_open()) {
        file_.close();
    }
}

void ZeroAllocLogger::log(LogLevel level, const std::string& message) {
    // Skip if level is below current level
    if (level < level_) {
        return;
    }
    
    // Get current index
    size_t index = write_index_.fetch_add(1) % BUFFER_SIZE;
    
    // Fill log entry
    LogEntry& entry = buffer_[index];
    entry.timestamp = getCurrentNanoTime();
    entry.level = level;
    entry.thread_id = static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    
    // Copy message (truncate if too long)
    strncpy(entry.message, message.c_str(), sizeof(entry.message) - 1);
    entry.message[sizeof(entry.message) - 1] = '\0';
}

void ZeroAllocLogger::setLevel(const std::string& level_str) {
    level_ = stringToLogLevel(level_str);
}

void ZeroAllocLogger::setLevel(LogLevel level) {
    level_ = level;
}

void ZeroAllocLogger::flush() {
    if (!file_.is_open()) {
        return;
    }
    
    // Get current index
    size_t current_index = write_index_.load();
    
    // Write all entries
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
        const LogEntry& entry = buffer_[i];
        
        // Skip empty entries
        if (entry.timestamp == 0) {
            continue;
        }
        
        // Format timestamp
        auto ns = std::chrono::nanoseconds(entry.timestamp);
        auto time_point = std::chrono::system_clock::time_point(ns);
        auto time_t = std::chrono::system_clock::to_time_t(time_point);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(ns).count() % 1000;
        
        // Write to file
        file_ << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
              << "." << std::setfill('0') << std::setw(3) << ms
              << " [" << logLevelToString(entry.level) << "] "
              << "[" << entry.thread_id << "] "
              << entry.message << std::endl;
    }
    
    // Flush file
    file_.flush();
}

LogLevel ZeroAllocLogger::stringToLogLevel(const std::string& level_str) {
    if (level_str == "DEBUG") {
        return LogLevel::DEBUG;
    } else if (level_str == "INFO") {
        return LogLevel::INFO;
    } else if (level_str == "WARNING") {
        return LogLevel::WARNING;
    } else if (level_str == "ERROR") {
        return LogLevel::ERROR;
    } else if (level_str == "CRITICAL") {
        return LogLevel::CRITICAL;
    } else {
        return LogLevel::INFO;  // Default to INFO
    }
}

const char* ZeroAllocLogger::logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

void ZeroAllocLogger::flushThreadFunc() {
    while (running_) {
        // Flush logs
        flush();
        
        // Sleep for a while
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

uint64_t ZeroAllocLogger::getCurrentNanoTime() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
}

} // namespace common
} // namespace trading_system