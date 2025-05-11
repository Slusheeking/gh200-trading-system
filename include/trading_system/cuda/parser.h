/**
 * CUDA-accelerated WebSocket parser
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <cstddef>
#include "trading_system/common/config.h"
#include "trading_system/data/market_data.h"

namespace trading_system {
namespace cuda {

class Parser {
public:
    Parser(const common::Config& config);
    ~Parser();
    
    // Parse market data using CUDA
    data::ParsedMarketData parse(const data::MarketData& market_data);
    
    // Reset parser state
    void reset();
    
private:
    // CUDA stream for asynchronous operations
    cudaStream_t stream_;
    
    // Device memory for input data
    void* d_input_buffer_;
    size_t d_input_buffer_size_;
    
    // Device memory for output data
    void* d_output_buffer_;
    size_t d_output_buffer_size_;
    
    // Host memory for output data (pinned)
    void* h_output_buffer_;
    
    // Configuration
    int batch_size_;
    bool use_zero_copy_;
    
    // Initialize CUDA resources
    void initCudaResources();
    
    // Free CUDA resources
    void freeCudaResources();
    
    // Launch CUDA kernel for parsing
    void launchParserKernel(const void* input_data, size_t input_size);
};

} // namespace cuda
} // namespace trading_system