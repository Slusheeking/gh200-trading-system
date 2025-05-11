cmake_minimum_required(VERSION 3.22)
project(TradingSystem LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80 86 90)  # For Ampere, Ada Lovelace, and Hopper

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

# Find packages
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread filesystem program_options)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(CURL REQUIRED)
# Find nlohmann_json
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
# Find spdlog
find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h REQUIRED)
include_directories(${SPDLOG_INCLUDE_DIR})
# Find TBB
find_path(TBB_INCLUDE_DIR tbb/tbb.h REQUIRED)
find_library(TBB_LIBRARY tbb REQUIRED)
include_directories(${TBB_INCLUDE_DIR})
# Find yaml-cpp
find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h REQUIRED)
find_library(YAML_CPP_LIBRARY yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Add library
add_library(trading_core SHARED
    src/cuda/parser.cu
    src/cuda/features.cu
    src/data/websocket_client.cpp
    src/data/market_data.cpp
    src/execution/order.cpp
    src/execution/execution_engine.cpp
    src/execution/broker_client.cpp
    src/execution/paper_trading_broker_client.cpp
    src/common/config.cpp
    src/common/logging.cpp
    src/risk/risk_manager.cpp
    src/ml/inference.cpp
    src/ml/model.cpp
)

# Link libraries
target_link_libraries(trading_core
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    ZLIB::ZLIB
    spdlog
    ${TBB_LIBRARY}
    ${YAML_CPP_LIBRARY}
    Threads::Threads
    ${CURL_LIBRARIES}
)

# Add ML inference library
add_library(ml_inference SHARED
    src/cuda/inference.cu
    src/cuda/model_loader.cu
)

target_link_libraries(ml_inference
    ${CUDA_LIBRARIES}
)

# Add monitoring library
add_library(monitoring SHARED
    src/monitoring/metrics_publisher.cpp
)

target_link_libraries(monitoring
    ${Boost_LIBRARIES}
    spdlog
    Threads::Threads
)

# Add executable
add_executable(trading_system src/main.cpp)
target_link_libraries(trading_system
    trading_core
    ml_inference
    monitoring
)

# Install targets
install(TARGETS trading_system DESTINATION bin)
install(TARGETS trading_core ml_inference DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)