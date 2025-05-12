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
# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
# Find pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/pybind11-src/third_party/simdjson
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Add library
add_library(trading_core SHARED
    src/data/websocket_client.cpp
    src/data/market_data.cpp
    src/data/rest_client.cpp
    src/data/polygon_client.cpp
    src/execution/order.cpp
    src/execution/execution_engine.cpp
    src/execution/broker_client.cpp
    src/common/config.cpp
    src/common/logging.cpp
    src/risk/risk_manager.cpp
    src/ml/inference.cpp
    src/ml/model.cpp
    src/ml/gbdt_model.cpp
    src/ml/axial_attention_model.cpp # Added Axial Attention model
    src/ml/lstm_gru_model.cpp      # Added LSTM/GRU model
    src/ml/exit_optimization.cpp
    src/ml/model_trainer.cpp       # Added Model Trainer
    src/ml/market_snapshot_processor.cpp # Added Market Snapshot Processor
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
    ta_lib
    lightgbm
    nvinfer
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
    monitoring
)

# Examples
add_executable(polygon_snapshot_example examples/polygon_snapshot_example.cpp)
target_link_libraries(polygon_snapshot_example
    trading_core
    monitoring
)

# Model trainer executable
add_executable(model_trainer src/ml/model_trainer_main.cpp)
target_link_libraries(model_trainer
    trading_core
    monitoring
)

# Install targets
install(TARGETS trading_system DESTINATION bin)
install(TARGETS trading_core DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
