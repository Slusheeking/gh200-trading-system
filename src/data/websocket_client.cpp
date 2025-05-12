/**
 * WebSocket client implementation
 */

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <nlohmann/json.hpp>

#include "trading_system/common/config.h"
#include "trading_system/common/logging.h"
#include "trading_system/data/market_data.h"
#include "trading_system/data/websocket_client.h"
#include <thread>
#include <mutex>
#include <condition_variable>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

namespace trading_system {
namespace data {

class WebSocketClient::Impl {
public:
    Impl(const common::Config& config)
        : config_(config),
          io_context_(),
          work_guard_(net::make_work_guard(io_context_)),
          running_(false),
          connected_(false) {
        
        // Try to read API keys from credentials file
        std::unordered_map<std::string, std::string> credentials = readCredentialsFile();
        bool has_credentials = !credentials.empty();
        
        if (has_credentials) {
            std::cout << "Found API keys in credentials file" << std::endl;
        }
        
        // Get data source config
        const auto& polygon_config = config.getDataSourceConfig("polygon");
        
        // Set up data sources
        if (polygon_config.enabled) {
            // Use API key from credentials file if available
            std::string api_key = polygon_config.api_key;
            if (has_credentials && credentials.find("POLYGON_API_KEY") != credentials.end()) {
                api_key = credentials["POLYGON_API_KEY"];
                std::cout << "Using Polygon API key from credentials file" << std::endl;
            }
            
            data_sources_.push_back({
                "polygon",
                polygon_config.websocket_url,
                api_key,
                polygon_config.subscription_type
            });
        }
        
        // Initialize data buffer
        buffer_size_ = config.getPerformanceConfig().websocket_parser_batch_size;
        data_buffer_.reserve(buffer_size_);
    }
    
    ~Impl() {
        disconnect();
        
        // Stop IO context
        work_guard_.reset();
        io_context_.stop();
        
        // Join worker thread
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    void connect() {
        if (running_) {
            return;
        }
        
        running_ = true;
        
        // Check if any data sources are enabled
        if (data_sources_.empty()) {
            std::cout << "No data sources enabled. Running in simulation mode." << std::endl;
            connected_ = true;  // Mark as connected to avoid timeout error
            return;
        }
        
        // Start worker thread
        worker_thread_ = std::thread([this]() {
            try {
                // Connect to all data sources
                for (auto& source : data_sources_) {
                    connectToDataSource(source);
                }
                
                // Run IO context
                io_context_.run();
            } catch (const std::exception& e) {
                std::cerr << "WebSocket worker thread exception: " << e.what() << std::endl;
            }
        });
        
        // Set thread affinity if specified
        if (thread_affinity_ >= 0) {
            common::pinThreadToCore(thread_affinity_);
        }
        
        // Wait for connection
        std::unique_lock<std::mutex> lock(mutex_);
        bool is_connected = false;
        cv_.wait_for(lock, std::chrono::seconds(5), [this, &is_connected]() {
            is_connected = connected_.load();
            return is_connected;
        });
        
        if (!connected_ && !data_sources_.empty()) {
            std::cerr << "Failed to connect to any data source within timeout" << std::endl;
        }
    }
    
    void disconnect() {
        if (!running_) {
            return;
        }
        
        running_ = false;
        
        // Close all WebSocket connections
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& ws : websockets_) {
            try {
                ws->close(websocket::close_code::normal);
            } catch (const std::exception& e) {
                std::cerr << "Error closing WebSocket: " << e.what() << std::endl;
            }
        }
        
        websockets_.clear();
        connected_ = false;
    }
    
    void getLatestData(MarketData& market_data) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // Copy data to output
        for (const auto& trade : data_buffer_) {
            market_data.addTrade(trade);
        }
        
        // Clear buffer
        data_buffer_.clear();
    }
    
    void setThreadAffinity(int core_id) {
        thread_affinity_ = core_id;
        
        // Apply to running thread if exists
        if (worker_thread_.joinable()) {
            common::pinThreadToCore(core_id);
        }
    }
    
private:
    // Data source information
    struct DataSource {
        std::string name;
        std::string url;
        std::string api_key;
        std::string subscription;
    };
    
    // Read API keys from credentials file
    std::unordered_map<std::string, std::string> readCredentialsFile() {
        std::unordered_map<std::string, std::string> credentials;
        
        // Try to open the credentials file
        std::ifstream file("/etc/trading-system/credentials");
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open credentials file at /etc/trading-system/credentials" << std::endl;
            trading_system::common::LOG_WARNING("Could not open credentials file at /etc/trading-system/credentials");
            return credentials;
        }
        
        // Read each line
        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines
            if (line.empty()) {
                continue;
            }
            
            // Parse key=value format
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                credentials[key] = value;
            }
        }
        
        return credentials;
    }
    
    // Configuration
    const common::Config& config_;
    
    // IO context and work guard
    net::io_context io_context_;
    net::executor_work_guard<net::io_context::executor_type> work_guard_;
    
    // Worker thread
    std::thread worker_thread_;
    int thread_affinity_ = -1;
    
    // WebSocket connections
    std::vector<std::unique_ptr<websocket::stream<beast::tcp_stream>>> websockets_;
    
    // Data sources
    std::vector<DataSource> data_sources_;
    
    // State
    std::atomic<bool> running_;
    std::atomic<bool> connected_;
    
    // Synchronization
    std::mutex mutex_;
    std::condition_variable cv_;
    
    // Data buffer
    std::vector<Trade> data_buffer_;
    std::mutex data_mutex_;
    size_t buffer_size_;
    
    // Connect to a data source
    void connectToDataSource(const DataSource& source) {
        try {
            // Parse URL
            std::string host = source.url.substr(6);  // Remove "wss://"
            std::string port = "443";
            
            // Split host and path
            std::string path = "/";
            auto pos = host.find('/');
            if (pos != std::string::npos) {
                path = host.substr(pos);
                host = host.substr(0, pos);
            }
            
            // Split host and port
            pos = host.find(':');
            if (pos != std::string::npos) {
                port = host.substr(pos + 1);
                host = host.substr(0, pos);
            }
            
            // Resolve endpoint
            tcp::resolver resolver(io_context_);
            auto const results = resolver.resolve(host, port);
            
            // Create WebSocket
            auto ws = std::make_unique<websocket::stream<beast::tcp_stream>>(io_context_);
            
            // Set options
            ws->set_option(websocket::stream_base::timeout::suggested(
                beast::role_type::client));
            
            // Connect to endpoint
            beast::get_lowest_layer(ws->next_layer()).connect(results);
            
            // Perform WebSocket handshake
            ws->handshake(host, path);
            
            // Authenticate
            if (source.name == "polygon") {
                // Polygon authentication
                json auth = {
                    {"action", "auth"},
                    {"params", source.api_key}
                };
                ws->write(net::buffer(auth.dump()));
                
                // Subscribe to data
                json subscribe = {
                    {"action", "subscribe"},
                    {"params", source.subscription}
                };
                ws->write(net::buffer(subscribe.dump()));
            }
            
            // Start reading
            startReading(ws.get());
            
            // Add to connections
            {
                std::lock_guard<std::mutex> lock(mutex_);
                websockets_.push_back(std::move(ws));
                connected_ = true;
            }
            
            // Notify connection
            cv_.notify_all();
            
            std::cout << "Connected to " << source.name << " WebSocket" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error connecting to " << source.name << ": " << e.what() << std::endl;
        }
    }
    
    // Start reading from WebSocket
    void startReading(websocket::stream<beast::tcp_stream>* ws) {
        auto buffer = std::make_shared<beast::flat_buffer>();
        
        ws->async_read(
            *buffer,
            [this, ws, buffer](beast::error_code ec, std::size_t bytes_transferred) {
                (void)bytes_transferred; // Unused parameter
                if (ec) {
                    std::cerr << "WebSocket read error: " << ec.message() << std::endl;
                    return;
                }
                
                // Process message
                processMessage(
                    beast::buffers_to_string(buffer->data()));
                
                // Continue reading if still running
                if (running_) {
                    startReading(ws);
                }
            });
    }
    
    // Process WebSocket message
    void processMessage(const std::string& message) {
        try {
            // Parse JSON
            json j = json::parse(message);
            
            // Process based on message type
            if (j.contains("ev") && j["ev"] == "T") {
                // Polygon trade
                Trade trade;
                trade.symbol = j["sym"].get<std::string>();
                trade.price = j["p"];
                trade.size = j["s"];
                trade.timestamp = j["t"];
                trade.exchange = j["x"].get<std::string>();
                trade.conditions = j["c"].dump();
                
                // Add to buffer
                std::lock_guard<std::mutex> lock(data_mutex_);
                data_buffer_.push_back(trade);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing WebSocket message: " << e.what() << std::endl;
        }
    }
}
};

// WebSocketClient implementation
WebSocketClient::WebSocketClient(const common::Config& config)
    : impl_(std::make_unique<Impl>(config)) {
}

WebSocketClient::~WebSocketClient() = default;

void WebSocketClient::connect() {
    impl_->connect();
}

void WebSocketClient::disconnect() {
    impl_->disconnect();
}

void WebSocketClient::getLatestData(MarketData& market_data) {
    impl_->getLatestData(market_data);
}

void WebSocketClient::setThreadAffinity(int core_id) {
    impl_->setThreadAffinity(core_id);
}

} // namespace data
} // namespace trading_system
