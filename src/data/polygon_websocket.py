"""
Polygon WebSocket client implementation

This module provides a client for the Polygon.io WebSocket API to receive real-time market data.
It handles authentication, connection management, and parsing of WebSocket messages.
"""

import json
import time
import threading
import websocket
import logging
import os
import random
# Use orjson if available (faster), otherwise fallback to standard json
try:
    import orjson as json_fast  # Faster JSON parsing
except ImportError:
    try:
        import ujson as json_fast  # Faster JSON parsing
    except ImportError:
        import json as json_fast
        logging.warning("Using standard json module. Consider installing orjson or ujson for better performance.")
from typing import Dict, List, Any, Optional, Callable, Deque
from collections import deque
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

class Trade:
    """Class to store trade data with memory-efficient representation"""
    __slots__ = ['symbol', 'price', 'size', 'timestamp', 'exchange', 'conditions']
    
    def __init__(self):
        self.symbol: str = ""
        self.price: float = 0.0
        self.size: int = 0
        self.timestamp: int = 0
        self.exchange: str = ""
        self.conditions: str = ""

class MarketData:
    """Class to store market data with optimized data structures"""
    def __init__(self, max_trades: int = 10000):
        self.trades: Deque[Trade] = deque(maxlen=max_trades)  # Use deque with maxlen for automatic size management
        self.quotes: Deque[Any] = deque(maxlen=max_trades)
        self.trade_count: int = 0
        self.quote_count: int = 0
        self.last_timestamp: int = 0
    
    def add_trade(self, trade: Trade):
        """Add a trade to the market data"""
        self.trades.append(trade)
        self.trade_count += 1
        self.last_timestamp = max(self.last_timestamp, trade.timestamp)
    
    def add_trades_batch(self, trades: List[Trade]):
        """Add multiple trades at once for better performance"""
        self.trades.extend(trades)
        self.trade_count += len(trades)
        if trades:
            self.last_timestamp = max(self.last_timestamp, max(trade.timestamp for trade in trades))
    
    def clear(self):
        """Clear all data"""
        self.trades.clear()
        self.quotes.clear()

class PolygonWebSocket:
    """High-performance WebSocket client for Polygon.io"""
    
    class DataSource:
        """Class to store data source information"""
        __slots__ = ['name', 'url', 'api_key', 'subscription', 'reconnect_attempts', 'last_connected']
        
        def __init__(self):
            self.name: str = ""
            self.url: str = ""
            self.api_key: str = ""
            self.subscription: str = ""
            self.reconnect_attempts: int = 0
            self.last_connected: float = 0
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the WebSocket client with high-performance optimizations
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.connected = False
        
        # Load environment variables
        load_dotenv()
        
        # Read API keys from config
        self.data_sources: List[PolygonWebSocket.DataSource] = []
        
        # Get data source config
        polygon_config = config.get("data_sources", {}).get("polygon", {})
        
        # Set up data sources
        if polygon_config.get("enabled", False):
            # Get API key from environment variables first, then fall back to config
            api_key = os.getenv("POLYGON_API_KEY", "")
            if not api_key:
                api_key = polygon_config.get("api_key", "")
            
            # Get websocket URL from environment variables first, then fall back to config
            websocket_url = os.getenv("POLYGON_WEBSOCKET_URL", "")
            if not websocket_url:
                websocket_url = polygon_config.get("websocket_url", "")
            
            data_source = PolygonWebSocket.DataSource()
            data_source.name = "polygon"
            data_source.url = websocket_url
            data_source.api_key = api_key
            data_source.subscription = polygon_config.get("subscription_type", "")
            
            self.data_sources.append(data_source)
        
        # Performance settings from config
        perf_config = config.get("performance", {})
        self.max_workers = perf_config.get("processor_threads", min(32, os.cpu_count() * 2))
        self.batch_size = perf_config.get("websocket_parser_batch_size", 1000)
        self.use_numpy = True
        
        # Initialize optimized data buffer with ring buffer (deque with maxlen)
        self.buffer_size = self.batch_size
        self.data_buffer: Deque[Trade] = deque(maxlen=self.buffer_size)
        self.message_queue: Deque[Dict[str, Any]] = deque(maxlen=self.buffer_size * 2)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="PolygonWS-Worker"
        )
        
        # Thread management
        self.worker_thread = None
        self.processor_thread = None
        self.thread_affinity = -1
        
        # WebSocket connections
        self.websockets: Dict[str, websocket.WebSocketApp] = {}
        self.ws_threads: Dict[str, threading.Thread] = {}
        
        # Connection management from system.yaml
        polygon_config = config.get("data_sources", {}).get("polygon", {})
        websocket_config = polygon_config.get("websocket", {})
        
        self.max_reconnect_attempts = websocket_config.get("max_reconnect_attempts", 10)
        self.reconnect_base_delay = websocket_config.get("reconnect_base_delay_ms", 1000) / 1000.0
        self.reconnect_max_delay = websocket_config.get("reconnect_max_delay_ms", 30000) / 1000.0
        self.heartbeat_interval = websocket_config.get("heartbeat_interval_ms", 30000) / 1000.0
        self.enable_trace = websocket_config.get("enable_trace", False)
        
        # Performance metrics
        self.message_count = 0
        self.last_message_time = 0
        self.message_rate = 0
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        
        logging.info(f"Configured high-performance WebSocket client with max_workers={self.max_workers}, " +
                    f"batch_size={self.batch_size}, " +
                    f"max_reconnect_attempts={self.max_reconnect_attempts}, " +
                    f"reconnect_base_delay={self.reconnect_base_delay}s, " +
                    f"reconnect_max_delay={self.reconnect_max_delay}s, " +
                    f"heartbeat_interval={self.heartbeat_interval}s")
        
        # Synchronization with more efficient locking
        self.mutex = threading.Lock()
        self.cv = threading.Condition(self.mutex)
        self.data_mutex = threading.Lock()
        self.message_event = threading.Event()  # Signal when new messages are available
        
        # Callbacks
        self.connection_status_callback: Optional[Callable[[str, bool], None]] = None
    
    def connect(self):
        """Connect to WebSocket data sources with optimized processing"""
        if self.running:
            logging.warning("WebSocket client already running")
            return
        
        self.running = True
        
        # Check if any data sources are enabled
        if not self.data_sources:
            logging.info("No data sources enabled. Running in simulation mode.")
            self.connected = True  # Mark as connected to avoid timeout error
            return
        
        # Start message processor thread
        self.processor_thread = threading.Thread(
            target=self._message_processor_thread_func,
            name="PolygonWebSocket-Processor"
        )
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self._worker_thread_func,
            name="PolygonWebSocket-Worker"
        )
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # Wait for connection
        with self.cv:
            is_connected = self.cv.wait_for(
                lambda: self.connected,
                timeout=5.0
            )
        
        if not is_connected and self.data_sources:
            logging.error("Failed to connect to any data source within timeout")
        elif self.data_sources:
            logging.info("Successfully connected to WebSocket data sources")
        else:
            logging.info("No data sources to connect to")
    
    def disconnect(self):
        """Disconnect from WebSocket data sources"""
        if not self.running:
            logging.warning("WebSocket client not running")
            return
        
        logging.info("Disconnecting from WebSocket data sources")
        self.running = False
        
        # Signal message processor to exit
        self.message_event.set()
        
        # Close all WebSocket connections
        with self.mutex:
            for source_name, ws in self.websockets.items():
                try:
                    logging.debug(f"Closing WebSocket connection to {source_name}")
                    ws.close()
                except Exception as e:
                    logging.error(f"Error closing WebSocket to {source_name}: {str(e)}")
            
            self.websockets = {}
            self.ws_threads = {}
            self.connected = False
        
        # Wait for worker thread to terminate
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logging.warning("Worker thread did not terminate within timeout")
            else:
                logging.debug("Worker thread terminated")
        
        # Wait for processor thread to terminate
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5.0)
            if self.processor_thread.is_alive():
                logging.warning("Processor thread did not terminate within timeout")
            else:
                logging.debug("Processor thread terminated")
    
    def set_connection_status_callback(self, callback: Callable[[str, bool], None]):
        """
        Set callback for connection status updates
        
        Args:
            callback: Callback function that takes source name and connection status
        """
        self.connection_status_callback = callback
        logging.debug("Connection status callback set")
    
    def get_latest_data(self, market_data: MarketData):
        """
        Get the latest market data with optimized batch transfer
        
        Args:
            market_data: MarketData object to populate
        """
        with self.data_mutex:
            # Use batch add for better performance
            if self.data_buffer:
                # Convert to list for batch add
                trades_list = list(self.data_buffer)
                market_data.add_trades_batch(trades_list)
                
                # Clear buffer
                trade_count = len(self.data_buffer)
                self.data_buffer.clear()  # More efficient than creating a new list
                
                if trade_count > 0:
                    logging.debug(f"Retrieved {trade_count} trades from buffer")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the WebSocket client
        
        Returns:
            Dictionary with performance metrics
        """
        with self.data_mutex:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            
            return {
                "message_rate": self.message_rate,
                "avg_processing_time_ms": avg_processing_time,
                "buffer_utilization": len(self.data_buffer) / self.buffer_size if self.buffer_size > 0 else 0,
                "queue_utilization": len(self.message_queue) / (self.buffer_size * 2) if self.buffer_size > 0 else 0,
                "connected_sources": len(self.websockets)
            }
    
    def set_thread_affinity(self, core_id: int):
        """
        Set thread affinity for the worker thread
        
        Args:
            core_id: CPU core ID to pin the thread to
        """
        self.thread_affinity = core_id
        
        # Apply to running thread if exists
        if self.worker_thread and self.worker_thread.is_alive():
            logging.debug(f"Thread affinity set to core {core_id} (not implemented)")
    
    def _worker_thread_func(self):
        """Worker thread function"""
        try:
            # Connect to all data sources
            for source in self.data_sources:
                self._connect_to_data_source(source)
            
            # Keep thread alive while running and monitor connections
            while self.running:
                # Check for disconnected sources and attempt reconnection
                self._check_connections()
                
                # Sleep for a bit
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"WebSocket worker thread exception: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
    
    def _check_connections(self):
        """Check connection status and reconnect if needed"""
        with self.mutex:
            current_time = time.time()
            
            # Check each data source
            for source in self.data_sources:
                # Skip if already connected
                if source.name in self.websockets:
                    continue
                
                # Check if we should attempt reconnection
                if source.reconnect_attempts < self.max_reconnect_attempts:
                    # Calculate backoff delay
                    backoff = min(
                        self.reconnect_base_delay * (2 ** source.reconnect_attempts) + random.uniform(0, 1),
                        self.reconnect_max_delay
                    )
                    
                    # Check if enough time has passed since last attempt
                    if current_time - source.last_connected > backoff:
                        logging.info(f"Attempting to reconnect to {source.name} (attempt {source.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                        source.reconnect_attempts += 1
                        source.last_connected = current_time
                        
                        # Connect in a separate thread to avoid blocking
                        reconnect_thread = threading.Thread(
                            target=self._connect_to_data_source,
                            args=(source,),
                            name=f"PolygonWebSocket-Reconnect-{source.name}"
                        )
                        reconnect_thread.daemon = True
                        reconnect_thread.start()
                elif source.reconnect_attempts >= self.max_reconnect_attempts:
                    logging.error(f"Maximum reconnection attempts reached for {source.name}")
    
    def _connect_to_data_source(self, source: DataSource):
        """
        Connect to a data source
        
        Args:
            source: DataSource object with connection information
        """
        try:
            # Parse URL
            url = source.url
            
            if not url:
                logging.error(f"No URL specified for {source.name}")
                return
                
            if not source.api_key:
                logging.error(f"No API key specified for {source.name}")
                return
            
            logging.info(f"Connecting to {source.name} WebSocket at {url}")
            
            # Create WebSocket with ping/pong enabled for heartbeat
            ws = websocket.WebSocketApp(
                url,
                on_open=lambda ws: self._on_open(ws, source),
                on_message=lambda ws, msg: self._on_message(ws, msg, source),
                on_error=lambda ws, error: self._on_error(ws, error, source),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg, source),
                on_ping=lambda ws, message: self._on_ping(ws, message, source),
                on_pong=lambda ws, message: self._on_pong(ws, message, source)
            )
            
            # Enable ping/pong for heartbeat
            websocket.enableTrace(self.enable_trace)
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(
                target=lambda: ws.run_forever(
                    ping_interval=self.heartbeat_interval,
                    ping_timeout=self.heartbeat_interval / 2
                ),
                name=f"PolygonWebSocket-{source.name}"
            )
            ws_thread.daemon = True
            ws_thread.start()
            
            # Add to connections
            with self.mutex:
                self.websockets[source.name] = ws
                self.ws_threads[source.name] = ws_thread
            
            logging.info(f"Connected to {source.name} WebSocket")
        except Exception as e:
            logging.error(f"Error connecting to {source.name}: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            
            # Mark for reconnection
            source.reconnect_attempts += 1
            source.last_connected = time.time()
    
    def _on_open(self, ws, source: DataSource):
        """
        WebSocket on_open callback
        
        Args:
            ws: WebSocket connection
            source: DataSource object with connection information
        """
        logging.info(f"WebSocket connection opened for {source.name}")
        
        try:
            # Authenticate
            if source.name == "polygon":
                # Polygon authentication
                auth = {
                    "action": "auth",
                    "params": source.api_key
                }
                ws.send(json.dumps(auth))
                logging.debug(f"Sent authentication request to {source.name}")
                
                # Subscribe to data
                if source.subscription:
                    subscribe = {
                        "action": "subscribe",
                        "params": source.subscription
                    }
                    ws.send(json.dumps(subscribe))
                    logging.debug(f"Subscribed to {source.subscription} on {source.name}")
                else:
                    logging.warning(f"No subscription specified for {source.name}")
            
            # Reset reconnection attempts on successful connection
            source.reconnect_attempts = 0
            source.last_connected = time.time()
            
            # Mark as connected
            with self.mutex:
                self.connected = True
                self.cv.notify_all()
            
            # Notify via callback if set
            if self.connection_status_callback:
                self.connection_status_callback(source.name, True)
                
        except Exception as e:
            logging.error(f"Error in WebSocket on_open for {source.name}: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
    
    def _on_message(self, ws, message: str, source: DataSource):
        """
        WebSocket on_message callback - optimized for high throughput
        
        Args:
            ws: WebSocket connection
            message: Message received from WebSocket
            source: DataSource object with connection information
        """
        try:
            # Update message count and rate metrics
            self.message_count += 1
            current_time = time.time()
            if current_time - self.last_message_time >= 1.0:  # Calculate rate every second
                self.message_rate = self.message_count / (current_time - self.last_message_time)
                self.message_count = 0
                self.last_message_time = current_time
            
            # Parse JSON using ujson for better performance
            start_time = time.time()
            data = json_fast.loads(message)
            
            # Add to message queue for batch processing
            if isinstance(data, list):
                # For arrays, add each item to the queue
                with self.data_mutex:
                    for item in data:
                        self.message_queue.append(item)
            else:
                # Add single message to queue
                with self.data_mutex:
                    self.message_queue.append(data)
            
            # Signal message processor that new messages are available
            self.message_event.set()
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_times.append(processing_time)
            
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON from {source.name}: {str(e)}")
            logging.debug(f"Raw message: {message[:100]}...")
        except Exception as e:
            logging.error(f"Error processing WebSocket message from {source.name}: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
    
    def _on_error(self, ws, error, source: DataSource):
        """
        WebSocket on_error callback
        
        Args:
            ws: WebSocket connection
            error: Error object
            source: DataSource object with connection information
        """
        logging.error(f"WebSocket error from {source.name}: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg, source: DataSource):
        """
        WebSocket on_close callback
        
        Args:
            ws: WebSocket connection
            close_status_code: Close status code
            close_msg: Close message
            source: DataSource object with connection information
        """
        logging.info(f"WebSocket closed for {source.name}: {close_status_code} {close_msg}")
        
        # Remove from connections
        with self.mutex:
            if source.name in self.websockets:
                del self.websockets[source.name]
            
            if source.name in self.ws_threads:
                del self.ws_threads[source.name]
            
            # Mark as disconnected if no connections left
            if not self.websockets:
                self.connected = False
        
        # Notify via callback if set
        if self.connection_status_callback:
            self.connection_status_callback(source.name, False)
    
    def _on_ping(self, ws, message, source: DataSource):
        """
        WebSocket on_ping callback
        
        Args:
            ws: WebSocket connection
            message: Ping message
            source: DataSource object with connection information
        """
        logging.debug(f"Received ping from {source.name}")
    
    def _on_pong(self, ws, message, source: DataSource):
        """
        WebSocket on_pong callback
        
        Args:
            ws: WebSocket connection
            message: Pong message
            source: DataSource object with connection information
        """
        logging.debug(f"Received pong from {source.name}")
    
    def _message_processor_thread_func(self):
        """
        Message processor thread function
        
        This thread processes messages in batches for better performance
        """
        logging.info("Message processor thread started")
        
        try:
            while self.running:
                # Wait for new messages or timeout
                self.message_event.wait(timeout=0.01)  # 10ms timeout
                self.message_event.clear()
                
                # Process messages in batch
                messages_to_process = []
                with self.data_mutex:
                    # Get up to batch_size messages
                    batch_size = min(len(self.message_queue), self.batch_size)
                    if batch_size > 0:
                        messages_to_process = [self.message_queue.popleft() for _ in range(batch_size)]
                
                if messages_to_process:
                    # Process messages in parallel using thread pool
                    if len(messages_to_process) > 10:  # Only use parallel for larger batches
                        # Create a list to store trades
                        batch_trades = []
                        
                        # Submit tasks to thread pool
                        futures = []
                        for msg in messages_to_process:
                            if msg.get("ev") == "T":  # Only process trades in parallel
                                futures.append(self.thread_pool.submit(self._process_trade_message, msg))
                            else:
                                self._process_message(msg)  # Process other messages directly
                        
                        # Collect results
                        for future in futures:
                            trade = future.result()
                            if trade:
                                batch_trades.append(trade)
                        
                        # Add trades to buffer in one batch
                        if batch_trades:
                            with self.data_mutex:
                                self.data_buffer.extend(batch_trades)
                                
                            # Log periodically
                            logging.debug(f"Processed {len(batch_trades)} trades in parallel batch")
                    else:
                        # Process sequentially for small batches
                        for msg in messages_to_process:
                            self._process_message(msg)
                
                # Sleep a tiny bit to avoid CPU spinning
                if not messages_to_process:
                    time.sleep(0.001)  # 1ms sleep when no messages
                    
        except Exception as e:
            logging.error(f"Message processor thread exception: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
    
    def _process_trade_message(self, message: Dict[str, Any]) -> Optional[Trade]:
        """
        Process a trade message and return a Trade object
        
        Args:
            message: Trade message to process
            
        Returns:
            Trade object or None if not a trade
        """
        if message.get("ev") == "T":
            # Polygon trade with actual data
            trade = Trade()
            trade.symbol = message.get("sym", "")
            trade.price = message.get("p", 0.0)  # Actual price
            trade.size = message.get("s", 0)  # Actual size
            trade.timestamp = message.get("t", 0)  # Actual timestamp
            trade.exchange = message.get("x", "")
            trade.conditions = json_fast.dumps(message.get("c", []))
            return trade
        return None
    
    def _process_message(self, message: Dict[str, Any]):
        """
        Process a WebSocket message
        
        Args:
            message: Message to process
        """
        try:
            # Process based on message type
            if message.get("ev") == "T":
                # Polygon trade with actual data
                trade = Trade()
                trade.symbol = message.get("sym", "")
                trade.price = message.get("p", 0.0)  # Actual price
                trade.size = message.get("s", 0)  # Actual size
                trade.timestamp = message.get("t", 0)  # Actual timestamp
                trade.exchange = message.get("x", "")
                trade.conditions = json_fast.dumps(message.get("c", []))
                
                # Add to buffer
                with self.data_mutex:
                    self.data_buffer.append(trade)
                    
                    # Log periodically to avoid excessive logging
                    if len(self.data_buffer) % 100 == 0:
                        logging.debug(f"Processed {len(self.data_buffer)} trades")
            
            elif message.get("ev") == "status":
                # Status message
                status = message.get("status", "")
                message_text = message.get("message", "")
                logging.info(f"Received status message: {status} - {message_text}")
            
            elif message.get("ev") == "error":
                # Error message
                error_msg = message.get("message", "")
                logging.error(f"Received error message: {error_msg}")
                
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            logging.debug(f"Message content: {message}")
    
    def cleanup(self):
        """
        Clean up resources used by the WebSocket client
        
        This method should be called when the client is no longer needed
        to ensure proper release of resources.
        """
        logging.info("Cleaning up PolygonWebSocket resources")
        
        # Disconnect if running
        if self.running:
            self.disconnect()
        
        # Shutdown thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            logging.debug("Thread pool shut down")
        
        # Clear data structures
        if hasattr(self, 'data_buffer'):
            self.data_buffer.clear()
        
        if hasattr(self, 'message_queue'):
            self.message_queue.clear()
        
        # Force garbage collection to ensure memory is released
        import gc
        gc.collect()
        logging.debug("Memory buffers released")
