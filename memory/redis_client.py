"""
Redis client for GH200 Trading System

This module provides Redis client classes for the trading system.
It includes classes for signal processing, distribution, and storage.
"""

import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import asdict

try:
    import memory
    from redis.exceptions import RedisError
except ImportError:
    logging.error("Redis Python client not installed. Please install with: pip install redis")
    raise

# Try to import Signal class from execution module
try:
    from src.execution.alpaca_execution import Signal
except ImportError:
    # Define a minimal Signal class if import fails
    from dataclasses import dataclass
    
    @dataclass
    class Signal:
        """Minimal Signal class if import fails"""
        symbol: str
        type: str
        price: float
        position_size: float = 0.0
        stop_loss: float = 0.0
        take_profit: float = 0.0
        confidence: float = 0.0
        timestamp: int = 0


class RedisClient:
    """
    Redis client for the trading system
    
    This class provides a wrapper around the Redis client with
    methods for common operations used in the trading system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Redis client
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Get Redis configuration
        redis_config = self.config.get("redis", {})
        self.host = redis_config.get("host", "localhost")
        self.port = redis_config.get("port", 6379)
        self.db = redis_config.get("db", 0)
        self.password = redis_config.get("password", None)
        
        # Performance settings
        self.socket_timeout = redis_config.get("socket_timeout", 0.1)  # 100ms timeout
        self.socket_connect_timeout = redis_config.get("socket_connect_timeout", 0.1)
        self.socket_keepalive = redis_config.get("socket_keepalive", True)
        self.retry_on_timeout = redis_config.get("retry_on_timeout", True)
        
        # Initialize Redis client
        self.redis = None
        self.connection_pool = None
        self.initialized = False
        
        # Thread safety
        self.lock = threading.RLock()
    
    def initialize(self) -> bool:
        """
        Initialize the Redis client
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Create connection pool
            self.connection_pool = memory.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                socket_keepalive=self.socket_keepalive,
                retry_on_timeout=self.retry_on_timeout,
                decode_responses=True  # Auto-decode responses to strings
            )
            
            # Create Redis client
            self.redis = memory.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.redis.ping()
            
            self.initialized = True
            logging.info(f"Redis client initialized: {self.host}:{self.port}")
            return True
            
        except RedisError as e:
            logging.error(f"Failed to initialize Redis client: {str(e)}")
            return False
    
    def get(self, key: str) -> Any:
        """
        Get a value from Redis
        
        Args:
            key: Key to get
            
        Returns:
            Value or None if key doesn't exist
        """
        if not self.initialized:
            logging.warning("Redis client not initialized")
            return None
            
        try:
            return self.redis.get(key)
        except RedisError as e:
            logging.error(f"Redis get error: {str(e)}")
            return None
    
    def set(self, key: str, value: str, expiry: Optional[int] = None) -> bool:
        """
        Set a value in Redis
        
        Args:
            key: Key to set
            value: Value to set
            expiry: Expiry time in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logging.warning("Redis client not initialized")
            return False
            
        try:
            return self.redis.set(key, value, ex=expiry)
        except RedisError as e:
            logging.error(f"Redis set error: {str(e)}")
            return False
    
    def publish(self, channel: str, message: str) -> int:
        """
        Publish a message to a channel
        
        Args:
            channel: Channel to publish to
            message: Message to publish
            
        Returns:
            Number of clients that received the message
        """
        if not self.initialized:
            logging.warning("Redis client not initialized")
            return 0
            
        try:
            return self.redis.publish(channel, message)
        except RedisError as e:
            logging.error(f"Redis publish error: {str(e)}")
            return 0
    
    def subscribe(self, channel: str, callback: Callable[[str, str], None]) -> bool:
        """
        Subscribe to a channel
        
        Args:
            channel: Channel to subscribe to
            callback: Callback function to call when a message is received
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logging.warning("Redis client not initialized")
            return False
            
        try:
            # Create a new Redis client for the subscription
            # (Redis clients with active subscriptions can't be used for other commands)
            pubsub = self.redis.pubsub()
            pubsub.subscribe(**{channel: lambda message: callback(channel, message["data"])})
            
            # Start a thread to listen for messages
            thread = threading.Thread(
                target=pubsub.run_in_thread,
                kwargs={"sleep_time": 0.001},  # 1ms sleep between checks
                daemon=True
            )
            thread.start()
            
            return True
        except RedisError as e:
            logging.error(f"Redis subscribe error: {str(e)}")
            return False
    
    def xadd(self, stream: str, fields: Dict[str, Any], max_len: Optional[int] = None) -> Optional[str]:
        """
        Add a message to a stream
        
        Args:
            stream: Stream name
            fields: Dictionary of field-value pairs
            max_len: Maximum length of the stream (optional)
            
        Returns:
            Message ID if successful, None otherwise
        """
        if not self.initialized:
            logging.warning("Redis client not initialized")
            return None
            
        try:
            # Convert values to strings
            string_fields = {k: str(v) for k, v in fields.items()}
            
            # Add to stream
            if max_len is not None:
                return self.redis.xadd(stream, string_fields, maxlen=max_len)
            else:
                return self.redis.xadd(stream, string_fields)
        except RedisError as e:
            logging.error(f"Redis xadd error: {str(e)}")
            return None
    
    def xread(self, streams: Dict[str, str], count: Optional[int] = None, block: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Read from streams
        
        Args:
            streams: Dictionary of stream names and IDs
            count: Maximum number of messages to read (optional)
            block: Block for this many milliseconds (optional)
            
        Returns:
            List of messages if successful, None otherwise
        """
        if not self.initialized:
            logging.warning("Redis client not initialized")
            return None
            
        try:
            return self.redis.xread(streams, count=count, block=block)
        except RedisError as e:
            logging.error(f"Redis xread error: {str(e)}")
            return None
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the Redis client
        
        This method should be called when the client is no longer needed
        to ensure proper release of resources.
        """
        if self.connection_pool:
            self.connection_pool.disconnect()
            logging.info("Redis connection pool disconnected")


class RedisSignalHandler:
    """
    Signal handler using Redis
    
    This class provides methods for processing and distributing trading signals
    using Redis streams and pub/sub.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signal handler
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Get Redis configuration
        redis_config = self.config.get("redis", {})
        signal_config = redis_config.get("signals", {})
        
        # Stream and channel names
        self.raw_signal_stream = signal_config.get("raw_stream", "raw_signals")
        self.validated_signal_stream = signal_config.get("validated_stream", "validated_signals")
        self.signal_channel = signal_config.get("channel", "trading_signals")
        
        # Performance settings
        self.batch_size = signal_config.get("batch_size", 100)
        self.max_stream_length = signal_config.get("max_stream_length", 10000)
        
        # Initialize Redis client
        self.redis_client = RedisClient(config)
        self.initialized = False
        
        # Signal callbacks
        self.signal_callbacks = []
    
    def initialize(self) -> bool:
        """
        Initialize the signal handler
        
        Returns:
            True if initialization was successful, False otherwise
        """
        # Initialize Redis client
        if not self.redis_client.initialize():
            return False
            
        self.initialized = True
        logging.info("Redis signal handler initialized")
        return True
    
    def publish_signal(self, signal: Signal) -> bool:
        """
        Publish a signal to Redis
        
        Args:
            signal: Trading signal
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logging.warning("Redis signal handler not initialized")
            return False
            
        try:
            # Convert signal to dictionary
            if hasattr(signal, "to_dict"):
                signal_dict = signal.to_dict()
            elif hasattr(signal, "__dict__"):
                signal_dict = signal.__dict__
            else:
                signal_dict = asdict(signal)
                
            # Add timestamp if not present
            if "timestamp" not in signal_dict or not signal_dict["timestamp"]:
                signal_dict["timestamp"] = int(time.time() * 1_000_000)  # Microsecond precision
                
            # Add to raw signal stream
            stream_id = self.redis_client.xadd(
                self.raw_signal_stream,
                signal_dict,
                max_len=self.max_stream_length
            )
            
            if not stream_id:
                return False
                
            # Also publish to signal channel as JSON
            json_signal = json.dumps(signal_dict)
            recipients = self.redis_client.publish(self.signal_channel, json_signal)
            
            logging.debug(f"Signal published: {signal.symbol} (received by {recipients} subscribers)")
            return True
            
        except Exception as e:
            logging.error(f"Error publishing signal: {str(e)}")
            return False
    
    def publish_validated_signal(self, signal: Signal) -> bool:
        """
        Publish a validated signal to Redis
        
        Args:
            signal: Validated trading signal
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logging.warning("Redis signal handler not initialized")
            return False
            
        try:
            # Convert signal to dictionary
            if hasattr(signal, "to_dict"):
                signal_dict = signal.to_dict()
            elif hasattr(signal, "__dict__"):
                signal_dict = signal.__dict__
            else:
                signal_dict = asdict(signal)
                
            # Add timestamp if not present
            if "timestamp" not in signal_dict or not signal_dict["timestamp"]:
                signal_dict["timestamp"] = int(time.time() * 1_000_000)  # Microsecond precision
                
            # Add validation timestamp
            signal_dict["validated_at"] = int(time.time() * 1_000_000)
                
            # Add to validated signal stream
            stream_id = self.redis_client.xadd(
                self.validated_signal_stream,
                signal_dict,
                max_len=self.max_stream_length
            )
            
            if not stream_id:
                return False
                
            # Also publish to signal channel with validation flag
            signal_dict["validated"] = True
            json_signal = json.dumps(signal_dict)
            recipients = self.redis_client.publish(self.signal_channel, json_signal)
            
            logging.debug(f"Validated signal published: {signal.symbol} (received by {recipients} subscribers)")
            return True
            
        except Exception as e:
            logging.error(f"Error publishing validated signal: {str(e)}")
            return False
    
    def subscribe_to_signals(self, callback: Callable[[Signal], None]) -> bool:
        """
        Subscribe to signals
        
        Args:
            callback: Callback function to call when a signal is received
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logging.warning("Redis signal handler not initialized")
            return False
            
        # Add callback to list
        self.signal_callbacks.append(callback)
        
        # Define message handler
        def handle_message(channel: str, message: str) -> None:
            try:
                # Parse JSON message
                signal_dict = json.loads(message)
                
                # Create Signal object
                signal = Signal(
                    symbol=signal_dict.get("symbol", ""),
                    type=signal_dict.get("type", ""),
                    price=float(signal_dict.get("price", 0.0)),
                    position_size=float(signal_dict.get("position_size", 0.0)),
                    stop_loss=float(signal_dict.get("stop_loss", 0.0)),
                    take_profit=float(signal_dict.get("take_profit", 0.0)),
                    confidence=float(signal_dict.get("confidence", 0.0)),
                    timestamp=int(signal_dict.get("timestamp", 0))
                )
                
                # Call callback
                callback(signal)
                
            except Exception as e:
                logging.error(f"Error handling signal message: {str(e)}")
        
        # Subscribe to channel
        return self.redis_client.subscribe(self.signal_channel, handle_message)
    
    def get_latest_signals(self, count: int = 10) -> List[Signal]:
        """
        Get latest signals from the raw signal stream
        
        Args:
            count: Maximum number of signals to get
            
        Returns:
            List of signals
        """
        if not self.initialized:
            logging.warning("Redis signal handler not initialized")
            return []
            
        try:
            # Read from stream
            result = self.redis_client.redis.xrevrange(
                self.raw_signal_stream,
                count=count
            )
            
            signals = []
            for _, fields in result:
                try:
                    # Create Signal object
                    signal = Signal(
                        symbol=fields.get("symbol", ""),
                        type=fields.get("type", ""),
                        price=float(fields.get("price", 0.0)),
                        position_size=float(fields.get("position_size", 0.0)),
                        stop_loss=float(fields.get("stop_loss", 0.0)),
                        take_profit=float(fields.get("take_profit", 0.0)),
                        confidence=float(fields.get("confidence", 0.0)),
                        timestamp=int(fields.get("timestamp", 0))
                    )
                    
                    signals.append(signal)
                except Exception as e:
                    logging.error(f"Error creating signal from stream: {str(e)}")
            
            return signals
            
        except RedisError as e:
            logging.error(f"Redis get_latest_signals error: {str(e)}")
            return []
    
    def get_latest_validated_signals(self, count: int = 10) -> List[Signal]:
        """
        Get latest validated signals from the validated signal stream
        
        Args:
            count: Maximum number of signals to get
            
        Returns:
            List of validated signals
        """
        if not self.initialized:
            logging.warning("Redis signal handler not initialized")
            return []
            
        try:
            # Read from stream
            result = self.redis_client.redis.xrevrange(
                self.validated_signal_stream,
                count=count
            )
            
            signals = []
            for _, fields in result:
                try:
                    # Create Signal object
                    signal = Signal(
                        symbol=fields.get("symbol", ""),
                        type=fields.get("type", ""),
                        price=float(fields.get("price", 0.0)),
                        position_size=float(fields.get("position_size", 0.0)),
                        stop_loss=float(fields.get("stop_loss", 0.0)),
                        take_profit=float(fields.get("take_profit", 0.0)),
                        confidence=float(fields.get("confidence", 0.0)),
                        timestamp=int(fields.get("timestamp", 0))
                    )
                    
                    signals.append(signal)
                except Exception as e:
                    logging.error(f"Error creating signal from stream: {str(e)}")
            
            return signals
            
        except RedisError as e:
            logging.error(f"Redis get_latest_validated_signals error: {str(e)}")
            return []
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the signal handler
        
        This method should be called when the handler is no longer needed
        to ensure proper release of resources.
        """
        if self.redis_client:
            self.redis_client.cleanup()
