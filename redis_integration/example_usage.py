#!/usr/bin/env python3
"""
Example usage of Redis for signal handling in the GH200 Trading System

This script demonstrates how to use Redis for signal handling in the trading system.
It includes examples of publishing and subscribing to signals.
"""

import os
import sys
import time
import logging
import threading
import random

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Signal class from execution module
from src.execution.alpaca_execution import Signal

# Import Redis signal handler
from redis_integration.redis_client import RedisSignalHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Sample configuration
config = {
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "socket_timeout": 0.1,
        "socket_connect_timeout": 0.1,
        "socket_keepalive": True,
        "retry_on_timeout": True,
        "signals": {
            "raw_stream": "raw_signals",
            "validated_stream": "validated_signals",
            "channel": "trading_signals",
            "batch_size": 100,
            "max_stream_length": 10000
        }
    }
}

# Sample symbols
SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "TSLA", "NVDA", "AMD", "INTC", "IBM"
]

def generate_random_signal() -> Signal:
    """Generate a random trading signal"""
    symbol = random.choice(SYMBOLS)
    signal_type = random.choice(["BUY", "SELL"])
    price = round(random.uniform(100.0, 1000.0), 2)
    confidence = round(random.uniform(0.5, 1.0), 2)
    
    # Calculate stop loss and take profit based on signal type
    if signal_type == "BUY":
        stop_loss = round(price * (1.0 - random.uniform(0.01, 0.05)), 2)
        take_profit = round(price * (1.0 + random.uniform(0.02, 0.10)), 2)
    else:
        stop_loss = round(price * (1.0 + random.uniform(0.01, 0.05)), 2)
        take_profit = round(price * (1.0 - random.uniform(0.02, 0.10)), 2)
    
    # Create signal
    signal = Signal(
        symbol=symbol,
        type=signal_type,
        price=price,
        position_size=round(random.uniform(1.0, 10.0), 2),
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        timestamp=int(time.time() * 1_000_000)  # Microsecond precision
    )
    
    return signal

def signal_publisher(handler: RedisSignalHandler, count: int = 10, interval: float = 1.0):
    """
    Publish random signals at regular intervals
    
    Args:
        handler: Redis signal handler
        count: Number of signals to publish
        interval: Interval between signals in seconds
    """
    logging.info(f"Starting signal publisher (count={count}, interval={interval}s)")
    
    for i in range(count):
        # Generate random signal
        signal = generate_random_signal()
        
        # Publish signal
        success = handler.publish_signal(signal)
        
        if success:
            logging.info(f"Published signal {i+1}/{count}: {signal.symbol} {signal.type} @ {signal.price}")
        else:
            logging.error(f"Failed to publish signal {i+1}/{count}")
        
        # Wait for next interval
        time.sleep(interval)
    
    logging.info("Signal publisher finished")

def signal_validator(handler: RedisSignalHandler, validation_rate: float = 0.7):
    """
    Validate signals from the raw signal stream
    
    Args:
        handler: Redis signal handler
        validation_rate: Percentage of signals to validate (0.0-1.0)
    """
    logging.info(f"Starting signal validator (validation_rate={validation_rate})")
    
    # Define signal callback
    def on_signal(signal: Signal):
        # Randomly validate signals based on validation rate
        if random.random() < validation_rate:
            # Validate signal
            success = handler.publish_validated_signal(signal)
            
            if success:
                logging.info(f"Validated signal: {signal.symbol} {signal.type} @ {signal.price}")
            else:
                logging.error(f"Failed to validate signal: {signal.symbol}")
        else:
            logging.info(f"Rejected signal: {signal.symbol} {signal.type} @ {signal.price}")
    
    # Subscribe to signals
    handler.subscribe_to_signals(on_signal)
    
    logging.info("Signal validator started")

def signal_consumer(handler: RedisSignalHandler):
    """
    Consume validated signals
    
    Args:
        handler: Redis signal handler
    """
    logging.info("Starting signal consumer")
    
    # Define signal callback
    def on_signal(signal: Signal):
        logging.info(f"Received signal: {signal.symbol} {signal.type} @ {signal.price}")
        
        # Process signal (e.g., execute trade)
        logging.info(f"Processing signal: {signal.symbol} {signal.type} @ {signal.price}")
    
    # Subscribe to signals
    handler.subscribe_to_signals(on_signal)
    
    logging.info("Signal consumer started")

def main():
    """Main function"""
    logging.info("Starting Redis signal handling example")
    
    # Create signal handler
    handler = RedisSignalHandler(config)
    
    # Initialize handler
    if not handler.initialize():
        logging.error("Failed to initialize Redis signal handler")
        return
    
    try:
        # Start signal validator
        validator_thread = threading.Thread(
            target=signal_validator,
            args=(handler, 0.7),
            daemon=True
        )
        validator_thread.start()
        
        # Start signal consumer
        consumer_thread = threading.Thread(
            target=signal_consumer,
            args=(handler,),
            daemon=True
        )
        consumer_thread.start()
        
        # Wait for threads to start
        time.sleep(1.0)
        
        # Start signal publisher
        signal_publisher(handler, count=10, interval=1.0)
        
        # Wait for a while to see the results
        logging.info("Waiting for 5 seconds to see the results...")
        time.sleep(5.0)
        
        # Get latest signals
        raw_signals = handler.get_latest_signals(count=5)
        logging.info(f"Latest raw signals ({len(raw_signals)}):")
        for signal in raw_signals:
            logging.info(f"  {signal.symbol} {signal.type} @ {signal.price}")
        
        # Get latest validated signals
        validated_signals = handler.get_latest_validated_signals(count=5)
        logging.info(f"Latest validated signals ({len(validated_signals)}):")
        for signal in validated_signals:
            logging.info(f"  {signal.symbol} {signal.type} @ {signal.price}")
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        # Clean up
        handler.cleanup()
        logging.info("Redis signal handler cleaned up")
    
    logging.info("Redis signal handling example finished")

if __name__ == "__main__":
    main()
