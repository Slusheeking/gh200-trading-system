#!/usr/bin/env python3
"""
Test script for Polygon REST API and WebSocket clients

This script tests the Polygon REST API and WebSocket clients using the
configuration from environment variables or system.yaml if available.
"""

import os
import time
import datetime
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import config_loader
sys.path.append(str(Path(__file__).parents[1]))
from config.config_loader import get_config
from polygon_rest_api import PolygonRestAPI, ParsedMarketData
from polygon_websocket import PolygonWebSocket, MarketData
from market_data_processor import MarketDataProcessor
from monitoring.log.logging import setup_logger

# Configure logging
logger = setup_logger("polygon_test", log_to_console=True)

def print_market_data(data: ParsedMarketData):
    """Print market data received from REST API"""
    logger.info(f"Received market data with {len(data.symbol_data)} symbols")
    # Convert nanosecond timestamp to datetime
    dt = datetime.datetime.fromtimestamp(data.timestamp / 1_000_000_000)
    logger.info(f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    
    # Print first 5 symbols (or fewer if less available)
    count = 0
    for symbol, symbol_data in data.symbol_data.items():
        if count >= 5:
            break
        
        logger.info(f"Symbol: {symbol}")
        logger.info(f"  Last price: {symbol_data.last_price}")
        logger.info(f"  Bid/Ask: {symbol_data.bid_price}/{symbol_data.ask_price}")
        logger.info(f"  Volume: {symbol_data.volume}")
        
        # Display technical indicators
        logger.info("  Technical Indicators:")
        logger.info(f"    RSI (14): {symbol_data.rsi_14:.2f}")
        logger.info(f"    MACD: {symbol_data.macd:.4f}")
        logger.info(f"    MACD Signal: {symbol_data.macd_signal:.4f}")
        logger.info(f"    MACD Histogram: {symbol_data.macd_histogram:.4f}")
        logger.info(f"    Bollinger Bands: {symbol_data.bb_lower:.2f} / {symbol_data.bb_middle:.2f} / {symbol_data.bb_upper:.2f}")
        logger.info(f"    ATR: {symbol_data.atr:.4f}")
        logger.info(f"    Volume Spike: {symbol_data.volume_spike:.2f}")
        logger.info(f"    SMA Cross Signal: {symbol_data.sma_cross_signal:.4f}")
        logger.info("  ----")
        count += 1
    
    logger.info(f"... and {len(data.symbol_data) - count} more symbols")

def test_rest_api():
    """Test the Polygon REST API client"""
    logger.info("Testing Polygon REST API client...")
    
    # Load configuration (will be empty if system.yaml doesn't exist)
    config = get_config()

    # Ensure we have a minimal configuration structure if config is empty
    if not config:
        config = {
            "data_sources": {
                "polygon": {
                    "enabled": True,
                    "websocket_url": "wss://socket.polygon.io/stocks"
                }
            },
            "performance": {
                "processor_threads": 4,
                "websocket_parser_batch_size": 1000
            }
        }
    
    # Initialize clients and processor
    rest_client = PolygonRestAPI(config)
    rest_client.initialize()
    
    processor = MarketDataProcessor(config)
    
    # Check if API key is available
    if not rest_client.polygon_api_key:
        logger.error("No Polygon API key found. Make sure to set POLYGON_API_KEY in your .env file")
        return False
    
    # Log API key (first 4 and last 4 characters only)
    api_key = rest_client.polygon_api_key
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else f"{api_key[:4]}..."
    
    logger.info(f"Using Polygon API key: {masked_key}")

    try:
        # Test fetching data for specific symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        logger.info(f"Fetching data for symbols: {', '.join(symbols)}")
        
        future = rest_client.fetch_symbol_data(symbols)
        raw_data = future.result()
        
        # Process the data
        processed_data = processor.process_rest_data(raw_data)
        
        print_market_data(processed_data)
        
        return True
    except Exception as e:
        logger.error(f"Error testing REST API: {e}")
        return False

def on_websocket_data(data: MarketData):
    """Callback for WebSocket data"""
    logger.info(f"Received WebSocket data with {len(data.trades)} trades")
    
    # Print first 5 trades (or fewer if less available)
    # Convert deque to list for slicing
    trades_list = list(data.trades)
    for i, trade in enumerate(trades_list[:5]):
        logger.info(f"Trade {i+1}:")
        logger.info(f"  Symbol: {trade.symbol}")
        logger.info(f"  Price: {trade.price}")
        logger.info(f"  Size: {trade.size}")
        logger.info(f"  Timestamp: {trade.timestamp}")
        logger.info(f"  Exchange: {trade.exchange}")
        logger.info("  ----")
    
    if len(data.trades) > 5:
        logger.info(f"... and {len(data.trades) - 5} more trades")

def test_websocket():
    """Test the Polygon WebSocket client"""
    logger.info("Testing Polygon WebSocket client...")
    
    # Load configuration (will be empty if system.yaml doesn't exist)
    config = get_config()
    
    # Ensure we have a minimal configuration structure if config is empty
    if not config:
        config = {
            "data_sources": {
                "polygon": {
                    "enabled": True,
                    "websocket_url": "wss://socket.polygon.io/stocks",
                    "subscription_type": "T.AAPL,T.MSFT,T.GOOGL,T.AMZN,T.META"
                }
            },
            "performance": {
                "processor_threads": 4,
                "websocket_parser_batch_size": 1000
            }
        }
    # Override subscription type for testing with a smaller set of symbols
    elif "data_sources" in config and "polygon" in config["data_sources"]:
        config["data_sources"]["polygon"]["subscription_type"] = "T.AAPL,T.MSFT,T.GOOGL,T.AMZN,T.META"
    
    # Initialize client
    websocket_client = PolygonWebSocket(config)
    processor = MarketDataProcessor(config)
    
    # Check if any data sources are configured
    if not websocket_client.data_sources:
        logger.error("No data sources configured. Make sure to set POLYGON_API_KEY in your .env file")
        return False
    
    # Log the data sources
    for source in websocket_client.data_sources:
        logger.info(f"Using data source: {source.name} at {source.url}")
    
    try:
        # Connect to WebSocket
        logger.info("Connecting to WebSocket...")
        websocket_client.connect()
        
        # Create a market data object to receive data
        market_data = MarketData()
        
        # Wait for some data (up to 30 seconds)
        logger.info("Waiting for WebSocket data (up to 30 seconds)...")
        for _ in range(30):
            websocket_client.get_latest_data(market_data)
            
            if market_data.trades:
                # Process the data
                processed_data = processor.process_websocket_data(market_data)
                on_websocket_data(processed_data)
                break
            
            time.sleep(1)
        
        # Disconnect
        logger.info("Disconnecting from WebSocket...")
        websocket_client.disconnect()
        
        if not market_data.trades:
            logger.warning("No WebSocket data received within timeout")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error testing WebSocket: {e}")
        return False

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = get_config()
    
    # If config is empty or polygon is not explicitly disabled, consider it enabled
    # This allows the script to run even without a system.yaml file
    polygon_enabled = True
    if config and "data_sources" in config and "polygon" in config["data_sources"]:
        polygon_enabled = config["data_sources"]["polygon"].get("enabled", True)
    
    if not polygon_enabled:
        logger.error("Polygon data source is explicitly disabled in system.yaml")
        return
    
    # Check if API key is set
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        logger.error("Please create a .env file with your Polygon API key")
        logger.error("Example: POLYGON_API_KEY=your_api_key_here")
        return
    
    if config:
        logger.info("Using configuration from system.yaml")
    else:
        logger.info("No system.yaml found, using default configuration")
    
    # Test REST API
    rest_success = test_rest_api()
    
    # Test WebSocket
    websocket_success = test_websocket()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"REST API Test: {'SUCCESS' if rest_success else 'FAILED'}")
    logger.info(f"WebSocket Test: {'SUCCESS' if websocket_success else 'FAILED'}")

if __name__ == "__main__":
    main()
