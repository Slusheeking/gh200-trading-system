#!/usr/bin/env python3
"""
Polygon Market Snapshot Fetcher

This script fetches a complete snapshot of the US stock market from Polygon.io
and saves it to a file for training data purposes.
"""

import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Import the PolygonMarketDataProvider using relative import
from polygon_market_data_provider import PolygonMarketDataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("market_snapshot")


def save_snapshot_to_file(snapshot_data: Dict[str, Any], output_dir: str = "data/market_snapshots") -> str:
    """
    Save market snapshot data to a file
    
    Args:
        snapshot_data: Market snapshot data
        output_dir: Directory to save the file
        
    Returns:
        str: Path to the saved file
    """
    if snapshot_data is None:
        logger.error("No data to save")
        return None
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ticker_count = snapshot_data.get("count", 0)
    filename = f"market_snapshot_{timestamp}_{ticker_count}_tickers.json"
    file_path = output_path / filename
    
    try:
        # Save data to file
        with open(file_path, "w") as f:
            json.dump(snapshot_data, f, indent=2)
        
        logger.info(f"Snapshot saved to {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving snapshot to file: {e}")
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fetch and save a complete US stock market snapshot from Polygon.io")
    parser.add_argument("--include-otc", action="store_true", help="Include OTC securities in the snapshot")
    parser.add_argument("--output-dir", default="data/market_snapshots", help="Directory to save the snapshot file")
    parser.add_argument("--min-price", type=float, default=5.0, help="Filter out stocks below this price (default: $5.00)")
    parser.add_argument("--max-price", type=float, default=500.0, help="Filter out stocks above this price (default: $500.00)")
    parser.add_argument("--min-volume", type=int, default=500000, help="Filter out stocks with volume below this threshold (default: 500,000)")
    parser.add_argument("--min-market-cap", type=float, default=500000000, help="Filter out stocks with market cap below this threshold (default: $500M)")
    parser.add_argument("--no-filter", action="store_true", help="Disable all filtering")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create a minimal config with filtering settings
    config = {
        "trading": {
            "market": {
                "min_price": args.min_price,
                "max_price": args.max_price,
                "min_volume": args.min_volume,
                "min_market_cap": args.min_market_cap
            }
        }
    }
    
    # Create the market data provider
    provider = PolygonMarketDataProvider(config)
    
    # Fetch market snapshot
    start_time = time.time()
    snapshot_data = provider.fetch_full_market_snapshot(
        include_otc=args.include_otc,
        apply_filters=not args.no_filter
    )
    
    if snapshot_data:
        # Save snapshot to file
        file_path = save_snapshot_to_file(snapshot_data, output_dir=args.output_dir)
        
        if file_path:
            elapsed_time = time.time() - start_time
            logger.info(f"Market snapshot fetched and saved in {elapsed_time:.2f} seconds")
            
            # Clean up resources
            provider.cleanup()
            return 0
    
    # Clean up resources
    provider.cleanup()
    return 1


if __name__ == "__main__":
    exit(main())