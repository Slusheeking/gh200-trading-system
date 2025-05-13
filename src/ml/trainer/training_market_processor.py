"""
Training Market Data Processor

This module extends the production MarketDataProcessor to support training workflows
while maintaining identical processing logic to ensure environment parity between
training and production.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any

from src.data.market_data_processor import MarketDataProcessor
from src.data.polygon_rest_api import ParsedMarketData
from src.monitoring.log import logging as log

class TrainingMarketDataProcessor(MarketDataProcessor):
    """
    Extends the production MarketDataProcessor to support training workflows
    while maintaining identical processing logic.
    
    This class ensures that data processing during training is identical to
    production, eliminating training-serving skew.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with same configuration as production
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = log.setup_logger("training_market_processor")
        self.logger.info("Initialized TrainingMarketDataProcessor with production-identical configuration")
    
    def process_historical_snapshots(self, snapshots: List[Dict[str, Any]]) -> List[ParsedMarketData]:
        """
        Process a batch of historical market snapshots using the same
        logic as production's process_rest_data method
        
        Args:
            snapshots: List of historical market snapshots
            
        Returns:
            List of processed snapshots with features computed identically to production
        """
        processed_snapshots = []
        
        for snapshot in snapshots:
            # Convert snapshot to ParsedMarketData if needed
            if not isinstance(snapshot, ParsedMarketData):
                parsed_snapshot = self._convert_to_parsed_market_data(snapshot)
            else:
                parsed_snapshot = snapshot
                
            # Use production's method directly to ensure identical processing
            processed_snapshot = self.process_rest_data(parsed_snapshot)
            processed_snapshots.append(processed_snapshot)
            
        self.logger.info(f"Processed {len(processed_snapshots)} historical snapshots using production-identical pipeline")
        return processed_snapshots
    
    def _convert_to_parsed_market_data(self, snapshot: Dict[str, Any]) -> ParsedMarketData:
        """
        Convert a dictionary snapshot to ParsedMarketData format
        
        Args:
            snapshot: Dictionary containing market snapshot data
            
        Returns:
            ParsedMarketData object
        """
        parsed_data = ParsedMarketData()
        
        # Set timestamp
        parsed_data.timestamp = snapshot.get("timestamp", 0)
        
        # Set symbol data
        if "symbols" in snapshot:
            for symbol, symbol_data in snapshot["symbols"].items():
                parsed_symbol_data = ParsedMarketData.SymbolData()
                
                # Copy all attributes from the dictionary to the object
                for key, value in symbol_data.items():
                    setattr(parsed_symbol_data, key, value)
                
                parsed_data.symbol_data[symbol] = parsed_symbol_data
        
        return parsed_data
    
    def create_training_features(self, symbol: str, symbol_data: Dict[str, Any], model_type: str) -> np.ndarray:
        """
        Extract features for training using the same methods as production
        
        Args:
            symbol: Symbol to extract features for
            symbol_data: Symbol data dictionary
            model_type: Type of model (gbdt, axial_attention, lstm_gru)
            
        Returns:
            Numpy array of features
        """
        # Convert symbol data to DataFrame format expected by production methods
        market_data = self._convert_symbol_data_to_df(symbol_data)
        
        # Use the appropriate feature extraction method based on model type
        if model_type.lower() == 'gbdt':
            features = self.extract_gbdt_features(market_data)
            # Return the last row of features (most recent)
            if isinstance(features, pd.DataFrame) and not features.empty:
                return features.iloc[-1].values
            
        elif model_type.lower() == 'axial_attention':
            features = self.extract_axial_attention_features(market_data)
            if len(features) > 0:
                return features[-1]  # Last sequence
            
        elif model_type.lower() == 'lstm_gru':
            features = self.extract_lstm_gru_features(market_data)
            if len(features) > 0:
                return features[-1]  # Last sequence
        
        # Return empty array if no features could be extracted
        return np.array([])
    
    def _convert_symbol_data_to_df(self, symbol_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert symbol data dictionary to DataFrame format expected by production methods
        
        Args:
            symbol_data: Symbol data dictionary
            
        Returns:
            DataFrame with the structure expected by market data processor
        """
        # Create DataFrame with the structure expected by market data processor
        df = pd.DataFrame({
            'timestamp': [symbol_data.get('timestamp', 0)],
            'open': [symbol_data.get('open_price', symbol_data.get('last_price', 0))],
            'high': [symbol_data.get('high_price', symbol_data.get('last_price', 0))],
            'low': [symbol_data.get('low_price', symbol_data.get('last_price', 0))],
            'close': [symbol_data.get('last_price', 0)],
            'volume': [symbol_data.get('volume', 0)]
        })
        
        # Convert timestamp to datetime if it's a numeric value
        if isinstance(df['timestamp'].iloc[0], (int, float)):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def process_snapshots_with_latency(self, snapshots: List[Dict[str, Any]], model_type: str) -> Dict[str, Any]:
        """
        Process snapshots one at a time, measuring latency for each snapshot
        
        Args:
            snapshots: List of market snapshots
            model_type: Type of model (gbdt, axial_attention, lstm_gru)
            
        Returns:
            Dictionary with features, labels, and latency metrics
        """
        features = []
        labels = []
        symbols = []
        dates = []
        latencies = []
        symbol_latencies = {}
        
        self.logger.info(f"Processing {len(snapshots)} snapshots one at a time, measuring latency")
        
        # Process each snapshot individually, just like in production
        for snapshot_idx, snapshot in enumerate(snapshots):
            snapshot_time = snapshot.get("timestamp", 0)
            snapshot_start_time = time.time()
            
            if "symbols" not in snapshot:
                continue
                
            # Track per-symbol latencies for this snapshot
            snapshot_symbol_latencies = {}
            snapshot_features = []
            snapshot_labels = []
            snapshot_symbols = []
            
            # Process each symbol in the snapshot
            for symbol, symbol_data in snapshot["symbols"].items():
                symbol_start_time = time.time()
                
                # Extract features using production-identical methods
                symbol_features = self.create_training_features(symbol, symbol_data, model_type)
                
                if len(symbol_features) > 0:
                    snapshot_features.append(symbol_features)
                    
                    # Generate label based on model type
                    if model_type.lower() == 'gbdt':
                        # Binary classification label
                        price_change = symbol_data.get("price_change_5m", 0)
                        label = 1 if price_change > 0.005 else 0
                    elif model_type.lower() in ['axial_attention', 'lstm_gru']:
                        # Multi-class or regression label
                        price_change = symbol_data.get("price_change_5m", 0)
                        if price_change > 0.005:
                            label = [1, 0, 0]  # Up
                        elif price_change < -0.005:
                            label = [0, 0, 1]  # Down
                        else:
                            label = [0, 1, 0]  # Neutral
                    else:
                        # Default label
                        label = 0
                        
                    snapshot_labels.append(label)
                    snapshot_symbols.append(symbol)
                    
                    # Calculate symbol processing latency
                    symbol_latency = (time.time() - symbol_start_time) * 1000  # Convert to ms
                    snapshot_symbol_latencies[symbol] = symbol_latency
                    
                    # Update overall symbol latencies
                    if symbol not in symbol_latencies:
                        symbol_latencies[symbol] = []
                    symbol_latencies[symbol].append(symbol_latency)
            
            # Add snapshot data to overall results
            features.extend(snapshot_features)
            labels.extend(snapshot_labels)
            symbols.extend(snapshot_symbols)
            dates.extend([snapshot_time] * len(snapshot_features))
            
            # Calculate snapshot processing latency
            snapshot_latency = (time.time() - snapshot_start_time) * 1000  # Convert to ms
            latencies.append({
                "snapshot_idx": snapshot_idx,
                "timestamp": snapshot_time,
                "total_latency_ms": snapshot_latency,
                "symbols_processed": len(snapshot_features),
                "avg_symbol_latency_ms": sum(snapshot_symbol_latencies.values()) / max(1, len(snapshot_symbol_latencies)),
                "symbol_latencies": snapshot_symbol_latencies
            })
            
            # Log progress every 10 snapshots
            if (snapshot_idx + 1) % 10 == 0 or snapshot_idx == len(snapshots) - 1:
                self.logger.info(f"Processed {snapshot_idx + 1}/{len(snapshots)} snapshots, "
                               f"avg latency: {sum(latency_item['total_latency_ms'] for latency_item in latencies) / len(latencies):.2f} ms")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Calculate overall latency metrics
        avg_snapshot_latency = sum(latency_item["total_latency_ms"] for latency_item in latencies) / max(1, len(latencies))
        
        # Calculate per-symbol latency statistics
        symbol_latency_stats = {}
        for symbol, symbol_lats in symbol_latencies.items():
            symbol_latency_stats[symbol] = {
                "count": len(symbol_lats),
                "avg_ms": sum(symbol_lats) / len(symbol_lats),
                "min_ms": min(symbol_lats),
                "max_ms": max(symbol_lats),
                "p95_ms": sorted(symbol_lats)[int(0.95 * len(symbol_lats))] if len(symbol_lats) > 20 else max(symbol_lats)
            }
        
        # Sort symbols by average latency (descending)
        slowest_symbols = sorted(
            symbol_latency_stats.items(),
            key=lambda x: x[1]["avg_ms"],
            reverse=True
        )[:10]  # Top 10 slowest
        
        # Log latency statistics
        self.logger.info(f"Processed {len(snapshots)} snapshots with {len(features)} symbols")
        self.logger.info(f"Average snapshot processing latency: {avg_snapshot_latency:.2f} ms")
        self.logger.info(f"Top 5 slowest symbols: {slowest_symbols[:5]}")
        
        return {
            "X": X,
            "y": y,
            "symbols": symbols,
            "dates": dates,
            "latency_metrics": {
                "snapshot_latencies": latencies,
                "avg_snapshot_latency_ms": avg_snapshot_latency,
                "symbol_latency_stats": symbol_latency_stats,
                "slowest_symbols": slowest_symbols
            }
        }
        
    def batch_process_snapshots(self, snapshots: List[Dict[str, Any]], model_type: str) -> Dict[str, Any]:
        """
        Process a batch of snapshots and extract features for a specific model type
        
        Args:
            snapshots: List of market snapshots
            model_type: Type of model (gbdt, axial_attention, lstm_gru)
            
        Returns:
            Dictionary with features and labels
        """
        # Use the new method that processes one snapshot at a time and measures latency
        return self.process_snapshots_with_latency(snapshots, model_type)