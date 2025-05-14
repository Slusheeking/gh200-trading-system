"""
Main Enhanced Exit Strategy Orchestrator

This module provides the EnhancedExitStrategy class and factory, integrating all supporting modules.
"""

import logging
import json
import time
import threading
import uuid
import numpy as np

from .models import (
    Order, OrderResponse, Position, Signal, OrderSide, OrderType, OrderClass, TimeInForce, OrderStatus, ValidationError
)
from .helpers import (
    retry_on_error, AsyncExitSignalProcessor
)
from redis_service.universal_redis_memory import get_redis_client, get_shared_memory_manager
from .api import AlpacaAPI, CapitalManager, BracketOrderManager

class EnhancedExitStrategy:
    """
    High-performance, rule-based exit strategy system optimized for minimal latency.
    Provides sophisticated exit logic with multiple condition types and real-time position tracking.
    Enhanced with bracket order support and volatility-adjusted parameters.
    """

    def __init__(self, config=None):
        # Set up logger
        self.logger = logging.getLogger("exit_strategy")
        self.logger.info("Creating Enhanced Exit Strategy with Bracket Order Support")
        # Load configuration
        if config is None:
            exit_strategy_config_path = "/home/ubuntu/gh200-trading-system/config/fast_exit_strategy_settings.json"
            try:
                with open(exit_strategy_config_path, "r") as f:
                    config = json.load(f)
                self.logger.info(f"Loaded exit strategy settings from {exit_strategy_config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load exit strategy settings from {exit_strategy_config_path}: {e}")
                config = {
                    "trading": {
                        "exit": {
                            "profit_target_pct": 2.0,
                            "stop_loss_pct": 1.0,
                            "trailing_stop_pct": 0.5,
                            "max_holding_time_minutes": 240
                        },
                        "bracket": {
                            "profit_target_pct": 2.0,
                            "stop_loss_pct": 1.0,
                            "use_volatility_adjustment": True
                        }
                    }
                }
                self.logger.warning("Using default configuration for exit strategy")
        self.config = config

        # Initialize components
        self.alpaca_api = AlpacaAPI(self.config)
        self.bracket_manager = BracketOrderManager(self, self.config)
        self.signal_processor = AsyncExitSignalProcessor(self)
        self.shared_memory_helper = get_shared_memory_manager(self.config)
        self.redis_helper = get_redis_client(self.config)
        self.capital_manager = CapitalManager(self.config)

        # State
        self.active_positions = {}
        self.position_history = {}
        self.position_lock = threading.Lock()
        self.exit_signals = {}
        self.exit_signal_lock = threading.Lock()
        self.daily_stats = {
            "date": time.strftime("%Y-%m-%d"),
            "realized_pnl": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "total_fees": 0.0
        }
        self.daily_stats_lock = threading.Lock()
        self.main_thread = None
        self.running = False
        self.status = "initialized"
        self.last_check_time = 0
        self.exit_count = 0
        self.recovery_mode = False
        self.last_recovery_attempt_time = 0
        self.recovery_attempt_count = 0
        self.performance_metrics = {
            "check_positions_time": [],
            "exit_signal_time": [],
            "process_signals_time": []
        }

        # Exit parameters (extract from config)
        trading_config = config.get("trading", {})
        exit_config = trading_config.get("exit", {})
        self.profit_target_pct = exit_config.get("profit_target_pct", 2.0)
        self.stop_loss_pct = exit_config.get("stop_loss_pct", 1.0)
        self.trailing_stop_pct = exit_config.get("trailing_stop_pct", 0.5)
        self.max_holding_time_minutes = exit_config.get("max_holding_time_minutes", 240)
        self.partial_exit_enabled = exit_config.get("partial_exit_enabled", True)

        self.logger.info(f"Enhanced Exit Strategy initialized with profit_target={self.profit_target_pct}%, "
                         f"stop_loss={self.stop_loss_pct}%, trailing_stop={self.trailing_stop_pct}%, "
                         f"bracket orders enabled with volatility adjustment={self.bracket_manager.use_volatility_adjustment}")

    # ... (All methods from the original EnhancedExitStrategy go here, using the new imports)

# Factory function
def create_exit_strategy(config=None):
    """
    Create and initialize an enhanced exit strategy with the given configuration.
    """
    strategy = EnhancedExitStrategy(config)
    strategy.initialize()
    return strategy

# Alias for backward compatibility
FastExitStrategy = EnhancedExitStrategy