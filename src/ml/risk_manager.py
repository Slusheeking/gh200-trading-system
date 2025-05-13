
"""
Risk management implementation
"""

import time
import threading
from typing import Dict, List

# Import from project modules
from config.config_loader import get_config
from src.monitoring.log import logging
from src.execution.alpaca_execution import Signal

# Define pin_thread_to_core function
def pin_thread_to_core(core_id: int) -> None:
    """
    Set thread affinity to a specific CPU core.
    
    Args:
        core_id: CPU core ID to pin the thread to
    """
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([core_id])
        logging.info(f"Thread affinity set to core {core_id}")
    except (ImportError, AttributeError):
        logging.warning(f"Could not set thread affinity to core {core_id}")

class Position:
    """Position class to store position information"""
    def __init__(self):
        self.quantity = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.risk_amount = 0.0
        self.entry_time = None
        self.last_update_time = None


class RiskManager:
    """
    Risk manager class responsible for validating trading signals
    and managing position risk.
    """
    
    def __init__(self, config=None):
        """
        Initialize the risk manager with configuration parameters.
        
        Args:
            config: Trading system configuration (optional)
        """
        # Get configuration
        if config is None:
            config = get_config()
            
        # Extract risk parameters from config
        trading_config = config.get("trading", {})
        risk_config = trading_config.get("risk", {})
        account_config = trading_config.get("account", {})
        
        self.max_position_size_pct = risk_config.get("max_position_size_pct", 5.0)
        self.position_sizing_method = risk_config.get("position_sizing_method", "fixed")
        self.kelly_fraction = risk_config.get("kelly_fraction", 0.5)
        self.max_daily_drawdown_pct = risk_config.get("max_daily_drawdown_pct", 3.0)
        self.max_total_risk_pct = (self.max_position_size_pct *
                                  account_config.get("max_positions", 10))
        self.account_value = account_config.get("initial_capital", 100000.0)
        
        # Initialize tracking variables
        self.daily_pnl = 0.0
        self.total_risk = 0.0
        
        # Initialize positions dictionary and lock
        self.positions = {}  # Dict[str, Position]
        self.positions_lock = threading.Lock()
        
        # Thread management
        self.thread_id = None
    
    def validate_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Validate trading signals based on risk parameters.
        
        Args:
            signals: List of trading signals to validate
            
        Returns:
            List of validated signals that pass risk checks
        """
        # Start timing
        start_time = time.time()
        
        validated_signals = []
        
        # Skip if no signals
        if not signals:
            return validated_signals
        
        # Process each signal
        for signal in signals:
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            
            # Skip if position size is too small
            if position_size <= 0.0:
                continue
            
            # Check if signal passes risk checks
            if self.passes_risk_checks(signal, position_size):
                # Set position size
                signal.position_size = position_size
                
                # Add to validated signals
                validated_signals.append(signal)
        
        # End timing
        end_time = time.time()
        duration_us = (end_time - start_time) * 1_000_000  # Convert to microseconds
        
        logging.info(f"Risk validation completed in {duration_us:.2f} µs, validated "
                    f"{len(validated_signals)} of {len(signals)} signals")
        
        return validated_signals
    
    def set_thread_affinity(self, core_id: int):
        """
        Set thread affinity to a specific CPU core.
        
        Args:
            core_id: CPU core ID to pin the thread to
        """
        # Store thread ID
        self.thread_id = threading.get_ident()
        
        # Set affinity
        pin_thread_to_core(core_id)
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get current positions.
        
        Returns:
            Dictionary of positions keyed by symbol
        """
        with self.positions_lock:
            # Return a copy to avoid threading issues
            return self.positions.copy()
    
    def update_position(self, symbol: str, position: Position):
        """
        Update a position for a symbol.
        
        Args:
            symbol: Trading symbol
            position: Updated position information
        """
        with self.positions_lock:
            # Calculate PnL change
            old_unrealized_pnl = 0.0
            if symbol in self.positions:
                old_unrealized_pnl = self.positions[symbol].unrealized_pnl
            
            # Update position
            self.positions[symbol] = position
            
            # Update daily P&L
            self.daily_pnl += position.unrealized_pnl - old_unrealized_pnl
            
            # Update total risk
            self.total_risk = sum(p.risk_amount for p in self.positions.values())
    
    def remove_position(self, symbol: str):
        """
        Remove a position for a symbol.
        
        Args:
            symbol: Trading symbol to remove
        """
        with self.positions_lock:
            # Update daily P&L before removing
            if symbol in self.positions:
                self.daily_pnl += self.positions[symbol].realized_pnl
            
            # Remove position
            if symbol in self.positions:
                del self.positions[symbol]
            
            # Update total risk
            self.total_risk = sum(p.risk_amount for p in self.positions.values())
    
    def calculate_position_size(self, signal: Signal) -> float:
        """
        Calculate position size based on the configured sizing method.
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size
        """
        # Use specified position sizing method
        if self.position_sizing_method == "fixed":
            # Fixed percentage of account
            return self.account_value * (self.max_position_size_pct / 100.0)
        
        elif self.position_sizing_method == "kelly":
            # Kelly criterion
            return self.calculate_kelly_position_size(signal)
        
        elif self.position_sizing_method == "volatility":
            # Volatility-based sizing
            # Check if signal has indicators attribute
            indicators = getattr(signal, "indicators", {})
            atr = indicators.get("atr", signal.price * 0.01) if isinstance(indicators, dict) else (signal.price * 0.01)
            
            # Size based on ATR
            risk_per_share = atr * 2.0  # 2x ATR for stop loss
            max_risk_amount = self.account_value * (self.max_position_size_pct / 100.0) * 0.01  # 1% risk
            
            return max_risk_amount / risk_per_share
        
        else:
            # Default to fixed percentage
            return self.account_value * (self.max_position_size_pct / 100.0)
    
    def passes_risk_checks(self, signal: Signal, position_size: float) -> bool:
        """
        Check if a signal passes all risk checks.
        
        Args:
            signal: Trading signal
            position_size: Calculated position size
            
        Returns:
            True if signal passes all risk checks, False otherwise
        """
        # Check daily drawdown
        if self.daily_pnl < -self.account_value * (self.max_daily_drawdown_pct / 100.0):
            logging.warning(f"Daily drawdown limit reached, rejecting signal for {signal.symbol}")
            return False
        
        # Check total risk
        signal_risk = position_size * 0.01  # Assume 1% risk per trade
        if self.total_risk + signal_risk > self.account_value * (self.max_total_risk_pct / 100.0):
            logging.warning(f"Total risk limit reached, rejecting signal for {signal.symbol}")
            return False
        
        # Check existing position
        with self.positions_lock:
            if signal.symbol in self.positions:
                logging.warning(f"Position already exists for {signal.symbol}, rejecting signal")
                return False
        
        # Check portfolio constraints
        if not self.check_portfolio_constraints(signal, position_size):
            logging.warning(f"Portfolio constraints not met, rejecting signal for {signal.symbol}")
            return False
        
        return True
    
    def calculate_kelly_position_size(self, signal: Signal) -> float:
        """
        Calculate position size using the Kelly criterion.
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size based on Kelly criterion
        """
        # Kelly formula: f* = (bp - q) / b
        # where:
        # f* = fraction of bankroll to bet
        # b = odds received on the bet (profit/loss ratio)
        # p = probability of winning
        # q = probability of losing (1 - p)
        
        # Estimate probability of winning from signal confidence
        p = signal.confidence
        q = 1.0 - p
        
        # Estimate profit/loss ratio from take profit and stop loss
        take_profit = signal.take_profit if signal.take_profit > 0.0 else (signal.price * 1.03)  # Default 3% profit
        stop_loss = signal.stop_loss if signal.stop_loss > 0.0 else (signal.price * 0.98)  # Default 2% loss
        
        profit = take_profit - signal.price
        loss = signal.price - stop_loss
        
        b = profit / loss
        
        # Calculate Kelly fraction
        f = (b * p - q) / b
        
        # Apply Kelly fraction and cap at max position size
        f = max(0.0, f) * self.kelly_fraction
        f = min(f, self.max_position_size_pct / 100.0)
        
        return self.account_value * f
    
    def check_portfolio_constraints(self, signal: Signal, position_size: float) -> bool:
        """
        Check if adding a position would violate portfolio constraints.
        
        Args:
            signal: Trading signal
            position_size: Calculated position size
            
        Returns:
            True if portfolio constraints are satisfied, False otherwise
        """
        # This is a simplified implementation
        # In a real system, this would check sector exposure, correlation, etc.
        
        # Check maximum number of positions
        with self.positions_lock:
            if len(self.positions) >= 10:  # Hardcoded limit for example
                return False
        
        return True
