"""
Trade metrics collector for the system exporter

This module provides a collector for trade metrics, tracking active and historical trades.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional


class TradeMetricsCollector:
    """Collector for trade metrics"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trade metrics collector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.collection_thread = None
        self.collection_interval = config.get("trade_metrics", {}).get(
            "collection_interval", 60
        )  # seconds

        # Initialize metrics storage
        self.metrics = {}
        self.metrics_lock = threading.Lock()

        # Initialize trade storage
        self.active_trades = []
        self.historical_trades = []
        self.max_historical_trades = config.get("trade_metrics", {}).get(
            "max_historical_trades", 1000
        )
        self.trades_lock = threading.Lock()

        # Initialize trade statistics
        self.trade_stats = {}
        self.trade_stats_lock = threading.Lock()

        # Initialize daily profit/loss tracking
        self.daily_pnl = {}
        self.daily_pnl_lock = threading.Lock()

    def start(self):
        """Start the collector"""
        if self.running:
            logging.warning("Trade metrics collector already running")
            return

        self.running = True

        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True
        )
        self.collection_thread.start()

        logging.info("Trade metrics collector started")

    def stop(self):
        """Stop the collector"""
        if not self.running:
            logging.warning("Trade metrics collector not running")
            return

        self.running = False

        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)

        logging.info("Trade metrics collector stopped")

    def _collection_loop(self):
        """Collection loop for periodic collection"""
        while self.running:
            try:
                # Update trade metrics
                self._update_trade_metrics()

                # Calculate trade statistics
                self._calculate_trade_statistics()

                # Calculate daily profit/loss
                self._calculate_daily_pnl()
            except Exception as e:
                logging.error(f"Error collecting trade metrics: {str(e)}")

            # Sleep until next collection
            time.sleep(self.collection_interval)

    def _update_trade_metrics(self):
        """Update trade metrics"""
        # This method would typically update trade metrics from various sources
        # For now, it's a placeholder that would be implemented based on specific requirements
        pass

    def _calculate_trade_statistics(self):
        """Calculate trade statistics"""
        with self.trades_lock:
            # Skip if no trades
            if not self.historical_trades:
                return
            
            # Calculate statistics
            total_trades = len(self.historical_trades)
            winning_trades = [
                t for t in self.historical_trades if t.get("profit", 0) > 0
            ]
            losing_trades = [
                t for t in self.historical_trades if t.get("profit", 0) <= 0
            ]

            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = win_count / total_trades if total_trades > 0 else 0

            total_profit = sum(t.get("profit", 0) for t in winning_trades)
            total_loss = sum(abs(t.get("profit", 0)) for t in losing_trades)
            net_profit = total_profit - total_loss

            avg_profit = total_profit / win_count if win_count > 0 else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            profit_factor = (
                total_profit / total_loss if total_loss > 0 else (0 if total_profit == 0 else float("inf"))
            )

            # Calculate trade durations
            durations = []
            for trade in self.historical_trades:
                entry_time = trade.get("entry_time")
                exit_time = trade.get("exit_time")

                if entry_time and exit_time:
                    try:
                        entry_dt = datetime.fromisoformat(
                            entry_time.replace("Z", "+00:00")
                        )
                        exit_dt = datetime.fromisoformat(
                            exit_time.replace("Z", "+00:00")
                        )
                        duration = (exit_dt - entry_dt).total_seconds()
                        if duration is not None:
                            durations.append(duration)
                    except (ValueError, TypeError):
                        pass

            avg_duration = sum(durations) / len(durations) if durations else 0

            # Calculate drawdown
            equity_curve = []
            running_equity = 0
            max_drawdown = 0
            peak = 0

            for trade in sorted(
                self.historical_trades, key=lambda x: x.get("exit_time", "")
            ):
                profit = trade.get("profit", 0)
                if profit is None:
                    profit = 0
                    
                running_equity += profit
                equity_curve.append(running_equity)

                if running_equity > peak:
                    peak = running_equity
                elif peak > 0:  # Only calculate drawdown if peak is positive
                    drawdown = peak - running_equity
                    max_drawdown = max(max_drawdown, drawdown)

            # Calculate by symbol
            trades_by_symbol = {}
            for trade in self.historical_trades:
                symbol = trade.get("symbol", "UNKNOWN")
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(trade)

            symbol_stats = {}
            for symbol, trades in trades_by_symbol.items():
                win_trades = [t for t in trades if t.get("profit", 0) > 0]
                loss_trades = [t for t in trades if t.get("profit", 0) <= 0]

                symbol_win_count = len(win_trades)
                symbol_loss_count = len(loss_trades)
                symbol_win_rate = symbol_win_count / len(trades) if trades else 0

                symbol_total_profit = sum(t.get("profit", 0) for t in win_trades)
                symbol_total_loss = sum(abs(t.get("profit", 0)) for t in loss_trades)
                symbol_net_profit = symbol_total_profit - symbol_total_loss

                symbol_stats[symbol] = {
                    "trade_count": len(trades),
                    "win_count": symbol_win_count,
                    "loss_count": symbol_loss_count,
                    "win_rate": symbol_win_rate,
                    "total_profit": symbol_total_profit,
                    "total_loss": symbol_total_loss,
                    "net_profit": symbol_net_profit,
                }

            # Update trade statistics
            with self.trade_stats_lock:
                self.trade_stats = {
                    "timestamp": int(time.time()),
                    "total_trades": total_trades,
                    "win_count": win_count,
                    "loss_count": loss_count,
                    "win_rate": win_rate,
                    "total_profit": total_profit,
                    "total_loss": total_loss,
                    "net_profit": net_profit,
                    "avg_profit": avg_profit,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor,
                    "avg_duration": avg_duration,
                    "max_drawdown": max_drawdown,
                    "by_symbol": symbol_stats,
                }

    def add_trade(self, trade: Dict[str, Any]):
        """
        Add a new trade

        Args:
            trade: Trade data
        """
        with self.trades_lock:
            # Ensure trade has is_active boolean field
            is_active = trade.get("status") == "active"
            trade["is_active"] = is_active

            # Check if trade is active or historical
            if is_active:
                # Add to active trades
                self.active_trades.append(trade)
            else:
                # Add to historical trades
                self.historical_trades.append(trade)

                # Trim historical trades if exceeding maximum
                if len(self.historical_trades) > self.max_historical_trades:
                    self.historical_trades = self.historical_trades[
                        -self.max_historical_trades :
                    ]

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing trade

        Args:
            trade_id: Trade ID
            updates: Updates to apply

        Returns:
            True if trade was found and updated, False otherwise
        """
        with self.trades_lock:
            # Update is_active boolean based on status if provided
            if "status" in updates:
                updates["is_active"] = updates.get("status") == "active"

            # Check active trades
            for i, trade in enumerate(self.active_trades):
                if trade.get("id") == trade_id:
                    # Update trade
                    self.active_trades[i].update(updates)

                    # Check if trade is now closed
                    if not self.active_trades[i].get("is_active", False):
                        # Move to historical trades
                        self.historical_trades.append(self.active_trades[i])
                        del self.active_trades[i]

                        # Trim historical trades if exceeding maximum
                        if len(self.historical_trades) > self.max_historical_trades:
                            self.historical_trades = self.historical_trades[
                                -self.max_historical_trades :
                            ]

                    return True

            # Check historical trades
            for i, trade in enumerate(self.historical_trades):
                if trade.get("id") == trade_id:
                    # Update trade
                    self.historical_trades[i].update(updates)
                    return True

            return False

    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get active trades

        Returns:
            List of active trades
        """
        with self.trades_lock:
            return self.active_trades.copy()

    def get_historical_trades(
        self,
        limit: int = 100,
        offset: int = 0,
        symbol: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get historical trades with filtering and pagination

        Args:
            limit: Maximum number of trades to return
            offset: Offset for pagination
            symbol: Filter by symbol
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)

        Returns:
            List of historical trades
        """
        with self.trades_lock:
            # Apply filters
            filtered_trades = self.historical_trades

            if symbol:
                filtered_trades = [
                    t for t in filtered_trades if t.get("symbol") == symbol
                ]

            if start_time:
                filtered_trades = [
                    t for t in filtered_trades if t.get("exit_time", "") >= start_time
                ]

            if end_time:
                filtered_trades = [
                    t for t in filtered_trades if t.get("exit_time", "") <= end_time
                ]

            # Sort by exit time (newest first)
            sorted_trades = sorted(
                filtered_trades, key=lambda x: x.get("exit_time", ""), reverse=True
            )

            # Apply pagination
            paginated_trades = sorted_trades[offset : offset + limit]

            return paginated_trades

    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Get trade statistics

        Returns:
            Dictionary with trade statistics
        """
        with self.trade_stats_lock:
            return self.trade_stats.copy()

    def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trade by ID

        Args:
            trade_id: Trade ID

        Returns:
            Trade data or None if not found
        """
        with self.trades_lock:
            # Check active trades
            for trade in self.active_trades:
                if trade.get("id") == trade_id:
                    return trade.copy()

            # Check historical trades
            for trade in self.historical_trades:
                if trade.get("id") == trade_id:
                    return trade.copy()

            return None

    def _calculate_daily_pnl(self):
        """Calculate daily profit/loss"""
        with self.trades_lock:
            # Skip if no trades
            if not self.historical_trades:
                return

            # Group trades by exit date
            daily_trades = {}

            for trade in self.historical_trades:
                exit_time = trade.get("exit_time")

                if not exit_time:
                    continue

                try:
                    # Parse exit time
                    exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))

                    # Get date string (YYYY-MM-DD)
                    date_str = exit_dt.strftime("%Y-%m-%d")

                    # Add trade to daily trades
                    if date_str not in daily_trades:
                        daily_trades[date_str] = []

                    daily_trades[date_str].append(trade)
                except (ValueError, TypeError):
                    pass

            # Calculate daily profit/loss
            daily_pnl = {}

            for date_str, trades in daily_trades.items():
                # Calculate profit/loss
                total_profit = sum(
                    t.get("profit", 0) for t in trades if t.get("profit", 0) > 0
                )
                total_loss = sum(
                    abs(t.get("profit", 0)) for t in trades if t.get("profit", 0) <= 0
                )
                net_profit = sum(t.get("profit", 0) for t in trades)

                # Count trades
                win_count = sum(1 for t in trades if t.get("profit", 0) > 0)
                loss_count = sum(1 for t in trades if t.get("profit", 0) <= 0)

                # Calculate win rate
                win_rate = win_count / len(trades) if trades else 0

                # Store daily profit/loss
                daily_pnl[date_str] = {
                    "date": date_str,
                    "trade_count": len(trades),
                    "win_count": win_count,
                    "loss_count": loss_count,
                    "win_rate": win_rate,
                    "total_profit": total_profit,
                    "total_loss": total_loss,
                    "net_profit": net_profit,
                    "trades": [
                        {
                            "id": t.get("id"),
                            "symbol": t.get("symbol"),
                            "entry_time": t.get("entry_time"),
                            "exit_time": t.get("exit_time"),
                            "entry_price": t.get("entry_price"),
                            "exit_price": t.get("exit_price"),
                            "quantity": t.get("quantity"),
                            "profit": t.get("profit", 0),
                            "side": t.get("side"),
                            "is_active": t.get("is_active", False),
                        }
                        for t in trades
                    ],
                }

            # Update daily profit/loss
            with self.daily_pnl_lock:
                self.daily_pnl = daily_pnl

    def get_daily_pnl(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get daily profit/loss

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with daily profit/loss
        """
        with self.daily_pnl_lock:
            # Copy daily profit/loss
            daily_pnl = self.daily_pnl.copy()

            # Filter by date range
            if start_date or end_date:
                filtered_pnl = {}

                for date_str, pnl in daily_pnl.items():
                    if start_date and date_str < start_date:
                        continue

                    if end_date and date_str > end_date:
                        continue

                    filtered_pnl[date_str] = pnl

                return filtered_pnl

            return daily_pnl

    def get_pnl_calendar(
        self, year: Optional[int] = None, month: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get profit/loss calendar

        Args:
            year: Year (e.g., 2025)
            month: Month (1-12)

        Returns:
            Dictionary with profit/loss calendar
        """
        # Get current year and month if not provided
        now = datetime.now()
        year = year or now.year

        # Get daily profit/loss
        with self.daily_pnl_lock:
            daily_pnl = self.daily_pnl.copy()

        # Filter by year and month
        filtered_pnl = {}

        for date_str, pnl in daily_pnl.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")

                if date.year != year:
                    continue

                if month and date.month != month:
                    continue

                filtered_pnl[date_str] = pnl
            except ValueError:
                continue

        # Create calendar
        if month:
            # Monthly calendar
            calendar = {"year": year, "month": month, "days": {}}

            # Get number of days in month
            if month == 2:
                # February
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    # Leap year
                    days_in_month = 29
                else:
                    days_in_month = 28
            elif month in [4, 6, 9, 11]:
                # April, June, September, November
                days_in_month = 30
            else:
                # January, March, May, July, August, October, December
                days_in_month = 31

            # Add days to calendar
            for day in range(1, days_in_month + 1):
                date_str = f"{year}-{month:02d}-{day:02d}"

                if date_str in filtered_pnl:
                    calendar["days"][day] = {
                        "net_profit": filtered_pnl[date_str]["net_profit"],
                        "trade_count": filtered_pnl[date_str]["trade_count"],
                        "win_rate": filtered_pnl[date_str]["win_rate"],
                    }
                else:
                    calendar["days"][day] = {
                        "net_profit": 0,
                        "trade_count": 0,
                        "win_rate": 0,
                    }

            return calendar
        else:
            # Yearly calendar
            calendar = {"year": year, "months": {}}

            # Initialize months
            for month in range(1, 13):
                calendar["months"][month] = {
                    "net_profit": 0,
                    "trade_count": 0,
                    "win_count": 0,
                    "loss_count": 0,
                }

            # Add data to calendar
            for date_str, pnl in filtered_pnl.items():
                date = datetime.strptime(date_str, "%Y-%m-%d")
                month = date.month

                calendar["months"][month]["net_profit"] += pnl["net_profit"]
                calendar["months"][month]["trade_count"] += pnl["trade_count"]
                calendar["months"][month]["win_count"] += pnl["win_count"]
                calendar["months"][month]["loss_count"] += pnl["loss_count"]

            # Calculate win rate for each month
            for month in range(1, 13):
                total_trades = (
                    calendar["months"][month]["win_count"]
                    + calendar["months"][month]["loss_count"]
                )
                if total_trades > 0:
                    calendar["months"][month]["win_rate"] = (
                        calendar["months"][month]["win_count"] / total_trades
                    )
                else:
                    calendar["months"][month]["win_rate"] = 0

            return calendar
