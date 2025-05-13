"""
Alpaca portfolio collector for the system exporter

This module provides a collector for Alpaca portfolio metrics, P&L, and trading statistics.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional

# Import Alpaca API client
from src.execution.alpaca_rest_api import AlpacaRestAPI

class AlpacaPortfolioCollector:
    """Collector for Alpaca portfolio metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Alpaca portfolio collector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.collection_thread = None
        self.collection_interval = config.get("alpaca_portfolio", {}).get("collection_interval", 60)  # seconds
        
        # Initialize metrics storage
        self.metrics = {}
        self.metrics_lock = threading.Lock()
        
        # Initialize Alpaca API client
        self.alpaca_api = AlpacaRestAPI(config)
        
        # Initialize trade history
        self.trade_history = []
        self.max_trade_history = config.get("alpaca_portfolio", {}).get("max_trade_history", 1000)
        self.trade_history_lock = threading.Lock()
        
        # Initialize performance metrics
        self.performance_metrics = {}
        self.performance_metrics_lock = threading.Lock()
    
    def start(self):
        """Start the collector"""
        if self.running:
            logging.warning("Alpaca portfolio collector already running")
            return
        
        # Initialize Alpaca API client
        if not self.alpaca_api.initialize():
            logging.error("Failed to initialize Alpaca API client")
            return
        
        self.running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logging.info("Alpaca portfolio collector started")
    
    def stop(self):
        """Stop the collector"""
        if not self.running:
            logging.warning("Alpaca portfolio collector not running")
            return
        
        self.running = False
        
        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        # Clean up Alpaca API client
        self.alpaca_api.cleanup()
        
        logging.info("Alpaca portfolio collector stopped")
    
    def _collection_loop(self):
        """Collection loop for periodic collection"""
        while self.running:
            try:
                # Collect portfolio metrics
                self._collect_portfolio_metrics()
                
                # Collect trade history
                self._collect_trade_history()
                
                # Calculate performance metrics
                self._calculate_performance_metrics()
            except Exception as e:
                logging.error(f"Error collecting Alpaca portfolio metrics: {str(e)}")
            
            # Sleep until next collection
            time.sleep(self.collection_interval)
    
    def _collect_portfolio_metrics(self):
        """Collect portfolio metrics from Alpaca API"""
        try:
            # Get account information
            account = self.alpaca_api.get_account()
            
            # Get positions
            positions = self.alpaca_api.get_positions()
            
            # Get open orders
            orders = self.alpaca_api.get_orders(status="open")
            
            # Extract metrics
            metrics = {
                "timestamp": int(time.time()),
                "account": {
                    "id": account.get("id", ""),
                    "status": account.get("status", ""),
                    "currency": account.get("currency", "USD"),
                    "buying_power": float(account.get("buying_power", "0")),
                    "cash": float(account.get("cash", "0")),
                    "portfolio_value": float(account.get("portfolio_value", "0")),
                    "equity": float(account.get("equity", "0")),
                    "long_market_value": float(account.get("long_market_value", "0")),
                    "short_market_value": float(account.get("short_market_value", "0")),
                    "initial_margin": float(account.get("initial_margin", "0")),
                    "maintenance_margin": float(account.get("maintenance_margin", "0")),
                    "last_equity": float(account.get("last_equity", "0")),
                    "last_maintenance_margin": float(account.get("last_maintenance_margin", "0")),
                    "multiplier": float(account.get("multiplier", "1")),
                    "day_trade_count": int(account.get("daytrade_count", "0")),
                    "sma": float(account.get("sma", "0"))
                },
                "positions": self._process_positions(positions),
                "orders": self._process_orders(orders)
            }
            
            # Update metrics
            with self.metrics_lock:
                self.metrics = metrics
        
        except Exception as e:
            logging.error(f"Error collecting portfolio metrics: {str(e)}")
    
    def _process_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process positions data
        
        Args:
            positions: List of positions from Alpaca API
            
        Returns:
            Processed positions data
        """
        processed_positions = {}
        
        # Group positions by symbol
        for position in positions:
            symbol = position.get("symbol", "")
            if not symbol:
                continue
            
            # Extract position data
            processed_positions[symbol] = {
                "symbol": symbol,
                "qty": float(position.get("qty", "0")),
                "side": position.get("side", ""),
                "avg_entry_price": float(position.get("avg_entry_price", "0")),
                "market_value": float(position.get("market_value", "0")),
                "cost_basis": float(position.get("cost_basis", "0")),
                "unrealized_pl": float(position.get("unrealized_pl", "0")),
                "unrealized_plpc": float(position.get("unrealized_plpc", "0")),
                "current_price": float(position.get("current_price", "0")),
                "lastday_price": float(position.get("lastday_price", "0")),
                "change_today": float(position.get("change_today", "0"))
            }
        
        # Calculate summary statistics
        total_market_value = sum(p.get("market_value", 0) for p in processed_positions.values())
        total_cost_basis = sum(p.get("cost_basis", 0) for p in processed_positions.values())
        total_unrealized_pl = sum(p.get("unrealized_pl", 0) for p in processed_positions.values())
        
        # Add summary
        summary = {
            "count": len(processed_positions),
            "total_market_value": total_market_value,
            "total_cost_basis": total_cost_basis,
            "total_unrealized_pl": total_unrealized_pl,
            "total_unrealized_plpc": (total_unrealized_pl / total_cost_basis) if total_cost_basis > 0 else 0
        }
        
        return {
            "positions": processed_positions,
            "summary": summary
        }
    
    def _process_orders(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process orders data
        
        Args:
            orders: List of orders from Alpaca API
            
        Returns:
            Processed orders data
        """
        processed_orders = {}
        
        # Group orders by symbol
        for order in orders:
            order_id = order.get("id", "")
            if not order_id:
                continue
            
            # Extract order data
            processed_orders[order_id] = {
                "id": order_id,
                "client_order_id": order.get("client_order_id", ""),
                "symbol": order.get("symbol", ""),
                "side": order.get("side", ""),
                "type": order.get("type", ""),
                "time_in_force": order.get("time_in_force", ""),
                "qty": float(order.get("qty", "0")),
                "filled_qty": float(order.get("filled_qty", "0")),
                "filled_avg_price": float(order.get("filled_avg_price", "0") or "0"),
                "status": order.get("status", ""),
                "created_at": order.get("created_at", ""),
                "updated_at": order.get("updated_at", ""),
                "submitted_at": order.get("submitted_at", ""),
                "filled_at": order.get("filled_at", ""),
                "expired_at": order.get("expired_at", ""),
                "canceled_at": order.get("canceled_at", ""),
                "failed_at": order.get("failed_at", ""),
                "asset_class": order.get("asset_class", ""),
                "order_class": order.get("order_class", "")
            }
        
        # Calculate summary statistics
        buy_orders = [o for o in processed_orders.values() if o.get("side") == "buy"]
        sell_orders = [o for o in processed_orders.values() if o.get("side") == "sell"]
        
        # Add summary
        summary = {
            "count": len(processed_orders),
            "buy_count": len(buy_orders),
            "sell_count": len(sell_orders)
        }
        
        return {
            "orders": processed_orders,
            "summary": summary
        }
    
    def _collect_trade_history(self):
        """Collect trade history from Alpaca API"""
        try:
            # Get closed orders
            closed_orders = self.alpaca_api.get_orders(status="closed")
            
            # Process closed orders
            for order in closed_orders:
                order_id = order.get("id", "")
                if not order_id:
                    continue
                
                # Check if order is already in trade history
                with self.trade_history_lock:
                    if any(trade.get("id") == order_id for trade in self.trade_history):
                        continue
                
                # Extract trade data
                trade = {
                    "id": order_id,
                    "client_order_id": order.get("client_order_id", ""),
                    "symbol": order.get("symbol", ""),
                    "side": order.get("side", ""),
                    "type": order.get("type", ""),
                    "time_in_force": order.get("time_in_force", ""),
                    "qty": float(order.get("qty", "0")),
                    "filled_qty": float(order.get("filled_qty", "0")),
                    "filled_avg_price": float(order.get("filled_avg_price", "0") or "0"),
                    "status": order.get("status", ""),
                    "created_at": order.get("created_at", ""),
                    "updated_at": order.get("updated_at", ""),
                    "submitted_at": order.get("submitted_at", ""),
                    "filled_at": order.get("filled_at", ""),
                    "expired_at": order.get("expired_at", ""),
                    "canceled_at": order.get("canceled_at", ""),
                    "failed_at": order.get("failed_at", ""),
                    "asset_class": order.get("asset_class", ""),
                    "order_class": order.get("order_class", "")
                }
                
                # Calculate trade value
                trade["value"] = trade["filled_qty"] * trade["filled_avg_price"]
                
                # Add trade to history
                with self.trade_history_lock:
                    self.trade_history.append(trade)
                    
                    # Trim trade history if exceeding maximum
                    if len(self.trade_history) > self.max_trade_history:
                        self.trade_history = self.trade_history[-self.max_trade_history:]
        
        except Exception as e:
            logging.error(f"Error collecting trade history: {str(e)}")
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from trade history"""
        try:
            with self.trade_history_lock:
                # Skip if no trades
                if not self.trade_history:
                    return
                
                # Sort trades by filled_at
                sorted_trades = sorted(self.trade_history, key=lambda x: x.get("filled_at", ""))
                
                # Calculate win/loss ratio
                wins = 0
                losses = 0
                total_profit = 0.0
                total_loss = 0.0
                
                # Group trades by symbol
                trades_by_symbol = {}
                for trade in sorted_trades:
                    symbol = trade.get("symbol", "")
                    if not symbol:
                        continue
                    
                    if symbol not in trades_by_symbol:
                        trades_by_symbol[symbol] = []
                    
                    trades_by_symbol[symbol].append(trade)
                
                # Calculate P&L for each symbol
                for symbol, trades in trades_by_symbol.items():
                    # Sort trades by filled_at
                    sorted_symbol_trades = sorted(trades, key=lambda x: x.get("filled_at", ""))
                    
                    # Track position
                    position = 0.0
                    cost_basis = 0.0
                    
                    for trade in sorted_symbol_trades:
                        side = trade.get("side", "")
                        qty = trade.get("filled_qty", 0.0)
                        price = trade.get("filled_avg_price", 0.0)
                        
                        if side == "buy":
                            # Add to position
                            new_position = position + qty
                            new_cost_basis = (position * cost_basis + qty * price) / new_position if new_position > 0 else 0.0
                            position = new_position
                            cost_basis = new_cost_basis
                        elif side == "sell":
                            # Calculate P&L
                            if position > 0:
                                pnl = qty * (price - cost_basis)
                                
                                if pnl > 0:
                                    wins += 1
                                    total_profit += pnl
                                else:
                                    losses += 1
                                    total_loss += abs(pnl)
                            
                            # Reduce position
                            position -= qty
                            # Cost basis remains the same
                
                # Calculate metrics
                win_loss_ratio = wins / losses if losses > 0 else float('inf')
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                # Calculate daily P&L
                daily_pnl = {}
                for trade in sorted_trades:
                    filled_at = trade.get("filled_at", "")
                    if not filled_at:
                        continue
                    
                    # Extract date
                    date = filled_at.split("T")[0]
                    
                    # Calculate P&L
                    side = trade.get("side", "")
                    qty = trade.get("filled_qty", 0.0)
                    price = trade.get("filled_avg_price", 0.0)
                    value = qty * price
                    
                    if date not in daily_pnl:
                        daily_pnl[date] = {
                            "buy_value": 0.0,
                            "sell_value": 0.0,
                            "net_value": 0.0
                        }
                    
                    if side == "buy":
                        daily_pnl[date]["buy_value"] += value
                    elif side == "sell":
                        daily_pnl[date]["sell_value"] += value
                    
                    daily_pnl[date]["net_value"] = daily_pnl[date]["sell_value"] - daily_pnl[date]["buy_value"]
                
                # Calculate drawdown
                equity_curve = []
                max_drawdown = 0.0
                peak = 0.0
                
                for date, pnl in sorted(daily_pnl.items()):
                    net_value = pnl["net_value"]
                    equity_curve.append((date, net_value))
                    
                    if net_value > peak:
                        peak = net_value
                    else:
                        drawdown = (peak - net_value) / peak if peak > 0 else 0.0
                        max_drawdown = max(max_drawdown, drawdown)
                
                # Update performance metrics
                with self.performance_metrics_lock:
                    self.performance_metrics = {
                        "timestamp": int(time.time()),
                        "win_count": wins,
                        "loss_count": losses,
                        "win_loss_ratio": win_loss_ratio,
                        "total_profit": total_profit,
                        "total_loss": total_loss,
                        "profit_factor": profit_factor,
                        "max_drawdown": max_drawdown,
                        "daily_pnl": daily_pnl
                    }
        
        except Exception as e:
            logging.error(f"Error calculating performance metrics: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the latest metrics
        
        Returns:
            Dictionary with portfolio metrics
        """
        with self.metrics_lock, self.performance_metrics_lock:
            # Combine metrics and performance metrics
            combined_metrics = {
                "portfolio": self.metrics,
                "performance": self.performance_metrics
            }
            
            return combined_metrics
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get active trades
        
        Returns:
            List of active trades
        """
        with self.metrics_lock:
            active_trades = []
            
            # Get positions from metrics
            positions = self.metrics.get("positions", {}).get("positions", {})
            
            for symbol, position in positions.items():
                active_trades.append({
                    "symbol": symbol,
                    "side": position.get("side", ""),
                    "quantity": position.get("qty", 0.0),
                    "entry_price": position.get("avg_entry_price", 0.0),
                    "current_price": position.get("current_price", 0.0),
                    "market_value": position.get("market_value", 0.0),
                    "cost_basis": position.get("cost_basis", 0.0),
                    "unrealized_pl": position.get("unrealized_pl", 0.0),
                    "unrealized_plpc": position.get("unrealized_plpc", 0.0),
                    "change_today": position.get("change_today", 0.0)
                })
            
            return active_trades
    
    def get_trade_history(self, limit: int = 100, offset: int = 0, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get trade history with pagination
        
        Args:
            limit: Maximum number of trades to return
            offset: Offset for pagination
            symbol: Filter by symbol
            
        Returns:
            List of trades
        """
        with self.trade_history_lock:
            # Apply filters
            filtered_trades = self.trade_history
            
            if symbol:
                filtered_trades = [trade for trade in filtered_trades if trade.get("symbol") == symbol]
            
            # Sort by filled_at (newest first)
            sorted_trades = sorted(filtered_trades, key=lambda x: x.get("filled_at", ""), reverse=True)
            
            # Apply pagination
            paginated_trades = sorted_trades[offset:offset + limit]
            
            return paginated_trades
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        with self.performance_metrics_lock:
            return self.performance_metrics.copy()
