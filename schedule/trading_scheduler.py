#!/usr/bin/env python3
# trading_scheduler.py

import subprocess
import time
import datetime
import pytz
import logging
import signal
import sys
import os
import json
import pandas_market_calendars as mcal  # Import pandas_market_calendars

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "trading_scheduler.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("trading_scheduler")

# Configuration
TRADING_SYSTEM_PATH = "/home/ubuntu/inavvi2"
TRADING_SYSTEM_CMD = "./trading_system --config config/system_simple.yaml --trading-config config/trading_simple.yaml"
MARKET_OPEN_TIME = datetime.time(9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = datetime.time(16, 0)  # 4:00 PM ET
STARTUP_BUFFER_HOURS = 1  # Start 1 hour before market open
SHUTDOWN_BUFFER_HOURS = 1  # Stop 1 hour after market close
TIMEZONE = pytz.timezone("US/Eastern")

# Trading process handle
trading_process = None


def is_market_day():
    """Check if today is a trading day (weekday and not a holiday)"""
    now = datetime.datetime.now(TIMEZONE)

    # Check if it's a weekend
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False

    # Add holiday calendar check
    # For a production system, integrate with a proper market calendar API
    # or use a library like pandas_market_calendars

    # Check if today is a market holiday for NYSE
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=now.date(), end_date=now.date())

    if schedule.empty:
        logger.info(f"Today {now.date()} is a market holiday or not a trading day.")
        return False

    return True


def should_be_running():
    """Determine if the trading system should be running now"""
    if not is_market_day():
        return False

    now = datetime.datetime.now(TIMEZONE)
    current_time = now.time()

    # Calculate start and end times with buffers
    start_time = (
        datetime.datetime.combine(now.date(), MARKET_OPEN_TIME)
        - datetime.timedelta(hours=STARTUP_BUFFER_HOURS)
    ).time()

    end_time = (
        datetime.datetime.combine(now.date(), MARKET_CLOSE_TIME)
        + datetime.timedelta(hours=SHUTDOWN_BUFFER_HOURS)
    ).time()

    return start_time <= current_time <= end_time


def start_trading_system():
    """Start the trading system as a subprocess"""
    global trading_process

    logger.info("Starting trading system...")
    os.chdir(TRADING_SYSTEM_PATH)

    trading_process = subprocess.Popen(
        TRADING_SYSTEM_CMD,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    logger.info(f"Trading system started with PID {trading_process.pid}")
    return trading_process


def stop_trading_system():
    """Gracefully stop the trading system"""
    global trading_process

    if trading_process is None:
        return

    logger.info(f"Stopping trading system (PID {trading_process.pid})...")

    # Send SIGTERM for graceful shutdown
    trading_process.send_signal(signal.SIGTERM)

    # Wait up to 30 seconds for graceful shutdown
    try:
        trading_process.wait(timeout=30)
        logger.info("Trading system stopped gracefully")
    except subprocess.TimeoutExpired:
        logger.warning("Trading system did not stop gracefully, forcing termination")
        trading_process.kill()

    trading_process = None


def place_order(
    symbol,
    order_type,
    quantity,
    side="BUY",
    price=None,
    time_in_force="DAY",
    use_bracket_order=False,
    take_profit_pct=None,
    stop_loss_pct=None,
):
    """
    Production implementation for order placement functionality.

    Args:
        symbol (str): The trading symbol (e.g., 'AAPL')
        order_type (str): Type of order ('MARKET', 'LIMIT', etc.)
        quantity (int): Number of shares to trade
        side (str): Order side ('BUY' or 'SELL')
        price (float, optional): Price for limit orders
        time_in_force (str): Time in force ('DAY', 'GTC', 'IOC', 'FOK')
        use_bracket_order (bool): Whether to use a bracket order
        take_profit_pct (float): Take profit percentage (for bracket orders)
        stop_loss_pct (float): Stop loss percentage (for bracket orders)

    Returns:
        dict: Order response containing order ID and status
    """
    logger.info(f"Placing {order_type} {side} order for {quantity} shares of {symbol}")

    try:
        # Prepare order data
        order_data = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        # Add price for limit orders
        if order_type.upper() == "LIMIT" and price is not None:
            order_data["limit_price"] = str(price)

        # Add bracket order details if requested
        if use_bracket_order:
            if take_profit_pct is not None:
                # Calculate take profit price
                take_profit_price = (
                    price * (1 + take_profit_pct / 100)
                    if side.upper() == "BUY"
                    else price * (1 - take_profit_pct / 100)
                )
                order_data["take_profit"] = {
                    "limit_price": str(round(take_profit_price, 2))
                }

            if stop_loss_pct is not None:
                # Calculate stop loss price
                stop_loss_price = (
                    price * (1 - stop_loss_pct / 100)
                    if side.upper() == "BUY"
                    else price * (1 + stop_loss_pct / 100)
                )
                order_data["stop_loss"] = {"stop_price": str(round(stop_loss_price, 2))}

            # Set order class to bracket
            order_data["order_class"] = "bracket"

        # Connect to trading system API

        # Use subprocess to call the trading system's CLI tool
        cmd = [
            f"{TRADING_SYSTEM_PATH}/trading_cli",
            "--action",
            "place_order",
            "--symbol",
            symbol,
            "--quantity",
            str(quantity),
            "--side",
            side,
            "--type",
            order_type,
        ]

        # Add optional parameters
        if price is not None and order_type.upper() == "LIMIT":
            cmd.extend(["--price", str(price)])

        if use_bracket_order:
            cmd.append("--bracket")
            if take_profit_pct is not None:
                cmd.extend(["--take-profit-pct", str(take_profit_pct)])
            if stop_loss_pct is not None:
                cmd.extend(["--stop-loss-pct", str(stop_loss_pct)])

        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse response
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                logger.info(f"Order placed successfully: {response['id']}")
                return response
            except json.JSONDecodeError:
                logger.error(f"Failed to parse order response: {result.stdout}")
                raise Exception("Invalid response format")
        else:
            logger.error(f"Order placement failed: {result.stderr}")
            raise Exception(f"Order placement failed: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Order placement failed: {e.stderr}")
        raise Exception(f"Order placement failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Order placement error: {str(e)}")
        raise


def handle_signal(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    stop_trading_system()
    sys.exit(0)


def main():
    """Main scheduler loop"""
    global trading_process

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Trading system scheduler started")

    while True:
        try:
            if should_be_running() and trading_process is None:
                start_trading_system()
            elif not should_be_running() and trading_process is not None:
                stop_trading_system()

            # Check if process is still running
            if trading_process is not None and trading_process.poll() is not None:
                logger.error(
                    f"Trading system exited unexpectedly with code {trading_process.returncode}"
                )
                # Read output for error information
                output = trading_process.stdout.read()
                logger.error(f"Trading system output: {output}")
                trading_process = None

                # Wait before restarting
                time.sleep(60)

                if should_be_running():
                    logger.info("Attempting to restart trading system...")
                    start_trading_system()

            # Sleep for a minute before checking again
            time.sleep(60)

        except Exception as e:
            logger.exception(f"Error in scheduler loop: {e}")
            time.sleep(60)  # Wait before retrying


def get_order_status(order_id):
    """
    Get the status of an order.

    Args:
        order_id (str): The order ID to check

    Returns:
        dict: Order status information
    """
    logger.info(f"Checking status for order {order_id}")

    try:
        # Use subprocess to call the trading system's CLI tool
        cmd = [
            f"{TRADING_SYSTEM_PATH}/trading_cli",
            "--action",
            "get_order",
            "--order-id",
            order_id,
        ]

        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse response
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                logger.info(f"Order status: {response['status']}")
                return response
            except json.JSONDecodeError:
                logger.error(f"Failed to parse order status response: {result.stdout}")
                raise Exception("Invalid response format")
        else:
            logger.error(f"Order status check failed: {result.stderr}")
            raise Exception(f"Order status check failed: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Order status check failed: {e.stderr}")
        raise Exception(f"Order status check failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Order status check error: {str(e)}")
        raise


def cancel_order(order_id):
    """
    Cancel an existing order.

    Args:
        order_id (str): The order ID to cancel

    Returns:
        bool: True if cancellation was successful
    """
    logger.info(f"Cancelling order {order_id}")

    try:
        # Use subprocess to call the trading system's CLI tool
        cmd = [
            f"{TRADING_SYSTEM_PATH}/trading_cli",
            "--action",
            "cancel_order",
            "--order-id",
            order_id,
        ]

        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Check result
        if result.returncode == 0:
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        else:
            logger.error(f"Order cancellation failed: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Order cancellation failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Order cancellation error: {str(e)}")
        return False


def get_account_info():
    """
    Get account information including equity, buying power, etc.

    Returns:
        dict: Account information
    """
    logger.info("Getting account information")

    try:
        # Use subprocess to call the trading system's CLI tool
        cmd = [f"{TRADING_SYSTEM_PATH}/trading_cli", "--action", "get_account"]

        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse response
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                logger.info(f"Account equity: ${response.get('equity', 'N/A')}")
                return response
            except json.JSONDecodeError:
                logger.error(f"Failed to parse account info response: {result.stdout}")
                raise Exception("Invalid response format")
        else:
            logger.error(f"Account info request failed: {result.stderr}")
            raise Exception(f"Account info request failed: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Account info request failed: {e.stderr}")
        raise Exception(f"Account info request failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Account info request error: {str(e)}")
        raise


def get_positions():
    """
    Get current positions.

    Returns:
        list: List of current positions
    """
    logger.info("Getting current positions")

    try:
        # Use subprocess to call the trading system's CLI tool
        cmd = [f"{TRADING_SYSTEM_PATH}/trading_cli", "--action", "get_positions"]

        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse response
        if result.returncode == 0:
            try:
                positions = json.loads(result.stdout)
                logger.info(f"Current positions: {len(positions)}")
                return positions
            except json.JSONDecodeError:
                logger.error(f"Failed to parse positions response: {result.stdout}")
                raise Exception("Invalid response format")
        else:
            logger.error(f"Positions request failed: {result.stderr}")
            raise Exception(f"Positions request failed: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Positions request failed: {e.stderr}")
        raise Exception(f"Positions request failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Positions request error: {str(e)}")
        raise


def calculate_position_size(symbol, risk_pct=1.0, stop_loss_pct=2.0):
    """
    Calculate position size based on risk management rules.

    Args:
        symbol (str): The trading symbol
        risk_pct (float): Percentage of account to risk on this trade
        stop_loss_pct (float): Stop loss percentage

    Returns:
        float: Number of shares to trade
    """
    try:
        # Get account information
        account = get_account_info()
        equity = float(account.get("equity", 0))

        # Calculate dollar amount to risk
        risk_amount = equity * (risk_pct / 100)

        # Get current price
        cmd = [
            f"{TRADING_SYSTEM_PATH}/trading_cli",
            "--action",
            "get_quote",
            "--symbol",
            symbol,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        quote = json.loads(result.stdout)
        current_price = float(quote.get("last_price", 0))

        if current_price == 0:
            logger.error(f"Invalid price for {symbol}")
            return 0

        # Calculate stop loss price
        stop_loss_price = current_price * (1 - stop_loss_pct / 100)

        # Calculate position size
        price_risk = current_price - stop_loss_price
        if price_risk <= 0:
            logger.error(f"Invalid stop loss for {symbol}")
            return 0

        shares = risk_amount / price_risk

        # Round down to whole shares
        shares = int(shares)

        logger.info(f"Calculated position size for {symbol}: {shares} shares")
        return shares

    except Exception as e:
        logger.error(f"Position size calculation error: {str(e)}")
        return 0


if __name__ == "__main__":
    main()
