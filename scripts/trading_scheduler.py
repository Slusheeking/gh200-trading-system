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
from pathlib import Path

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "trading_scheduler.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_scheduler")

# Configuration
TRADING_SYSTEM_PATH = "/home/ubuntu/inavvi2"
TRADING_SYSTEM_CMD = "./trading_system --config config/system_simple.yaml --trading-config config/trading_simple.yaml"
MARKET_OPEN_TIME = datetime.time(9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = datetime.time(16, 0)  # 4:00 PM ET
STARTUP_BUFFER_HOURS = 1  # Start 1 hour before market open
SHUTDOWN_BUFFER_HOURS = 1  # Stop 1 hour after market close
TIMEZONE = pytz.timezone('US/Eastern')

# Trading process handle
trading_process = None

def is_market_day():
    """Check if today is a trading day (weekday and not a holiday)"""
    now = datetime.datetime.now(TIMEZONE)
    
    # Check if it's a weekend
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # TODO: Add holiday calendar check
    # For a production system, integrate with a proper market calendar API
    # or use a library like pandas_market_calendars
    
    return True

def should_be_running():
    """Determine if the trading system should be running now"""
    if not is_market_day():
        return False
        
    now = datetime.datetime.now(TIMEZONE)
    current_time = now.time()
    
    # Calculate start and end times with buffers
    start_time = (
        datetime.datetime.combine(now.date(), MARKET_OPEN_TIME) - 
        datetime.timedelta(hours=STARTUP_BUFFER_HOURS)
    ).time()
    
    end_time = (
        datetime.datetime.combine(now.date(), MARKET_CLOSE_TIME) + 
        datetime.timedelta(hours=SHUTDOWN_BUFFER_HOURS)
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
        universal_newlines=True
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
                logger.error(f"Trading system exited unexpectedly with code {trading_process.returncode}")
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

if __name__ == "__main__":
    main()