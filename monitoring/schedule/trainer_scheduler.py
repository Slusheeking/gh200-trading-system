#!/usr/bin/env python3
# trainer_scheduler.py

import subprocess
import time
import datetime
import pytz
import logging
import signal
import sys
import os

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "trainer_scheduler.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("trainer_scheduler")

# Configuration
SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAINER_CMD = f"{SCRIPT_DIR}/start_trainer.sh"

# Schedule configuration - run trainer on weekends and after market hours
TRAINING_DAYS = [5, 6]  # 5=Saturday, 6=Sunday
WEEKDAY_START_TIME = datetime.time(18, 0)  # 6:00 PM ET
WEEKDAY_END_TIME = datetime.time(8, 0)  # 8:00 AM ET
TIMEZONE = pytz.timezone("US/Eastern")

# Trainer process handle
trainer_process = None


def should_be_running():
    """Determine if the trainer should be running now"""
    now = datetime.datetime.now(TIMEZONE)
    current_time = now.time()
    weekday = now.weekday()
    
    # Run on weekends (Saturday and Sunday)
    if weekday in TRAINING_DAYS:
        return True
    
    # Run on weekdays after market hours (6 PM to 8 AM)
    if WEEKDAY_START_TIME <= current_time or current_time <= WEEKDAY_END_TIME:
        return True
    
    return False


def start_trainer():
    """Start the model trainer as a subprocess"""
    global trainer_process
    
    logger.info("Starting model trainer...")
    
    trainer_process = subprocess.Popen(
        TRAINER_CMD,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    
    logger.info(f"Model trainer started with PID {trainer_process.pid}")
    return trainer_process


def stop_trainer():
    """Gracefully stop the model trainer"""
    global trainer_process
    
    if trainer_process is None:
        return
    
    logger.info(f"Stopping model trainer (PID {trainer_process.pid})...")
    
    # Send SIGTERM for graceful shutdown
    trainer_process.send_signal(signal.SIGTERM)
    
    # Wait up to 30 seconds for graceful shutdown
    try:
        trainer_process.wait(timeout=30)
        logger.info("Model trainer stopped gracefully")
    except subprocess.TimeoutExpired:
        logger.warning("Model trainer did not stop gracefully, forcing termination")
        trainer_process.kill()
    
    trainer_process = None


def handle_signal(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    stop_trainer()
    sys.exit(0)


def main():
    """Main scheduler loop"""
    global trainer_process
    
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    logger.info("Model trainer scheduler started")
    
    while True:
        try:
            if should_be_running() and trainer_process is None:
                start_trainer()
            elif not should_be_running() and trainer_process is not None:
                stop_trainer()
            
            # Check if process is still running
            if trainer_process is not None and trainer_process.poll() is not None:
                logger.error(
                    f"Model trainer exited unexpectedly with code {trainer_process.returncode}"
                )
                # Read output for error information
                output = trainer_process.stdout.read()
                logger.error(f"Model trainer output: {output}")
                trainer_process = None
                
                # Wait before restarting
                time.sleep(60)
                
                if should_be_running():
                    logger.info("Attempting to restart model trainer...")
                    start_trainer()
            
            # Sleep for a minute before checking again
            time.sleep(60)
            
        except Exception as e:
            logger.exception(f"Error in scheduler loop: {e}")
            time.sleep(60)  # Wait before retrying


if __name__ == "__main__":
    main()