# Trading System Scheduling

This document describes the scheduling system for the GH200 Trading System.

## Overview

The scheduling system consists of several components:

1. **Trading System Scripts**:
   - `start.sh` and `stop.sh`: Basic scripts to start and stop the trading system
   - `start_system.sh` and `stop_system.sh`: Enhanced scripts that check market calendar before starting/stopping

2. **Trainer Scripts**:
   - `start_trainer.sh` and `stop_trainer.sh`: Scripts to start and stop the ML model trainers

3. **Market Calendar**:
   - `market_calendar.sh`: Contains functions to check for trading days, weekends, and holidays

4. **Systemd Services**:
   - `trading-scheduler.service`: Runs 24/7 to manage the trading system based on market hours
   - `trainer-scheduler.service`: Runs 24/7 to manage the trainer based on a weekly schedule

## Market Calendar

The market calendar handles:
- Weekend detection (Saturday and Sunday)
- Holiday detection for major US market holidays
- Trading hours with configurable buffer periods

## Installation

To install the scheduling system:

1. Make sure all scripts are executable:
   ```
   chmod +x start.sh stop.sh start_trainer.sh stop_trainer.sh start_system.sh stop_system.sh market_calendar.sh install_services.sh
   ```

2. Run the installation script:
   ```
   ./install_services.sh
   ```

This will:
- Copy the systemd service files to the appropriate location
- Enable the services to start at boot
- Start the services immediately

## Manual Operation

You can manually operate the system using the following commands:

- Start the trading system (with market calendar checks):
  ```
  ./start_system.sh
  ```

- Stop the trading system (with market calendar checks):
  ```
  ./stop_system.sh
  ```

- Force stop the trading system regardless of market hours:
  ```
  ./stop_system.sh --force
  ```

- Start the ML model trainers:
  ```
  ./start_trainer.sh
  ```

- Stop the ML model trainers:
  ```
  ./stop_trainer.sh
  ```

## Service Management

The systemd services can be managed using standard systemd commands:

- Check service status:
  ```
  sudo systemctl status trading-scheduler.service
  sudo systemctl status trainer-scheduler.service
  ```

- Stop services:
  ```
  sudo systemctl stop trading-scheduler.service
  sudo systemctl stop trainer-scheduler.service
  ```

- Start services:
  ```
  sudo systemctl start trading-scheduler.service
  sudo systemctl start trainer-scheduler.service
  ```

- Disable services from starting at boot:
  ```
  sudo systemctl disable trading-scheduler.service
  sudo systemctl disable trainer-scheduler.service
  ```

## Configuration

- Market hours and buffer times can be configured in `market_calendar.sh`
- Holidays can be updated in `market_calendar.sh`
- Trainer schedule can be modified in `systemd/trainer-scheduler.service`
