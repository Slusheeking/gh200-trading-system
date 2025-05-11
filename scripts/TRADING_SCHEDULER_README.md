# Trading System Scheduler

This service manages the lifecycle of the trading system, ensuring it starts 1 hour before market open and stops 1 hour after market close.

## Features

- Automatically starts the trading system 1 hour before market open (8:30 AM ET)
- Automatically stops the trading system 1 hour after market close (5:00 PM ET)
- Runs only on trading days (weekdays, excluding holidays)
- Monitors the trading system and restarts it if it crashes
- Logs all activities for monitoring and debugging

## Files

- `trading_scheduler.py` - The main Python script that manages the trading system
- `trading-scheduler.service` - The systemd service file
- `setup_trading_scheduler.sh` - Script to install and set up the service
- `manage_trading_scheduler.sh` - Script to manage the service (start, stop, restart, etc.)

## Installation

To install the trading scheduler service, run the setup script with sudo:

```bash
sudo ./scripts/setup_trading_scheduler.sh
```

This will:
1. Install required Python packages
2. Set up the log file
3. Install the systemd service
4. Enable and start the service

## Management

You can manage the trading scheduler service using the management script:

```bash
./scripts/manage_trading_scheduler.sh [start|stop|restart|status|logs]
```

Commands:
- `start` - Start the service
- `stop` - Stop the service
- `restart` - Restart the service
- `status` - Check the status of the service
- `logs` - View the service logs

## Logs

The trading scheduler logs to `/home/ubuntu/inavvi2/logs/trading_scheduler.log` and to the systemd journal.

To view the logs:

```bash
# View the log file
cat /home/ubuntu/inavvi2/logs/trading_scheduler.log

# View the systemd journal logs
sudo journalctl -u trading-scheduler.service -f
```

## Configuration

The trading scheduler is configured in the `trading_scheduler.py` script. You can modify the following parameters:

- `MARKET_OPEN_TIME` - Market open time (default: 9:30 AM ET)
- `MARKET_CLOSE_TIME` - Market close time (default: 4:00 PM ET)
- `STARTUP_BUFFER_HOURS` - Hours before market open to start the system (default: 1)
- `SHUTDOWN_BUFFER_HOURS` - Hours after market close to stop the system (default: 1)

After modifying the configuration, restart the service:

```bash
sudo systemctl restart trading-scheduler.service
```

## Troubleshooting

If the service is not working as expected:

1. Check the service status:
   ```bash
   sudo systemctl status trading-scheduler.service
   ```

2. Check the logs:
   ```bash
   sudo journalctl -u trading-scheduler.service -f
   ```

3. Ensure the Python script has the correct permissions:
   ```bash
   chmod +x /home/ubuntu/inavvi2/scripts/trading_scheduler.py
   ```

4. Verify the service file is correctly installed:
   ```bash
   cat /etc/systemd/system/trading-scheduler.service
   ```

5. Restart the service:
   ```bash
   sudo systemctl restart trading-scheduler.service