#!/bin/bash
# manage_trading_scheduler.sh - Management script for the trading scheduler service

# Function to display usage
usage() {
    echo "Usage: $0 [start|stop|restart|status|logs]"
    echo "  start   - Start the trading scheduler service"
    echo "  stop    - Stop the trading scheduler service"
    echo "  restart - Restart the trading scheduler service"
    echo "  status  - Check the status of the trading scheduler service"
    echo "  logs    - View the logs of the trading scheduler service"
    exit 1
}

# Check if command is provided
if [ $# -eq 0 ]; then
    usage
fi

# Process command
case "$1" in
    start)
        echo "Starting trading scheduler service..."
        sudo systemctl start trading-scheduler.service
        ;;
    stop)
        echo "Stopping trading scheduler service..."
        sudo systemctl stop trading-scheduler.service
        ;;
    restart)
        echo "Restarting trading scheduler service..."
        sudo systemctl restart trading-scheduler.service
        ;;
    status)
        echo "Checking trading scheduler service status..."
        sudo systemctl status trading-scheduler.service
        ;;
    logs)
        echo "Viewing trading scheduler service logs..."
        sudo journalctl -u trading-scheduler.service -f
        ;;
    *)
        usage
        ;;
esac

exit 0