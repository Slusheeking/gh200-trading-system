#!/bin/bash
# Market calendar utility functions

# Check if today is a trading day (not weekend or holiday)
is_market_day() {
    # Get current date
    local day_of_week=$(date +%u)
    local date_str=$(date +%Y-%m-%d)
    
    # Check if weekend (6=Saturday, 7=Sunday)
    if [ "$day_of_week" -ge 6 ]; then
        return 1  # Not a trading day
    fi
    
    # Check if holiday (US market holidays for 2025)
    case "$date_str" in
        "2025-01-01") return 1 ;;  # New Year's Day
        "2025-01-20") return 1 ;;  # Martin Luther King Jr. Day
        "2025-02-17") return 1 ;;  # Presidents' Day
        "2025-04-18") return 1 ;;  # Good Friday
        "2025-05-26") return 1 ;;  # Memorial Day
        "2025-06-19") return 1 ;;  # Juneteenth
        "2025-07-04") return 1 ;;  # Independence Day
        "2025-09-01") return 1 ;;  # Labor Day
        "2025-11-27") return 1 ;;  # Thanksgiving Day
        "2025-12-25") return 1 ;;  # Christmas Day
    esac
    
    return 0  # Is a trading day
}

# Check if markets are currently open
is_market_open() {
    # First check if it's a trading day
    if ! is_market_day; then
        return 1  # Not open
    fi
    
    # Get current time in ET
    local hour=$(TZ="America/New_York" date +%H)
    local minute=$(TZ="America/New_York" date +%M)
    local time_val=$((hour * 60 + minute))
    
    # Market hours: 9:30 AM - 4:00 PM ET
    local market_open=$((9 * 60 + 30))
    local market_close=$((16 * 60))
    
    if [ "$time_val" -ge "$market_open" ] && [ "$time_val" -lt "$market_close" ]; then
        return 0  # Market is open
    else
        return 1  # Market is closed
    fi
}
