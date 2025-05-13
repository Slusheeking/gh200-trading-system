"""
API Server for GH200 Trading System

This module provides a REST API server for the GH200 Trading System, exposing
metrics and data collected by the system exporter through an ngrok tunnel.
"""

import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, Depends, HTTPException, Security, status, Query, Path
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

# Import system exporter
from src.monitoring.system_exporter import SystemExporter

# Create FastAPI app
app = FastAPI(
    title="GH200 Trading System API",
    description="REST API for GH200 Trading System metrics and data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Global system exporter instance
system_exporter = None

# Authentication dependency
async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validate API key
    
    Args:
        api_key_header: API key from header
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key_header == app.state.config.get("api_key"):
        return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )

# Dependency to get system exporter
async def get_system_exporter():
    """
    Get system exporter instance
    
    Returns:
        System exporter instance
        
    Raises:
        HTTPException: If system exporter is not initialized
    """
    if system_exporter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System exporter not initialized",
        )
    
    return system_exporter

# Models for API responses
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: int = Field(..., description="Current timestamp")

# API routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health check response
    """
    return {
        "status": "ok",
        "version": app.version,
        "timestamp": int(time.time())
    }

@app.get("/metrics/system")
async def get_system_metrics(
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get system metrics
    
    Args:
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        System metrics
    """
    return exporter.get_system_metrics()

@app.get("/metrics/hardware")
async def get_hardware_metrics(
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get hardware metrics
    
    Args:
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Hardware metrics
    """
    return exporter.get_hardware_metrics()

@app.get("/metrics/portfolio")
async def get_portfolio_metrics(
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get portfolio metrics
    
    Args:
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Portfolio metrics
    """
    return exporter.get_portfolio_metrics()

@app.get("/trades/active")
async def get_active_trades(
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get active trades
    
    Args:
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Active trades
    """
    return exporter.get_active_trades()

@app.get("/trades/history")
async def get_trade_history(
    limit: int = Query(100, description="Maximum number of trades to return"),
    offset: int = Query(0, description="Offset for pagination"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get trade history
    
    Args:
        limit: Maximum number of trades to return
        offset: Offset for pagination
        symbol: Filter by symbol
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Trade history
    """
    return exporter.get_trade_history(limit=limit, offset=offset, symbol=symbol)

@app.get("/trades/statistics")
async def get_trade_statistics(
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get trade statistics
    
    Args:
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Trade statistics
    """
    return exporter.get_trade_statistics()

@app.get("/logs")
async def get_logs(
    level: Optional[str] = Query(None, description="Filter by log level"),
    limit: int = Query(100, description="Maximum number of logs to return"),
    offset: int = Query(0, description="Offset for pagination"),
    component: Optional[str] = Query(None, description="Filter by component"),
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get logs
    
    Args:
        level: Filter by log level
        limit: Maximum number of logs to return
        offset: Offset for pagination
        component: Filter by component
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Logs
    """
    return exporter.get_logs(level=level, limit=limit, offset=offset, component=component)

@app.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    category: Optional[str] = Query(None, description="Filter by category"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged status"),
    limit: int = Query(100, description="Maximum number of alerts to return"),
    offset: int = Query(0, description="Offset for pagination"),
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get alerts
    
    Args:
        severity: Filter by severity
        category: Filter by category
        acknowledged: Filter by acknowledged status
        limit: Maximum number of alerts to return
        offset: Offset for pagination
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Alerts
    """
    return exporter.get_alerts(
        severity=severity, category=category, acknowledged=acknowledged,
        limit=limit, offset=offset
    )

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str = Path(..., description="Alert ID"),
    user: str = Query("system", description="User who acknowledged the alert"),
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Acknowledge an alert
    
    Args:
        alert_id: Alert ID
        user: User who acknowledged the alert
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Acknowledgement result
    """
    result = exporter.acknowledge_alert(alert_id, user)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )
    
    return {"acknowledged": True}

@app.get("/market/chart/{symbol}")
async def get_chart_data(
    symbol: str = Path(..., description="Stock symbol"),
    period: str = Query("1y", description="Time period"),
    interval: str = Query("1d", description="Data interval"),
    chart_type: str = Query("candlestick", description="Chart type"),
    include_indicators: bool = Query(True, description="Whether to include technical indicators"),
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get chart data for a symbol
    
    Args:
        symbol: Stock symbol
        period: Time period
        interval: Data interval
        chart_type: Chart type
        include_indicators: Whether to include technical indicators
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Chart data
    """
    return exporter.get_chart_data(
        symbol=symbol, period=period, interval=interval,
        chart_type=chart_type, include_indicators=include_indicators
    )

@app.get("/market/quote/{symbol}")
async def get_quote(
    symbol: str = Path(..., description="Stock symbol"),
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get current quote for a symbol
    
    Args:
        symbol: Stock symbol
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Quote data
    """
    return exporter.get_quote(symbol)

@app.get("/trades/daily-pnl")
async def get_daily_pnl(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get daily profit/loss
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Daily profit/loss data
    """
    return exporter.get_daily_pnl(start_date=start_date, end_date=end_date)

@app.get("/trades/pnl-calendar")
async def get_pnl_calendar(
    year: Optional[int] = Query(None, description="Year (e.g., 2025)"),
    month: Optional[int] = Query(None, description="Month (1-12)"),
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get profit/loss calendar
    
    Args:
        year: Year (e.g., 2025)
        month: Month (1-12)
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        Profit/loss calendar
    """
    return exporter.get_pnl_calendar(year=year, month=month)

@app.get("/metrics/all")
async def get_all_metrics(
    api_key: str = Depends(get_api_key),
    exporter: SystemExporter = Depends(get_system_exporter)
):
    """
    Get all metrics
    
    Args:
        api_key: API key
        exporter: System exporter instance
        
    Returns:
        All metrics
    """
    return exporter.get_all_metrics()

def start_api_server(config: Dict[str, Any], exporter_instance: SystemExporter):
    """
    Start the API server
    
    Args:
        config: Configuration dictionary
        exporter_instance: System exporter instance
    """
    global system_exporter
    
    # Set system exporter instance
    system_exporter = exporter_instance
    
    # Set config
    app.state.config = config
    
    # Get API server settings
    host = config.get("api_server", {}).get("host", "0.0.0.0")
    port = config.get("api_server", {}).get("port", 8000)
    
    # Start API server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # This is for development/testing only
    import argparse
    from config.config_loader import load_config
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Start the API server")
    parser.add_argument("--config", type=str, default="settings/system.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create system exporter
    exporter = SystemExporter(config)
    
    # Start system exporter
    exporter.start()
    
    try:
        # Start API server
        start_api_server(config, exporter)
    finally:
        # Stop system exporter
        exporter.stop()
