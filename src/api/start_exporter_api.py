"""
Start System Exporter API with Ngrok Tunnel

This script starts the system exporter, API server, and ngrok tunnel.
"""

import os
import sys
import time
import logging
import argparse
import threading
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import modules
from config.config_loader import load_config
from src.monitoring.system_exporter import SystemExporter
from src.api.api_server import app
from src.api.ngrok_tunnel import start_ngrok_tunnel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)

def start_api_server_thread(config: Dict[str, Any], exporter: SystemExporter):
    """
    Start API server in a separate thread
    
    Args:
        config: Configuration dictionary
        exporter: System exporter instance
    """
    # Get API server settings
    host = config.get("api_server", {}).get("host", "0.0.0.0")
    port = config.get("api_server", {}).get("port", 8000)
    
    # Set system exporter instance and config in app state
    app.state.system_exporter = exporter
    app.state.config = config
    
    # Import uvicorn here to avoid circular imports
    import uvicorn
    
    # Start API server
    uvicorn.run(app, host=host, port=port)

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Start System Exporter API with Ngrok Tunnel")
    parser.add_argument("--config", type=str, default="settings/system.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Add API key to config if not present
    if "api_key" not in config:
        config["api_key"] = "2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
    
    # Create system exporter
    exporter = SystemExporter(config)
    
    # Start system exporter
    logging.info("Starting system exporter")
    exporter.start()
    
    try:
        # Get API server port
        port = config.get("api_server", {}).get("port", 8000)
        
        # Start API server in a separate thread
        api_thread = threading.Thread(
            target=start_api_server_thread,
            args=(config, exporter),
            daemon=True
        )
        api_thread.start()
        
        # Wait for API server to start
        logging.info("Waiting for API server to start")
        time.sleep(3)
        
        # Start ngrok tunnel
        logging.info("Starting ngrok tunnel")
        tunnel_url = start_ngrok_tunnel(port)
        
        if tunnel_url:
            logging.info(f"System exporter API available at: {tunnel_url}")
            
            # Keep the script running
            try:
                logging.info("Press Ctrl+C to stop")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("Stopping")
        else:
            logging.error("Failed to start ngrok tunnel")
    
    finally:
        # Stop system exporter
        logging.info("Stopping system exporter")
        exporter.stop()

if __name__ == "__main__":
    main()
