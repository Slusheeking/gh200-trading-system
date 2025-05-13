"""
Ngrok Tunnel for GH200 Trading System API

This module provides functionality to create an ngrok tunnel to expose the API server.
"""

import time
import logging
import subprocess
import requests
from typing import Optional

def start_ngrok_tunnel(port: int = 8000) -> Optional[str]:
    """
    Start ngrok tunnel to expose the API server
    
    Args:
        port: Port number of the API server
        
    Returns:
        Tunnel URL if successful, None otherwise
    """
    # Ngrok auth token and API key
    auth_token = "2vB4mEpkOKCPryJJTqcnQZu17mU_2mHUjAc8Gp4egYp8iDVRJ"
    api_key = "2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
    
    try:
        # Check if ngrok is installed
        try:
            subprocess.run(["ngrok", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            logging.error("ngrok is not installed. Please install it first.")
            return None
        
        # Configure ngrok with auth token
        subprocess.run(["ngrok", "config", "add-authtoken", auth_token], check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Start ngrok tunnel
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", f"localhost:{port}", "--log=stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for tunnel to start
        time.sleep(3)
        
        # Check if process is still running
        if ngrok_process.poll() is not None:
            stderr = ngrok_process.stderr.read()
            logging.error(f"Ngrok process exited: {stderr}")
            return None
        
        # Get tunnel URL from local API
        try:
            response = requests.get("http://localhost:4040/api/tunnels")
            if response.status_code == 200:
                data = response.json()
                for tunnel in data.get("tunnels", []):
                    if tunnel.get("proto") == "https":
                        tunnel_url = tunnel.get("public_url")
                        logging.info(f"Ngrok tunnel started: {tunnel_url}")
                        return tunnel_url
        except Exception as e:
            logging.error(f"Error getting tunnel URL from local API: {str(e)}")
        
        # If local API fails, try using the ngrok API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Ngrok-Version": "2"
        }
        
        response = requests.get("https://api.ngrok.com/tunnels", headers=headers)
        if response.status_code == 200:
            data = response.json()
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    tunnel_url = tunnel.get("public_url")
                    logging.info(f"Ngrok tunnel started: {tunnel_url}")
                    return tunnel_url
        
        logging.error("Failed to get tunnel URL")
        return None
    
    except Exception as e:
        logging.error(f"Error starting ngrok tunnel: {str(e)}")
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s"
    )
    
    # Start ngrok tunnel
    tunnel_url = start_ngrok_tunnel()
    
    if tunnel_url:
        print(f"Ngrok tunnel URL: {tunnel_url}")
        
        # Keep the script running
        try:
            print("Press Ctrl+C to stop the tunnel")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping ngrok tunnel")
    else:
        print("Failed to start ngrok tunnel")
