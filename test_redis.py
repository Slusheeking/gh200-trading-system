import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load configuration
try:
    with open('config/system_settings.json', 'r') as f:
        config = json.load(f)
    logging.info("Loaded configuration from system_settings.json")
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    config = {}

# Import and initialize Redis client
try:
    from redis_service.redis_client import RedisClient
    logging.info("Successfully imported RedisClient")
    
    # Initialize Redis client
    redis_client = RedisClient(config)
    result = redis_client.initialize()
    
    if result:
        logging.info("Successfully initialized Redis client")
        # Test connection
        redis_client.set("test_key", "test_value")
        value = redis_client.get("test_key")
        logging.info(f"Test key value: {value}")
    else:
        logging.error("Failed to initialize Redis client")
        
except Exception as e:
    logging.error(f"Error with Redis client: {e}")