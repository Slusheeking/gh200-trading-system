"""
Universal Redis and Shared Memory Integration Module

Provides high-performance integration for Redis and shared memory (CPU/GPU) with robust error handling,
modular design, and extensible APIs for real-time trading and analytics systems.
"""

import os
import time
import logging
import threading
import uuid
import importlib.util
import queue
from typing import Dict, Any, Optional, Tuple, Union, Callable, List
import numpy as np

# --- Logging Setup ---
def setup_logger(name="universal_redis_memory"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

# --- Module Availability Checks ---
def is_module_available(module_name):
    return importlib.util.find_spec(module_name) is not None

# Redis
try:
    import redis
    from redis.exceptions import RedisError
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("Redis Python client not installed. Please install with: pip install redis")

# CuPy (GPU)
HAS_CUPY = is_module_available("cupy")
if HAS_CUPY:
    try:
        import cupy as cp
    except ImportError:
        HAS_CUPY = False
        cp = None
        logger.warning("CuPy not available. GPU acceleration will be disabled.")

# Multiprocessing Shared Memory (CPU)
HAS_MULTIPROCESSING = is_module_available("multiprocessing")
HAS_SHARED_MEMORY = False
if HAS_MULTIPROCESSING:
    try:
        from multiprocessing import shared_memory
        HAS_SHARED_MEMORY = True
    except ImportError:
        logger.warning("Multiprocessing shared memory not available. CPU shared memory will be disabled.")
else:
    logger.warning("Multiprocessing module not available. CPU shared memory will be disabled.")

# --- SharedMemoryManager (CPU/GPU) ---
class SharedMemoryManager:
    """
    Manages shared memory blocks for CPU and GPU, with fallback to numpy arrays.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        memory_config = self.config.get("memory", {}).get("shared_memory", {})
        self.max_blocks = memory_config.get("max_blocks", 1000)
        self.default_block_size = memory_config.get("default_block_size", 1024 * 1024)
        self.cleanup_interval = memory_config.get("cleanup_interval", 60)
        self.block_ttl = memory_config.get("block_ttl", 300)
        self.blocks = {}
        self.block_metadata = {}
        self.block_lock = threading.RLock()
        self.gpu_memory_pool = None
        if HAS_CUPY and self.config.get("performance", {}).get("use_gpu", True):
            try:
                self.gpu_memory_pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(self.gpu_memory_pool.malloc)
                logger.info("Initialized GPU memory pool for shared memory")
            except Exception as e:
                logger.error(f"Failed to initialize GPU memory pool: {str(e)}")
        self.running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True, name="SharedMemoryCleanup"
        )
        self.cleanup_thread.start()
        logger.info(
            f"Shared memory manager initialized with max_blocks={self.max_blocks}, "
            f"default_block_size={self.default_block_size}, "
            f"GPU support: {HAS_CUPY}, CPU shared memory: {HAS_SHARED_MEMORY}"
        )

    def create_block(self, size: Optional[int] = None, name: Optional[str] = None, device: str = "cpu") -> str:
        with self.block_lock:
            if len(self.blocks) >= self.max_blocks:
                self._cleanup_expired_blocks()
                if len(self.blocks) >= self.max_blocks:
                    raise RuntimeError(f"Maximum number of shared memory blocks reached ({self.max_blocks})")
            block_id = name or str(uuid.uuid4())
            block_size = size or self.default_block_size
            if device == "gpu" and HAS_CUPY:
                try:
                    memory = cp.zeros(block_size // 4, dtype=cp.float32)
                    self.blocks[block_id] = memory
                    self.block_metadata[block_id] = {
                        "size": block_size, "device": "gpu", "created": time.time(),
                        "last_accessed": time.time(), "shape": memory.shape, "dtype": str(memory.dtype)
                    }
                    logger.debug(f"Created GPU shared memory block {block_id} with size {block_size}")
                    return block_id
                except Exception as e:
                    logger.error(f"Failed to create GPU shared memory block: {str(e)}")
                    device = "cpu"
            if HAS_SHARED_MEMORY:
                try:
                    shm = shared_memory.SharedMemory(name=block_id, create=True, size=block_size)
                    self.blocks[block_id] = shm
                    self.block_metadata[block_id] = {
                        "size": block_size, "device": "cpu", "created": time.time(),
                        "last_accessed": time.time(), "shape": (block_size,), "dtype": "uint8"
                    }
                    logger.debug(f"Created CPU shared memory block {block_id} with size {block_size}")
                    return block_id
                except Exception as e:
                    logger.error(f"Failed to create CPU shared memory block: {str(e)}")
                    raise
            else:
                memory = np.zeros(block_size, dtype=np.uint8)
                self.blocks[block_id] = memory
                self.block_metadata[block_id] = {
                    "size": block_size, "device": "cpu_fallback", "created": time.time(),
                    "last_accessed": time.time(), "shape": memory.shape, "dtype": str(memory.dtype)
                }
                logger.warning(f"Created fallback memory block {block_id} (shared memory not available)")
                return block_id

    def get_block(self, block_id: str) -> Any:
        with self.block_lock:
            if block_id not in self.blocks:
                raise KeyError(f"Shared memory block {block_id} not found")
            self.block_metadata[block_id]["last_accessed"] = time.time()
            return self.blocks[block_id]

    def get_array(self, block_id: str, shape: Tuple[int, ...], dtype: Any) -> Union[np.ndarray, Any]:
        with self.block_lock:
            if block_id not in self.blocks:
                raise KeyError(f"Shared memory block {block_id} not found")
            self.block_metadata[block_id]["last_accessed"] = time.time()
            block = self.blocks[block_id]
            device = self.block_metadata[block_id]["device"]
            if device == "gpu":
                return block.reshape(shape).astype(dtype)
            elif device == "cpu":
                return np.ndarray(shape, dtype=dtype, buffer=block.buf)
            else:
                return block.reshape(shape).astype(dtype)

    def put_array(self, array: Union[np.ndarray, Any], block_id: Optional[str] = None) -> str:
        if HAS_CUPY and isinstance(array, cp.ndarray):
            device = "gpu"
            size = array.nbytes
        else:
            device = "cpu"
            size = array.nbytes
        if block_id is None:
            block_id = self.create_block(size=size, device=device)
        else:
            with self.block_lock:
                if block_id not in self.blocks:
                    block_id = self.create_block(size=size, name=block_id, device=device)
                elif self.block_metadata[block_id]["size"] < size:
                    self.delete_block(block_id)
                    block_id = self.create_block(size=size, name=block_id, device=device)
        block = self.get_block(block_id)
        if device == "gpu":
            if isinstance(array, np.ndarray):
                block.set(array)
            else:
                cp.copyto(block, array)
        elif device == "cpu" and HAS_SHARED_MEMORY:
            np_array = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
            np.copyto(np_array, array)
        else:
            np.copyto(block, array.flatten())
        with self.block_lock:
            self.block_metadata[block_id].update({
                "shape": array.shape, "dtype": str(array.dtype), "last_accessed": time.time()
            })
        return block_id

    def delete_block(self, block_id: str) -> bool:
        with self.block_lock:
            if block_id not in self.blocks:
                return False
            block = self.blocks[block_id]
            metadata = self.block_metadata[block_id]
            try:
                if metadata["device"] == "gpu":
                    del block
                elif metadata["device"] == "cpu" and HAS_SHARED_MEMORY:
                    block.close()
                    block.unlink()
                del self.blocks[block_id]
                del self.block_metadata[block_id]
                return True
            except Exception as e:
                logger.error(f"Error deleting shared memory block {block_id}: {str(e)}")
                return False

    def _cleanup_expired_blocks(self):
        current_time = time.time()
        expired_blocks = [block_id for block_id, meta in self.block_metadata.items()
                          if current_time - meta["last_accessed"] > self.block_ttl]
        for block_id in expired_blocks:
            self.delete_block(block_id)
        if expired_blocks:
            logger.debug(f"Cleaned up {len(expired_blocks)} expired shared memory blocks")

    def _cleanup_worker(self):
        while self.running:
            try:
                time.sleep(self.cleanup_interval)
                with self.block_lock:
                    self._cleanup_expired_blocks()
            except Exception as e:
                logger.error(f"Error in shared memory cleanup worker: {str(e)}")

    def cleanup(self):
        self.running = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2.0)
        with self.block_lock:
            block_ids = list(self.blocks.keys())
            for block_id in block_ids:
                self.delete_block(block_id)
        if self.gpu_memory_pool is not None and HAS_CUPY:
            try:
                self.gpu_memory_pool.free_all_blocks()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                logger.error(f"Error cleaning up GPU memory pool: {str(e)}")
        logger.info("Shared memory manager cleaned up")

# --- RedisClient with Shared Memory Integration ---
class RedisClient:
    """
    Redis client with optional shared memory integration for high-performance data sharing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        redis_config = self.config.get("memory", {}).get("redis", {})
        if not redis_config:
            redis_config = self.config.get("redis", {})
        self.host = redis_config.get("host", "localhost")
        self.port = redis_config.get("port", 6379)
        self.db = redis_config.get("db", 0)
        self.password = redis_config.get("password", None)
        self.socket_timeout = redis_config.get("socket_timeout", 0.1)
        self.retry_on_timeout = redis_config.get("retry_on_timeout", True)
        self.max_retries = redis_config.get("max_retries", 3)
        self.retry_backoff = redis_config.get("retry_backoff", 0.1)
        self.use_shared_memory = redis_config.get("use_shared_memory", True) and HAS_SHARED_MEMORY
        self.shared_memory_manager = None
        if self.use_shared_memory:
            self.shared_memory_manager = get_shared_memory_manager(self.config)
        self.redis = None
        self.connection_pool = None
        self.initialized = False
        self.healthy = False
        self.last_health_check = 0
        self.lock = threading.RLock()
        self.logger = setup_logger("RedisClient")

    def initialize(self) -> bool:
        if self.initialized:
            return True
        if not HAS_REDIS:
            self.logger.error("Redis Python client not installed")
            return False
        try:
            self.connection_pool = redis.ConnectionPool(
                host=self.host, port=self.port, db=self.db, password=self.password,
                socket_timeout=self.socket_timeout, retry_on_timeout=self.retry_on_timeout, decode_responses=True
            )
            self.redis = redis.Redis(connection_pool=self.connection_pool)
            self.redis.ping()
            self.healthy = True
            self.last_health_check = time.time()
            self.initialized = True
            self.logger.info(f"Redis client initialized and connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {str(e)}")
            self.initialized = False
            self.healthy = False
            return False

    def get(self, key: str) -> Any:
        if not self.initialized:
            self.logger.warning("Redis client not initialized")
            return None
        try:
            return self.redis.get(key)
        except Exception as e:
            self.logger.error(f"Redis get error for key '{key}': {str(e)}")
            return None

    def set(self, key: str, value: str, expiry: Optional[int] = None) -> bool:
        if not self.initialized:
            self.logger.warning("Redis client not initialized")
            return False
        try:
            return self.redis.set(key, value, ex=expiry)
        except Exception as e:
            self.logger.error(f"Redis set error for key '{key}': {str(e)}")
            return False

    def set_array(self, key: str, array: Union[np.ndarray, Any], expiry: Optional[int] = None) -> bool:
        if not self.initialized:
            self.logger.warning("Redis client not initialized")
            return False
        if self.use_shared_memory and self.shared_memory_manager:
            try:
                block_id = self.shared_memory_manager.put_array(array)
                metadata = {
                    "block_id": block_id, "shape": array.shape, "dtype": str(array.dtype),
                    "timestamp": time.time(), "shared_memory": True
                }
                return self.redis.set(key, str(metadata), ex=expiry)
            except Exception as e:
                self.logger.error(f"Error storing array in shared memory: {str(e)}")
        try:
            # Fallback: store as bytes
            array_bytes = array.tobytes()
            return self.redis.set(key, array_bytes, ex=expiry)
        except Exception as e:
            self.logger.error(f"Error storing array as bytes: {str(e)}")
            return False

    def get_array(self, key: str) -> Optional[np.ndarray]:
        if not self.initialized:
            self.logger.warning("Redis client not initialized")
            return None
        try:
            metadata_str = self.get(key)
            if not metadata_str:
                return None
            if isinstance(metadata_str, dict) and metadata_str.get("shared_memory", False):
                block_id = metadata_str.get("block_id")
                shape = tuple(metadata_str.get("shape"))
                dtype = metadata_str.get("dtype")
                return self.shared_memory_manager.get_array(block_id, shape, dtype)
            # Fallback: try to interpret as bytes
            # (User must know shape/dtype)
            return None
        except Exception as e:
            self.logger.error(f"Error getting array from Redis: {str(e)}")
            return None

    def publish(self, channel: str, message: str) -> int:
        if not self.initialized:
            self.logger.warning("Redis client not initialized")
            return 0
        try:
            return self.redis.publish(channel, message)
        except Exception as e:
            self.logger.error(f"Redis publish error to channel '{channel}': {str(e)}")
            return 0

    def subscribe(self, channel: str, callback: Callable[[str, str], None]) -> bool:
        if not self.initialized:
            self.logger.warning("Redis client not initialized")
            return False
        try:
            pubsub = self.redis.pubsub()
            pubsub.subscribe(**{channel: lambda message: callback(channel, message["data"])})
            thread = threading.Thread(
                target=pubsub.run_in_thread, kwargs={"sleep_time": 0.001}, daemon=True, name=f"RedisSubscriber-{channel}"
            )
            thread.start()
            return True
        except Exception as e:
            self.logger.error(f"Redis subscribe error: {str(e)}")
            return False

    def cleanup(self):
        if self.connection_pool:
            try:
                self.connection_pool.disconnect()
                self.logger.info("Redis connection pool disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting Redis connection pool: {str(e)}")
        if self.shared_memory_manager:
            self.shared_memory_manager.cleanup()

# --- Utility Functions for Global Access ---
_shared_memory_manager = None
_redis_client = None

def get_shared_memory_manager(config: Optional[Dict[str, Any]] = None) -> SharedMemoryManager:
    global _shared_memory_manager
    if _shared_memory_manager is None:
        _shared_memory_manager = SharedMemoryManager(config)
    return _shared_memory_manager

def get_redis_client(config: Optional[Dict[str, Any]] = None) -> RedisClient:
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient(config)
        _redis_client.initialize()
    return _redis_client

# --- Exports ---
__all__ = [
    "SharedMemoryManager", "RedisClient", "get_shared_memory_manager", "get_redis_client",
    "HAS_REDIS", "HAS_CUPY", "HAS_SHARED_MEMORY"
]