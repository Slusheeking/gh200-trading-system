# Memory management package for GH200 Trading System
"""
This package provides memory management components for the GH200 Trading System.
It includes Redis integration for signal processing, distribution, and storage,
as well as shared memory components for high-performance data sharing between
different components of the system.
"""

# Import only what's needed for type checking
import importlib.util
from typing import Any

# Define __all__ to control what's exported
__all__ = ['RedisClient', 'RedisSignalHandler']

# Create a proxy class for lazy loading
class _LazyLoader:
    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._instance = None
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._instance is None:
            module = importlib.import_module(self.module_name)
            self._instance = getattr(module, self.class_name)
        return self._instance(*args, **kwargs)

# Create lazy loaders for Redis components
RedisClient = _LazyLoader('.redis_client', 'RedisClient')
RedisSignalHandler = _LazyLoader('.redis_client', 'RedisSignalHandler')

# Check if shared memory is available
HAS_SHARED_MEMORY = importlib.util.find_spec("memory.shared_memory") is not None

# Create lazy-loaded functions for shared memory components
if HAS_SHARED_MEMORY:
    # Add to __all__
    __all__.extend(['get_shared_memory_manager', 'SharedMemoryManager', 'SharedGPUMemoryPool'])
    
    # Create lazy loaders
    get_shared_memory_manager = _LazyLoader('.shared_memory', 'get_shared_memory_manager')
    SharedMemoryManager = _LazyLoader('.shared_memory', 'SharedMemoryManager')
    SharedGPUMemoryPool = _LazyLoader('.shared_memory', 'SharedGPUMemoryPool')
