"""
GH200-Optimized HFT model implementation for high-performance inference

This module provides a production-ready implementation of the Gradient Boosted Decision Trees
model optimized for NVIDIA GH200 hardware. It leverages GPU acceleration, INT8 quantization,
and CUDA streams for parallel processing to achieve maximum performance on high-frequency
trading workloads.
"""

import os
import time
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
import threading
import json
import importlib.util
from pathlib import Path
from contextlib import contextmanager
from monitoring.log import logging as log_utils
from monitoring.collectors.system_metrics_collector import MetricsCollector
from data.polygon_market_data_provider import PolygonMarketDataProvider
from ml.fast_exit_strategy import FastExitStrategy, Signal


# Check for module availability using importlib.util.find_spec
def is_module_available(module_name):
    """Check if a module is available without importing it."""
    return importlib.util.find_spec(module_name) is not None


HAS_LIGHTGBM = is_module_available("lightgbm")
HAS_CUPY = is_module_available("cupy")
HAS_NUMBA = is_module_available("numba")
HAS_TENSORRT = is_module_available("tensorrt") and is_module_available("pycuda")
HAS_TREELITE = is_module_available("treelite") and is_module_available(
    "treelite_runtime"
)

# Import dependencies if available
if HAS_LIGHTGBM:
    import lightgbm as lgb
else:
    logging.warning("LightGBM not available. GBDT model will not function.")

if HAS_CUPY:
    import cupy as cp
else:
    logging.warning("CuPy not available. GPU acceleration will be limited.")

if HAS_NUMBA:
    from numba import cuda, float32, int8, int32
    # Import jit/njit only if we're going to use them for CPU functions
    # from numba import jit, njit
else:
    logging.warning("Numba not available. Custom CUDA kernels will not be available.")

if HAS_TENSORRT:
    import tensorrt as trt
    import pycuda.driver as cuda_driver
    # Note: We don't import pycuda.autoinit to have explicit control over CUDA initialization
else:
    logging.warning("TensorRT not available. INT8 inference will be simulated.")

if HAS_TREELITE:
    try:
        import treelite
        import treelite_runtime

        logging.info("Treelite available for optimized tree model deployment.")
    except ImportError:
        logging.warning(
            "Failed to import treelite modules. Disabling treelite support."
        )
        HAS_TREELITE = False
else:
    logging.warning("Treelite not available. Falling back to LightGBM/TensorRT.")

# Define CUDA kernels for optimized processing if Numba is available
if HAS_NUMBA:

    @cuda.jit(device=True, inline=True)
    def _clip_value(val, min_val, max_val):
        """Helper to clip value within min/max."""
        if val < min_val:
            return min_val
        if val > max_val:
            return max_val
        return val

    @cuda.jit
    def int8_quantize_kernel(
        input_array, output_array, scales, zero_points, channel_dim=None
    ):
        """CUDA kernel for INT8 quantization with per-channel support"""
        i = int32(cuda.grid(1))  # Use int32 for array indices
        if i < input_array.size:
            # Per-feature quantization if channel_dim is provided
            if channel_dim is not None:
                # Find which channel this element belongs to
                channel = (i // channel_dim) % input_array.shape[channel_dim]
                scale = scales[channel]
                zero_point = zero_points[channel]
            else:
                # Global quantization
                scale = scales[0]
                zero_point = zero_points[0]

            # Quantize to INT8 range
            val = _clip_value(input_array[i] / scale + zero_point, -128.0, 127.0)
            output_array[i] = int8(val)  # Use Numba int8 type

    @cuda.jit
    def int8_dequantize_kernel(
        input_array, output_array, scales, zero_points, channel_dim=None
    ):
        """CUDA kernel for INT8 dequantization with per-channel support"""
        i = int32(cuda.grid(1))  # Use int32 for array indices
        if i < input_array.size:
            # Per-feature dequantization if channel_dim is provided
            if channel_dim is not None:
                # Find which channel this element belongs to
                channel = (i // channel_dim) % input_array.shape[channel_dim]
                scale = scales[channel]
                zero_point = zero_points[channel]
            else:
                # Global dequantization
                scale = scales[0]
                zero_point = zero_points[0]

            # Dequantize from INT8 range
            output_array[i] = (float32(input_array[i]) - zero_point) * scale

    @cuda.jit
    def batch_preprocess_kernel(input_array, output_array, feature_means, feature_stds):
        """CUDA kernel for batch preprocessing (normalization)"""
        row, col = cuda.grid(2)
        if row < output_array.shape[0] and col < output_array.shape[1]:
            if feature_stds[col] > 1e-7:  # Avoid division by extremely small values
                output_array[row, col] = (
                    input_array[row, col] - feature_means[col]
                ) / feature_stds[col]
            else:
                output_array[row, col] = input_array[row, col] - feature_means[col]

    @cuda.jit
    def batch_preprocess_soa_kernel(
        input_features,
        output_features,
        feature_means,
        feature_stds,
        batch_size,
        num_features,
    ):
        """CUDA kernel for batch preprocessing with Structure-of-Arrays layout"""
        feature_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        sample_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if feature_idx < num_features and sample_idx < batch_size:
            # SoA layout: All values for feature_idx, then all values for next feature
            input_idx = feature_idx * batch_size + sample_idx
            output_idx = feature_idx * batch_size + sample_idx

            # Normalize feature
            if feature_stds[feature_idx] > 1e-7:
                output_features[output_idx] = (
                    input_features[input_idx] - feature_means[feature_idx]
                ) / feature_stds[feature_idx]
            else:
                output_features[output_idx] = (
                    input_features[input_idx] - feature_means[feature_idx]
                )


class GH200GBDTModel:
    """
    Production-ready GBDT model optimized for NVIDIA GH200 hardware.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.logger = log_utils.setup_logger("gh200_gbdt_model")
        self.logger.info("Creating GH200-optimized GBDT model")

        if config is None:
            # Load HFT model specific settings from dedicated config file
            hft_config_path = (
                "/home/ubuntu/gh200-trading-system/config/hft_model_settings.json"
            )
            try:
                with open(hft_config_path, "r") as f:
                    config = json.load(f)
                self.logger.info(f"Loaded HFT model settings from {hft_config_path}")
            except Exception as e:
                self.logger.error(
                    f"Failed to load HFT model settings from {hft_config_path}: {e}"
                )
                # No fallback, raise error if config can't be loaded
                raise RuntimeError(
                    f"Could not load required configuration from {hft_config_path}"
                )

        # Cache for preprocessed features between preprocessing and inference
        self.last_processed_features = None

        # Resource management
        self.model_lock = threading.RLock()
        self._is_closing = False

        # Extract configuration sections
        hardware_config = config.get("hardware", {})
        ml_config = config.get(
            "model", {}
        )  # Use "model" section from hft_model_settings.json
        inference_config = config.get("inference", {})
        monitoring_config = config.get("monitoring", {})

        project_root = Path(__file__).resolve().parents[2]
        gbdt_config_path = project_root / "models" / "gbdt" / "model_config.json"
        model_cache_dir = project_root / "models" / "cache"

        # Create model cache directory if it doesn't exist
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        if gbdt_config_path.exists():
            with open(gbdt_config_path, "r") as f:
                model_params_config = json.load(f)
        else:
            self.logger.warning(
                f"GBDT model_config.json not found at {gbdt_config_path}. Using defaults from main config."
            )
            model_params_config = (
                ml_config  # Use model section directly from hft_model_settings.json
            )

        # Core model parameters
        self.num_trees = model_params_config.get("num_trees", 100)
        self.num_leaves = model_params_config.get("num_leaves", 31)
        self.num_boost_round = model_params_config.get(
            "num_boost_round", self.num_trees
        )
        self.objective = model_params_config.get("objective", "binary")
        self.metric = model_params_config.get("metric", "auc")
        self.learning_rate = model_params_config.get("learning_rate", 0.1)
        self.feature_fraction = model_params_config.get("feature_fraction", 0.8)
        self.bagging_fraction = model_params_config.get("bagging_fraction", 0.8)

        # Inference optimization settings
        self.use_fp16 = inference_config.get("use_fp16", True)
        self.use_int8 = inference_config.get("use_int8", True)
        self.max_batch_size = inference_config.get("batch_size", 256)
        self.use_soa_layout = inference_config.get(
            "use_soa_layout", True
        )  # Structure-of-Arrays
        self.use_channel_quantization = inference_config.get(
            "use_channel_quantization", True
        )
        self.use_treelite = inference_config.get("use_treelite", HAS_TREELITE)

        # Hardware settings
        self.device_type = hardware_config.get("device_type", "cuda")  # 'cuda' or 'cpu'
        self.gpu_device_id = hardware_config.get("gpu_device_id", 0)
        self.use_gpu = (
            (self.device_type == "cuda") and (HAS_LIGHTGBM or HAS_TREELITE) and HAS_CUPY
        )

        # TensorRT settings
        self.use_tensorrt = (
            inference_config.get("use_tensorrt", True) and HAS_TENSORRT and self.use_gpu
        )
        self.trt_workspace_size = inference_config.get(
            "trt_workspace_size", 1 << 30
        )  # 1GB default
        self.trt_engine = None
        self.trt_context = None
        self.trt_bindings = []
        self.trt_stream = None
        self.trt_engine_path = model_cache_dir / "tensorrt_engine.engine"

        # GH200-specific optimizations
        self.gh200_optimizations = (
            hardware_config.get("gh200_optimizations", True) and self.use_gpu
        )
        self.use_cuda_streams = (
            hardware_config.get("use_cuda_streams", True) and self.gh200_optimizations
        )
        self.cuda_stream_count = hardware_config.get(
            "cuda_stream_count", 4
        )  # Adjusted default
        self.use_pinned_memory = (
            hardware_config.get("use_pinned_memory", True) and self.gh200_optimizations
        )
        self.use_unified_memory = (
            hardware_config.get("use_unified_memory", True) and self.gh200_optimizations
        )
        self.grace_cpu_offload = (
            hardware_config.get("grace_cpu_offload", True) and self.gh200_optimizations
        )

        # Memory pool sizes in bytes
        self.gpu_memory_pool_size = hardware_config.get(
            "gpu_memory_pool_size", 1 << 30
        )  # 1GB default
        self.pinned_memory_pool_size = hardware_config.get(
            "pinned_memory_pool_size", 256 << 20
        )  # 256MB default

        self.input_shape = [self.max_batch_size, 0]  # Num features will be set on load
        self.output_shape = [self.max_batch_size, 1]

        self.model_path: Optional[str] = None
        self.is_loaded = False
        self.booster: Optional[lgb.Booster] = None
        self.treelite_model = None
        self.predictor = None
        self.model_version = 1

        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        self.num_threads = inference_config.get(
            "inference_threads", os.cpu_count() or 1
        )

        # Memory pools and streams
        self.cuda_streams: List[Any] = []
        self.memory_pool = None
        self.pinned_memory_pool = None
        self.stream_pool = {}  # Map sizes to allocated streams

        # Initialize CUDA resources
        self.cuda_initialized = False
        self._init_cuda_resources()

        # Initialize quantization parameters
        self.quant_scales = np.array(
            [1.0], dtype=np.float32
        )  # Will be resized for per-channel
        self.quant_zero_points = np.array(
            [0.0], dtype=np.float32
        )  # Will be resized for per-channel

        # Initialize metrics collector
        self.enable_metrics = monitoring_config.get("enable_metrics", True)
        if self.enable_metrics:
            self.metrics = MetricsCollector(
                service_name="gbdt_inference",
                namespace="hft_ml",
                labels={"model": "gh200_gbdt", "version": str(self.model_version)},
            )
        else:
            self.metrics = None

        # Performance stats
        self.inference_count = 0
        self.total_inference_time_ms = 0
        self.latency_p50 = 0
        self.latency_p99 = 0
        self.latency_buffer = []
        self.max_latency_buffer_size = 1000
        self.stats_lock = threading.Lock()

        self.logger.info(
            f"GH200 GBDT Model initialized. GPU: {self.use_gpu}, "
            f"GH200 Opts: {self.gh200_optimizations}, TensorRT: {self.use_tensorrt}, "
            f"Treelite: {self.use_treelite}, SoA layout: {self.use_soa_layout}"
        )

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources when exiting context."""
        self.cleanup_all()
        return False  # Don't suppress exceptions

    def _init_cuda_resources(self):
        """Initialize CUDA resources including memory pools, streams, and context."""
        if not self.use_gpu:
            return

        try:
            # Initialize CUDA explicitly if using TensorRT
            if self.use_tensorrt and HAS_TENSORRT:
                if not self.cuda_initialized:
                    # Initialize CUDA driver API explicitly instead of using pycuda.autoinit
                    cuda_driver.init()
                    self.cuda_device = cuda_driver.Device(self.gpu_device_id)
                    self.cuda_context = self.cuda_device.make_context()
                    self.cuda_initialized = True
                    self.logger.info(
                        f"Explicitly initialized CUDA for device {self.gpu_device_id}"
                    )

            # Set CuPy device
            if HAS_CUPY:
                cp.cuda.runtime.setDevice(self.gpu_device_id)

                # Initialize memory pools
                if self.gh200_optimizations:
                    # Create memory pool with specified size limit
                    self.memory_pool = cp.cuda.MemoryPool()
                    cp.cuda.set_allocator(self.memory_pool.malloc)

                    if self.use_pinned_memory:
                        self.pinned_memory_pool = cp.cuda.PinnedMemoryPool()
                        cp.cuda.set_pinned_memory_allocator(
                            self.pinned_memory_pool.malloc
                        )
                        self.logger.info(
                            f"Initialized pinned memory pool with size {self.pinned_memory_pool_size // (1024 * 1024)}MB"
                        )

                # Initialize CUDA streams
                if self.use_cuda_streams:
                    self.cuda_streams = [
                        cp.cuda.Stream(non_blocking=True)
                        for _ in range(self.cuda_stream_count)
                    ]
                    self.logger.info(
                        f"Initialized {self.cuda_stream_count} CUDA streams on GPU {self.gpu_device_id}"
                    )

                # Get device properties
                device_props = cp.cuda.runtime.getDeviceProperties(self.gpu_device_id)

                # Store device properties for kernel optimizations
                self.max_threads_per_block = int32(device_props["maxThreadsPerBlock"])
                self.max_threads_per_dim = (
                    int32(device_props["maxThreadsDim"][0]),
                    int32(device_props["maxThreadsDim"][1]),
                    int32(device_props["maxThreadsDim"][2]),
                )
                self.max_grid_size = (
                    int32(device_props["maxGridSize"][0]),
                    int32(device_props["maxGridSize"][1]),
                    int32(device_props["maxGridSize"][2]),
                )
                self.warp_size = int32(device_props["warpSize"])

                # Store compute capability for kernel optimizations
                self.compute_capability = (
                    int32(device_props["major"]),
                    int32(device_props["minor"]),
                )

                # Log device capabilities
                self.logger.info(
                    f"CUDA device {self.gpu_device_id}: {device_props['name'].decode()}, "
                    f"Compute: {self.compute_capability[0]}.{self.compute_capability[1]}, "
                    f"Memory: {device_props['totalGlobalMem'] // (1024 * 1024)}MB, "
                    f"Max threads per block: {self.max_threads_per_block}"
                )

            if self.gh200_optimizations and HAS_NUMBA:
                # ARM64 specific Numba flags for Grace CPU
                os.environ["NUMBA_CPU_NAME"] = "arm64"
                os.environ["NUMBA_CPU_FEATURES"] = "+sve,+dotprod,+fp16fml"
                # Enable Numba JIT compilation optimizations
                os.environ["NUMBA_OPT"] = "3"  # Maximum optimization level
                os.environ["NUMBA_FASTMATH"] = "1"  # Enable fast math
                # Enable Numba cache
                os.environ["NUMBA_CACHE_DIR"] = str(Path.home() / ".numba_cache")
                self.logger.info(
                    "Set ARM64-specific Numba flags and JIT optimizations for GH200."
                )

        except Exception as e:
            self.logger.error(
                f"Failed to initialize CUDA resources: {e}. Disabling GPU usage.",
                exc_info=True,
            )
            self.use_gpu = False
            self.gh200_optimizations = False
            self.use_tensorrt = False

    @contextmanager
    def _get_stream(self, size_hint: int = None):
        """
        Get an appropriate CUDA stream for a given workload size.
        This is a context manager that returns a stream and ensures synchronization.

        Args:
            size_hint: Hint for workload size to optimize stream selection

        Yields:
            A CUDA stream
        """
        if not self.use_cuda_streams or not self.cuda_streams:
            yield None
            return

        # Use size hint to choose appropriate stream
        if size_hint is not None:
            # Map size to a stream index - larger batches get their own streams
            # to minimize contention
            if size_hint > 1000:
                # For very large batches, use a dedicated high-priority stream
                if "large_batch" not in self.stream_pool:
                    self.stream_pool["large_batch"] = cp.cuda.Stream(
                        priority=1
                    )  # Higher priority
                stream = self.stream_pool["large_batch"]
                self.logger.debug(
                    f"Using dedicated high-priority stream for large batch (size {size_hint})"
                )
            elif size_hint > 100:
                # For medium batches, use a medium-priority stream
                if "medium_batch" not in self.stream_pool:
                    self.stream_pool["medium_batch"] = cp.cuda.Stream(
                        priority=0
                    )  # Normal priority
                stream = self.stream_pool["medium_batch"]
                self.logger.debug(
                    f"Using medium-priority stream for medium batch (size {size_hint})"
                )
            else:
                # For small batches, use round-robin from the main pool
                stream_idx = self.inference_count % len(self.cuda_streams)
                stream = self.cuda_streams[stream_idx]
                self.logger.debug(
                    f"Using stream {stream_idx} for small batch (size {size_hint})"
                )
        else:
            # Default to round-robin scheduling if no size hint
            stream_idx = self.inference_count % len(self.cuda_streams)
            stream = self.cuda_streams[stream_idx]
            self.logger.debug(f"Using stream {stream_idx} (round-robin)")

        try:
            yield stream
        finally:
            # Always synchronize to ensure operations complete
            stream.synchronize()

    def load(self, model_path: str, feature_stats_path: Optional[str] = None) -> None:
        """
        Load a LightGBM model and optimize it for GH200 inference.

        Args:
            model_path: Path to the LightGBM model file
            feature_stats_path: Optional path to feature statistics for normalization
        """
        with self.model_lock:
            self.logger.info(f"Loading GH200-optimized GBDT model from {model_path}")

            if not HAS_LIGHTGBM and not HAS_TREELITE:
                raise RuntimeError(
                    "Neither LightGBM nor Treelite is available. Cannot load GBDT model."
                )

            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model_path = model_path
            self.model_version += 1  # Increment model version

            # Update metrics labels with new model version
            if self.metrics:
                self.metrics.update_labels({"version": str(self.model_version)})

            params = {
                "objective": self.objective,
                "metric": self.metric,
                "boosting_type": "gbdt",
                "num_leaves": self.num_leaves,
                "learning_rate": self.learning_rate,
                "feature_fraction": self.feature_fraction,
                "bagging_fraction": self.bagging_fraction,
                "bagging_freq": 5,  # Common default
                "verbose": -1,
                "n_jobs": self.num_threads,
            }

            if self.use_gpu and HAS_LIGHTGBM:
                params["device_type"] = "cuda"
                params["gpu_platform_id"] = 0  # Assuming single platform
                params["gpu_device_id"] = self.gpu_device_id
                # LightGBM GPU uses float32 by default. FP16 might need specific compilation/settings.
                # For now, we rely on INT8 quantization for speed if enabled.
                self.logger.info(
                    f"Configuring LightGBM for GPU device {self.gpu_device_id}"
                )
            else:
                params["device_type"] = "cpu"
                self.logger.info("Configuring LightGBM for CPU execution")

            try:
                load_start_time = time.perf_counter()

                # Load booster
                if HAS_LIGHTGBM:
                    self.booster = lgb.Booster(
                        model_file=self.model_path, params=params
                    )
                    self._extract_model_metadata()

                # If Treelite is available, convert and compile model for optimized inference
                if self.use_treelite and HAS_TREELITE:
                    self._convert_to_treelite()

                if feature_stats_path and Path(feature_stats_path).exists():
                    self.load_feature_stats(feature_stats_path)
                else:
                    self.logger.warning(
                        f"Feature stats file not provided or not found: {feature_stats_path}. "
                        "Normalization might be suboptimal."
                    )

                if self.gh200_optimizations:
                    self._apply_gh200_optimizations()

                if (
                    self.use_int8 and self.use_gpu
                ):  # INT8 only makes sense with GPU for now
                    self._apply_int8_quantization()

                # Convert to TensorRT if enabled and Treelite isn't used
                if (
                    self.use_tensorrt
                    and self.use_gpu
                    and not (self.use_treelite and self.predictor)
                ):
                    # First check if we already have a cached engine file
                    if self.trt_engine_path.exists():
                        self._load_trt_engine_from_cache()
                    else:
                        self._convert_to_tensorrt()
                        self._save_trt_engine_to_cache()

                # Warmup the model to ensure consistent performance
                self._warmup_model()

                load_time = time.perf_counter() - load_start_time
                self.logger.info(f"Model loading time: {load_time:.2f} seconds")

                self.is_loaded = True
                self.logger.info(
                    f"Model loaded successfully. Features: {len(self.feature_names)}. "
                    f"Using: {'Treelite' if self.predictor else 'TensorRT' if self.trt_engine else 'LightGBM'}"
                )

                # Record model loading metrics
                if self.metrics:
                    self.metrics.record_gauge("model_load_time_seconds", load_time)
                    self.metrics.record_gauge(
                        "model_num_features", len(self.feature_names)
                    )
                    self.metrics.record_gauge("model_num_trees", self.num_trees)

            except Exception as e:
                self.logger.error(f"Failed to load LightGBM model: {e}", exc_info=True)
                raise RuntimeError(
                    f"Failed to load GBDT model from {self.model_path}"
                ) from e

    def _warmup_model(self, num_samples: int = 10):
        """
        Warm up the model with some synthetic data to prime caches and JIT compilation.

        Args:
            num_samples: Number of synthetic samples to use for warmup
        """
        if not self.is_loaded:
            return

        self.logger.info(f"Warming up model with {num_samples} synthetic samples")
        try:
            # Generate random features
            if self.feature_means is not None and self.feature_stds is not None:
                # Generate data that follows the distribution of real data
                features = np.random.normal(
                    loc=self.feature_means,
                    scale=self.feature_stds,
                    size=(num_samples, len(self.feature_names)),
                ).astype(np.float32)
            else:
                # If we don't have stats, just use random uniform between -1 and 1
                features = np.random.uniform(
                    low=-1, high=1, size=(num_samples, len(self.feature_names))
                ).astype(np.float32)

            # Run predictions with each available inference method
            if self.predictor:  # Treelite
                with self._get_stream(num_samples):
                    # Ensure data is in proper format for treelite
                    if self.use_soa_layout:
                        # Convert to Structure of Arrays format
                        features_soa = np.ascontiguousarray(features.T.reshape(-1))
                        dmat = treelite_runtime.DMatrix(
                            features_soa,
                            num_row=features.shape[0],
                            num_col=features.shape[1],
                            format="soa",
                        )
                    else:
                        dmat = treelite_runtime.DMatrix(features)
                    self.predictor.predict(dmat)
                self.logger.info("Warmed up Treelite predictor")

            if self.trt_engine:  # TensorRT
                self.predict_with_tensorrt(features)
                self.logger.info("Warmed up TensorRT engine")

            if self.booster:  # LightGBM
                self.predict(features)
                self.logger.info("Warmed up LightGBM booster")

        except Exception as e:
            self.logger.warning(
                f"Model warmup failed: {e}. This may affect initial inference latency."
            )

    def _extract_model_metadata(self):
        """Extract feature names, importance, and other metadata from the loaded model."""
        if not self.booster:
            return

        self.feature_names = self.booster.feature_name()
        self.input_shape = [self.max_batch_size, len(self.feature_names)]

        try:
            self.feature_importance = dict(
                zip(
                    self.feature_names,
                    self.booster.feature_importance(importance_type="gain").astype(
                        float
                    ),
                )
            )
        except Exception as e:
            self.logger.warning(f"Could not get feature importance: {e}")

        self.logger.info(f"Extracted metadata: {len(self.feature_names)} features.")

    def load_feature_stats(self, file_path: str):
        """
        Loads feature means and stds from a JSON file.

        Args:
            file_path: Path to the feature statistics JSON file
        """
        try:
            with open(file_path, "r") as f:
                stats = json.load(f)

            self.feature_means = np.array(stats["means"], dtype=np.float32)
            self.feature_stds = np.array(stats["stds"], dtype=np.float32)

            if len(self.feature_means) != len(self.feature_names) or len(
                self.feature_stds
            ) != len(self.feature_names):
                self.logger.error(
                    "Feature stats dimensions mismatch model features. Disabling normalization."
                )
                self.feature_means = None
                self.feature_stds = None
            else:
                self.logger.info(f"Loaded feature statistics from {file_path}")

                # If using channel-wise quantization, initialize per-feature scales and zero-points
                if self.use_channel_quantization and self.use_int8:
                    self.quant_scales = np.ones(
                        len(self.feature_names), dtype=np.float32
                    )
                    self.quant_zero_points = np.zeros(
                        len(self.feature_names), dtype=np.float32
                    )

        except Exception as e:
            self.logger.error(f"Error loading feature statistics from {file_path}: {e}")
            self.feature_means = None
            self.feature_stds = None

    def _apply_gh200_optimizations(self):
        """Apply GH200 specific LightGBM parameters and environment settings."""
        if not self.gh200_optimizations:
            return

        self.logger.info("Applying GH200-specific configurations")

        # Set specific parameters for GH200 hardware if LightGBM supports them
        try:
            if self.booster and hasattr(self.booster, "set_parameter"):
                # Use unified memory if available for better memory management
                self.booster.set_parameter(
                    "gpu_use_dp", False
                )  # Disable double precision

                # Set specific CUDA parameters if running on GH200
                if HAS_CUPY:
                    # Adjust max_bin based on available memory
                    total_mem_mb = cp.cuda.runtime.getDeviceProperties(
                        self.gpu_device_id
                    )["totalGlobalMem"] / (1024 * 1024)
                    if total_mem_mb > 40000:  # If more than 40GB memory
                        self.booster.set_parameter("max_bin", 512)

                    # Set optimal thread settings based on GH200 hardware
                    if self.compute_capability[0] >= 9:  # Hopper or later
                        self.booster.set_parameter("gpu_max_bin_per_block", 128)

                    self.logger.info(
                        f"Configured LightGBM for GH200 with {total_mem_mb:.2f}MB memory, "
                        f"compute capability: {self.compute_capability[0]}.{self.compute_capability[1]}"
                    )

            # For Treelite, apply specific optimizations if available
            if self.predictor and hasattr(self.predictor, "set_parameter"):
                if self.use_soa_layout:
                    self.predictor.set_parameter("layout", "soa")

        except Exception as e:
            self.logger.warning(f"Could not apply all GH200 optimizations: {e}")

    def _apply_int8_quantization(self):
        """
        Prepare for INT8 quantization by calculating optimal scale factors and zero points.
        Supports per-channel (feature-wise) quantization for better accuracy.
        """
        if not self.use_int8 or not self.use_gpu:
            self.logger.info("INT8 quantization skipped (not enabled or no GPU).")
            return

        # Use calibration data to determine optimal scale/zero_point
        try:
            # Get feature statistics from model metadata or compute from calibration dataset
            if hasattr(self, "calibration_data") and self.calibration_data is not None:
                # Use actual calibration data if available (preferred method)
                self.logger.info(
                    "Using provided calibration data for INT8 quantization"
                )

                # Calculate min/max from calibration data for each feature
                if self.use_channel_quantization:
                    # Calculate per-feature (channel) quantization params
                    all_data = np.vstack(self.calibration_data)
                    channel_mins = np.min(all_data, axis=0)
                    channel_maxs = np.max(all_data, axis=0)

                    # Calculate scale and zero point for each feature
                    self.quant_scales = (channel_maxs - channel_mins) / 255.0
                    # Avoid division by zero for features with no variation
                    self.quant_scales = np.maximum(self.quant_scales, 1e-6)
                    self.quant_zero_points = -128.0 - (channel_mins / self.quant_scales)

                    self.logger.info(
                        f"Applied per-feature quantization for {len(self.feature_names)} features"
                    )

                else:
                    # Global quantization (same scale for all features)
                    all_data = np.vstack(self.calibration_data).flatten()
                    data_min = float(np.min(all_data))
                    data_max = float(np.max(all_data))

                    # Calculate scale and zero point for symmetric quantization
                    self.quant_scales[0] = (data_max - data_min) / 255.0
                    self.quant_zero_points[0] = -128.0 - (
                        data_min / self.quant_scales[0]
                    )

                    self.logger.info(
                        f"Applied global quantization with range [{data_min:.4f}, {data_max:.4f}]"
                    )

            elif self.feature_means is not None and self.feature_stds is not None:
                # Fallback to feature statistics-based approach
                if self.use_channel_quantization:
                    # Use 3-sigma range for each feature
                    channel_mins = self.feature_means - 3 * self.feature_stds
                    channel_maxs = self.feature_means + 3 * self.feature_stds

                    # Calculate scale and zero point for each feature
                    self.quant_scales = (channel_maxs - channel_mins) / 255.0
                    # Avoid division by zero for features with no variation
                    self.quant_scales = np.maximum(self.quant_scales, 1e-6)
                    self.quant_zero_points = -128.0 - (channel_mins / self.quant_scales)

                    self.logger.info(
                        "Applied per-feature quantization based on feature statistics"
                    )

                else:
                    # Global quantization (same scale for all features)
                    data_min = float(np.min(self.feature_means - 3 * self.feature_stds))
                    data_max = float(np.max(self.feature_means + 3 * self.feature_stds))

                    # Calculate scale and zero point for symmetric quantization
                    self.quant_scales[0] = (data_max - data_min) / 255.0
                    self.quant_zero_points[0] = -128.0 - (
                        data_min / self.quant_scales[0]
                    )

                    self.logger.info(
                        f"Applied global quantization with range [{data_min:.4f}, {data_max:.4f}]"
                    )
            else:
                # Fallback to reasonable defaults if no statistics or calibration data available
                data_min = -5.0
                data_max = 5.0
                self.quant_scales[0] = (data_max - data_min) / 255.0
                self.quant_zero_points[0] = -128.0 - (data_min / self.quant_scales[0])

                self.logger.warning(
                    "Using default quantization range [-5,5]. Consider providing feature statistics "
                    "or calibration data for better accuracy."
                )

            # Round zero points for stable quantization
            if not self.use_channel_quantization:
                self.quant_zero_points[0] = round(self.quant_zero_points[0])
            else:
                self.quant_zero_points = np.round(self.quant_zero_points)

            # Log quantization parameters
            if self.use_channel_quantization:
                self.logger.info(
                    "INT8 quantization ready with per-feature scales and zero-points"
                )

                # Log sample of scales for diagnostic purposes
                sample_idx = min(5, len(self.quant_scales))
                scale_sample = self.quant_scales[:sample_idx]
                zero_sample = self.quant_zero_points[:sample_idx]
                self.logger.debug(
                    f"Sample scales: {scale_sample}, Sample zero-points: {zero_sample}"
                )
            else:
                self.logger.info(
                    f"INT8 quantization ready. Scale: {self.quant_scales[0]:.4f}, Zero-point: {self.quant_zero_points[0]:.1f}"
                )

            # Store calibration parameters for TensorRT
            if self.use_tensorrt:
                self.calibration_params = {
                    "quant_scales": self.quant_scales,
                    "quant_zero_points": self.quant_zero_points,
                    "channel_wise": self.use_channel_quantization,
                }

        except Exception as e:
            self.logger.warning(
                f"Error during quantization calibration: {e}. Using defaults.",
                exc_info=True,
            )
            # Fallback to default values
            self.quant_scales = np.array([0.0392], dtype=np.float32)  # ~(10/255)
            self.quant_zero_points = np.array([-128.0], dtype=np.float32)

    def _convert_to_treelite(self):
        """
        Convert LightGBM model to Treelite format for optimized inference.
        Treelite provides better performance than raw LightGBM for GBDT models.
        """
        if not self.use_treelite or not HAS_TREELITE:
            self.logger.info(
                "Treelite conversion skipped (not enabled or not available)"
            )
            return False

        if not self.booster:
            self.logger.warning("Cannot convert to Treelite: No booster loaded")
            return False

        try:
            self.logger.info("Converting LightGBM model to Treelite format")

            # Convert LightGBM to Treelite model
            self.treelite_model = treelite.Model.from_lightgbm(self.booster)

            # Save metadata for reference
            self.treelite_model_info = {
                "num_tree": self.treelite_model.num_tree,
                "num_feature": self.treelite_model.num_feature,
                "num_class": self.treelite_model.num_class,
            }

            # Create optimized predictor with desired options
            predictor_options = {
                "verbose": False,
                "threads": self.num_threads,
            }

            if self.use_gpu:
                predictor_options["device_type"] = "cuda"
                predictor_options["device_id"] = self.gpu_device_id
            else:
                predictor_options["device_type"] = "cpu"

            # Add special flags for GH200
            if self.gh200_optimizations:
                # For ARM64 Grace CPU
                predictor_options["native_lib_name"] = "treelite_runtime_aarch64"
                # Use vectorized code
                predictor_options["use_simd"] = True

            # Create predictor (compiles the model)
            self.logger.info("Compiling Treelite model for optimized inference")
            self.predictor = treelite_runtime.Predictor(
                self.treelite_model, **predictor_options
            )

            self.logger.info(
                f"Successfully converted to Treelite: {self.treelite_model.num_tree} trees, "
                f"{self.treelite_model.num_feature} features"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to convert LightGBM to Treelite: {e}", exc_info=True
            )
            self.use_treelite = False
            return False

    def _preprocess_batch_gpu(
        self, features_batch: np.ndarray, stream: Optional[Any]
    ) -> cp.ndarray:
        """
        Preprocesses a batch of features on GPU using CuPy and Numba kernel.
        Supports both standard Array-of-Structures (AoS) and Structure-of-Arrays (SoA) layouts.

        Args:
            features_batch: Input features in NumPy array
            stream: CUDA stream to use for processing

        Returns:
            Preprocessed features in CuPy array
        """
        if not HAS_CUPY or self.feature_means is None or self.feature_stds is None:
            return cp.asarray(features_batch, dtype=cp.float32)  # Ensure CuPy array

        # Get batch dimensions
        batch_size, num_features = features_batch.shape

        # Decide on memory layout - SoA is better for many operations
        use_soa_layout = self.use_soa_layout and HAS_NUMBA

        # Prep features array with appropriate layout
        if use_soa_layout:
            # Structure-of-Arrays: features are stored contiguously by feature, then by sample
            # This improves memory access patterns for many operations
            features_soa = np.ascontiguousarray(features_batch.T.reshape(-1))
        else:
            # Standard Array-of-Structures layout (rows are samples, columns are features)
            features_aos = np.ascontiguousarray(features_batch)

        # Transfer data to GPU with appropriate memory handling
        if self.use_pinned_memory:
            # Allocate pinned memory for host array for faster transfers
            if use_soa_layout:
                host_pinned = cp.cuda.alloc_pinned_memory(features_soa.nbytes)
                host_array = np.frombuffer(
                    host_pinned, dtype=features_soa.dtype
                ).reshape(features_soa.shape)
                np.copyto(host_array, features_soa)
                d_features = cp.asarray(host_array, stream=stream)
            else:
                host_pinned = cp.cuda.alloc_pinned_memory(features_aos.nbytes)
                host_array = np.frombuffer(
                    host_pinned, dtype=features_aos.dtype
                ).reshape(features_aos.shape)
                np.copyto(host_array, features_aos)
                d_features = cp.asarray(host_array, stream=stream)
        else:
            # Standard transfer
            if use_soa_layout:
                d_features = cp.asarray(features_soa, dtype=cp.float32, stream=stream)
            else:
                d_features = cp.asarray(features_aos, dtype=cp.float32, stream=stream)

        # Allocate output array with appropriate layout
        if use_soa_layout:
            d_output = cp.empty_like(d_features, dtype=cp.float32)
        else:
            d_output = cp.empty_like(d_features, dtype=cp.float32)

        # Transfer normalization parameters to GPU
        d_means = cp.asarray(self.feature_means, dtype=cp.float32)
        d_stds = cp.asarray(self.feature_stds, dtype=cp.float32)

        # Process with appropriate kernel
        if HAS_NUMBA:
            # Choose appropriate kernel based on layout
            if use_soa_layout:
                # For SoA layout: use specialized kernel that understands this layout
                # Calculate optimal thread block size
                threads_x = min(int32(self.warp_size), 32)
                threads_y = min(int32(self.max_threads_per_block // threads_x), 32)
                threads_per_block = (threads_x, threads_y)

                blocks_per_grid = (
                    int32(
                        (num_features + threads_per_block[0] - 1)
                        // threads_per_block[0]
                    ),
                    int32(
                        (batch_size + threads_per_block[1] - 1) // threads_per_block[1]
                    ),
                )

                self.logger.debug(
                    f"SoA preprocessing: {batch_size} samples, {num_features} features, "
                    f"block size {threads_per_block}, grid size {blocks_per_grid}"
                )

                # Launch kernel with stream context if provided
                if stream:
                    with stream:
                        # Convert arrays to Numba format
                        d_features_numba = cuda.as_cuda_array(d_features)
                        d_output_numba = cuda.as_cuda_array(d_output)
                        d_means_numba = cuda.as_cuda_array(d_means)
                        d_stds_numba = cuda.as_cuda_array(d_stds)

                        # Launch kernel
                        batch_preprocess_soa_kernel[blocks_per_grid, threads_per_block](
                            d_features_numba,
                            d_output_numba,
                            d_means_numba,
                            d_stds_numba,
                            batch_size,
                            num_features,
                        )
                else:
                    # Same but without stream context
                    d_features_numba = cuda.as_cuda_array(d_features)
                    d_output_numba = cuda.as_cuda_array(d_output)
                    d_means_numba = cuda.as_cuda_array(d_means)
                    d_stds_numba = cuda.as_cuda_array(d_stds)

                    batch_preprocess_soa_kernel[blocks_per_grid, threads_per_block](
                        d_features_numba,
                        d_output_numba,
                        d_means_numba,
                        d_stds_numba,
                        batch_size,
                        num_features,
                    )
            else:
                # For standard Array-of-Structures layout
                # Calculate optimal thread block size for 2D kernel
                if hasattr(self, "max_threads_per_block") and hasattr(
                    self, "warp_size"
                ):
                    # For 2D kernels, try to maximize occupancy while respecting hardware limits
                    warp_size = self.warp_size
                    threads_x = min(int32(32), self.max_threads_per_dim[0])
                    max_y = min(
                        int32(self.max_threads_per_block // threads_x),
                        self.max_threads_per_dim[1],
                    )
                    threads_y = (
                        max_y - (max_y % warp_size) if max_y >= warp_size else max_y
                    )
                    threads_per_block = (threads_x, threads_y)
                else:
                    threads_per_block = (16, 16)  # Default 2D block size

                blocks_per_grid_x = int32(
                    (batch_size + threads_per_block[0] - 1) // threads_per_block[0]
                )
                blocks_per_grid_y = int32(
                    (num_features + threads_per_block[1] - 1) // threads_per_block[1]
                )
                blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

                self.logger.debug(
                    f"AoS preprocessing: {batch_size} samples, {num_features} features, "
                    f"block size {threads_per_block}, grid size {blocks_per_grid}"
                )

                # Launch kernel with stream context if provided
                if stream:
                    with stream:
                        # Convert arrays to Numba format
                        d_features_numba = cuda.as_cuda_array(d_features)
                        d_output_numba = cuda.as_cuda_array(d_output)
                        d_means_numba = cuda.as_cuda_array(d_means)
                        d_stds_numba = cuda.as_cuda_array(d_stds)

                        # Launch kernel
                        batch_preprocess_kernel[blocks_per_grid, threads_per_block](
                            d_features_numba,
                            d_output_numba,
                            d_means_numba,
                            d_stds_numba,
                        )
                else:
                    # Same but without stream context
                    d_features_numba = cuda.as_cuda_array(d_features)
                    d_output_numba = cuda.as_cuda_array(d_output)
                    d_means_numba = cuda.as_cuda_array(d_means)
                    d_stds_numba = cuda.as_cuda_array(d_stds)

                    batch_preprocess_kernel[blocks_per_grid, threads_per_block](
                        d_features_numba, d_output_numba, d_means_numba, d_stds_numba
                    )
        else:
            # Fallback to CuPy element-wise operation if Numba not available
            if stream:
                with stream:
                    # For SoA layout, we need to reshape, normalize, and reshape back
                    if use_soa_layout:
                        # Reshape to 2D for element-wise operations
                        d_features_2d = d_features.reshape(num_features, batch_size)
                        # Broadcast means and stds properly
                        d_means_expanded = d_means.reshape(
                            -1, 1
                        )  # Shape (num_features, 1)
                        d_stds_expanded = d_stds.reshape(
                            -1, 1
                        )  # Shape (num_features, 1)
                        # Normalize features
                        d_output_2d = (d_features_2d - d_means_expanded) / (
                            d_stds_expanded + 1e-7
                        )
                        # Reshape back to 1D SoA layout
                        d_output = d_output_2d.reshape(-1)
                    else:
                        # Standard AoS layout normalization
                        d_output = (d_features - d_means) / (d_stds + 1e-7)
            else:
                # Same operations without stream context
                if use_soa_layout:
                    d_features_2d = d_features.reshape(num_features, batch_size)
                    d_means_expanded = d_means.reshape(-1, 1)
                    d_stds_expanded = d_stds.reshape(-1, 1)
                    d_output_2d = (d_features_2d - d_means_expanded) / (
                        d_stds_expanded + 1e-7
                    )
                    d_output = d_output_2d.reshape(-1)
                else:
                    d_output = (d_features - d_means) / (d_stds + 1e-7)

        # If we're using SoA layout but the output needs to be AoS (for LightGBM)
        # we need to convert it back
        if use_soa_layout and not self.use_soa_layout:
            # Reshape to 2D, transpose, and flatten to convert back to AoS
            d_output_2d = d_output.reshape(num_features, batch_size)
            d_output_aos = cp.ascontiguousarray(d_output_2d.T)
            return d_output_aos

        return d_output

    def _quantize_features_gpu(
        self, d_features: cp.ndarray, stream: Optional[Any]
    ) -> cp.ndarray:
        """
        Quantizes features on GPU using Numba kernel.
        Supports both global and per-channel quantization.

        Args:
            d_features: Input features in CuPy array
            stream: CUDA stream to use for processing

        Returns:
            Quantized features in INT8 format (CuPy array)
        """
        if not self.use_int8 or not HAS_NUMBA or not HAS_CUPY:
            return d_features  # Return as float32 CuPy array

        # Allocate output array for quantized features
        d_quantized_features = cp.empty(d_features.shape, dtype=cp.int8)

        # Create CuPy arrays for quantization parameters
        if self.use_channel_quantization:
            # For per-channel quantization, we need the full array of scales and zero-points
            d_scales = cp.asarray(self.quant_scales, dtype=cp.float32)
            d_zero_points = cp.asarray(self.quant_zero_points, dtype=cp.float32)
            # Determine channel dimension (1 for AoS layout, or calculate for SoA)
            if self.use_soa_layout:
                channel_dim = 1  # In SoA, we reshape inside the kernel
            else:
                channel_dim = d_features.shape[1]  # Number of features
        else:
            # For global quantization, we just need single values
            d_scales = cp.asarray([self.quant_scales[0]], dtype=cp.float32)
            d_zero_points = cp.asarray([self.quant_zero_points[0]], dtype=cp.float32)
            channel_dim = None

        # Optimize thread block size based on device properties
        if hasattr(self, "max_threads_per_block") and hasattr(self, "warp_size"):
            # For 1D kernels, use a multiple of warp size up to max_threads_per_block
            warp_size = self.warp_size
            # Find largest multiple of warp size that fits in max_threads_per_block
            threads_per_block = int32(
                self.max_threads_per_block - (self.max_threads_per_block % warp_size)
            )
            self.logger.debug(f"Using optimized 1D block size: {threads_per_block}")
        else:
            threads_per_block = int32(256)  # Default 1D block size

        # Calculate grid size
        blocks_per_grid = int32(
            (d_features.size + threads_per_block - 1) // threads_per_block
        )

        self.logger.debug(
            f"Quantize kernel: block size {threads_per_block}, grid size {blocks_per_grid}"
        )

        # Convert arrays to Numba compatible format
        d_features_numba = cuda.as_cuda_array(d_features)
        d_quantized_numba = cuda.as_cuda_array(d_quantized_features)
        d_scales_numba = cuda.as_cuda_array(d_scales)
        d_zero_points_numba = cuda.as_cuda_array(d_zero_points)

        # Execute the quantization with appropriate stream context
        if stream:
            with stream:
                int8_quantize_kernel[blocks_per_grid, threads_per_block](
                    d_features_numba,
                    d_quantized_numba,
                    d_scales_numba,
                    d_zero_points_numba,
                    channel_dim,
                )
        else:
            int8_quantize_kernel[blocks_per_grid, threads_per_block](
                d_features_numba,
                d_quantized_numba,
                d_scales_numba,
                d_zero_points_numba,
                channel_dim,
            )

        # Return the quantized features
        return d_quantized_features

    def _dequantize_results_gpu(
        self, d_quantized_results: cp.ndarray, stream: Optional[Any]
    ) -> cp.ndarray:
        """
        Dequantizes results on GPU using Numba kernel.
        Supports both global and per-channel dequantization.

        Args:
            d_quantized_results: Quantized data in INT8 format (CuPy array)
            stream: CUDA stream to use for processing

        Returns:
            Dequantized data in float32 format (CuPy array)
        """
        if not self.use_int8 or not HAS_NUMBA or not HAS_CUPY:
            # If not using INT8, results are already float. If no Numba/CuPy, this path shouldn't be hit.
            return d_quantized_results

        # Allocate output array for dequantized results
        d_dequantized_results = cp.empty(d_quantized_results.shape, dtype=cp.float32)

        # Create CuPy arrays for quantization parameters
        if self.use_channel_quantization:
            # For per-channel quantization, we need the full array of scales and zero-points
            d_scales = cp.asarray(self.quant_scales, dtype=cp.float32)
            d_zero_points = cp.asarray(self.quant_zero_points, dtype=cp.float32)
            # Determine channel dimension (1 for AoS layout, or calculate for SoA)
            if self.use_soa_layout:
                channel_dim = 1  # In SoA, we reshape inside the kernel
            else:
                channel_dim = (
                    d_quantized_results.shape[1]
                    if d_quantized_results.ndim > 1
                    else None
                )
        else:
            # For global quantization, we just need single values
            d_scales = cp.asarray([self.quant_scales[0]], dtype=cp.float32)
            d_zero_points = cp.asarray([self.quant_zero_points[0]], dtype=cp.float32)
            channel_dim = None

        # Optimize thread block size based on device properties
        if hasattr(self, "max_threads_per_block") and hasattr(self, "warp_size"):
            # For 1D kernels, use a multiple of warp size up to max_threads_per_block
            warp_size = self.warp_size
            # Find largest multiple of warp size that fits in max_threads_per_block
            threads_per_block = int32(
                self.max_threads_per_block - (self.max_threads_per_block % warp_size)
            )
            self.logger.debug(f"Using optimized 1D block size: {threads_per_block}")
        else:
            threads_per_block = int32(256)  # Default 1D block size

        # Calculate grid size
        blocks_per_grid = int32(
            (d_quantized_results.size + threads_per_block - 1) // threads_per_block
        )

        self.logger.debug(
            f"Dequantize kernel: block size {threads_per_block}, grid size {blocks_per_grid}"
        )

        # Convert arrays to Numba compatible format
        d_quantized_numba = cuda.as_cuda_array(d_quantized_results)
        d_dequantized_numba = cuda.as_cuda_array(d_dequantized_results)
        d_scales_numba = cuda.as_cuda_array(d_scales)
        d_zero_points_numba = cuda.as_cuda_array(d_zero_points)

        # Execute the dequantization with appropriate stream context
        if stream:
            with stream:
                int8_dequantize_kernel[blocks_per_grid, threads_per_block](
                    d_quantized_numba,
                    d_dequantized_numba,
                    d_scales_numba,
                    d_zero_points_numba,
                    channel_dim,
                )
        else:
            int8_dequantize_kernel[blocks_per_grid, threads_per_block](
                d_quantized_numba,
                d_dequantized_numba,
                d_scales_numba,
                d_zero_points_numba,
                channel_dim,
            )

        # Return the dequantized results
        return d_dequantized_results

    def _parse_lightgbm_trees(self, model_json: Dict) -> List[Dict]:
        """
        Parse LightGBM model JSON to extract tree structures.

        Args:
            model_json: LightGBM model in JSON format

        Returns:
            List of tree structures with essential info
        """
        trees = []
        for tree_info in model_json["tree_info"]:
            tree = {
                "tree_index": tree_info["tree_index"],
                "num_leaves": tree_info["num_leaves"],
                "nodes": tree_info["tree_structure"],
            }
            trees.append(tree)
        return trees

    def _collect_constants(self, node, thresholds, leaf_values, feature_indices):
        """
        Recursively collect constants from tree nodes.

        Args:
            node: Current tree node
            thresholds: List to store threshold values
            leaf_values: List to store leaf values
            feature_indices: List to store feature indices
        """
        if "leaf_value" in node:
            # This is a leaf node
            leaf_values.append(float(node["leaf_value"]))
            return

        # This is a decision node
        feature_indices.append(int(node["split_feature"]))
        thresholds.append(float(node["threshold"]))

        # Process child nodes
        if "left_child" in node:
            self._collect_constants(
                node["left_child"], thresholds, leaf_values, feature_indices
            )
        if "right_child" in node:
            self._collect_constants(
                node["right_child"], thresholds, leaf_values, feature_indices
            )

    def _build_tree_network(
        self,
        network,
        input_tensor,
        node,
        feature_indices_tensor,
        thresholds_tensor,
        leaf_values_tensor,
        node_index,
        prefix,
    ):
        """
        Recursively build TensorRT network for tree traversal.

        Args:
            network: TensorRT network
            input_tensor: Input tensor
            node: Current tree node
            feature_indices_tensor: Tensor of feature indices
            thresholds_tensor: Tensor of threshold values
            leaf_values_tensor: Tensor of leaf values
            node_index: Current node index
            prefix: Prefix for naming layers

        Returns:
            Output tensor for the current node
        """
        if "leaf_value" in node:
            # This is a leaf node, return its value
            leaf_index = node["leaf_index"]
            leaf_value = network.add_constant(
                (1, 1), np.array([[leaf_values_tensor[leaf_index]]], dtype=np.float32)
            )
            leaf_value.name = f"{prefix}_leaf_{leaf_index}"
            return leaf_value.get_output(0)

        # This is a decision node
        feature_index = node["split_feature"]
        threshold = node["threshold"]

        # Get the feature value using gather
        gather_indices = network.add_constant(
            (1, 1), np.array([[feature_index]], dtype=np.int32)
        )
        gather_indices.name = f"{prefix}_feature_idx_{node_index}"

        gather_layer = network.add_gather(input_tensor, gather_indices.get_output(0), 1)
        gather_layer.name = f"{prefix}_gather_{node_index}"
        feature_value = gather_layer.get_output(0)

        # Create threshold constant
        threshold_const = network.add_constant(
            (1, 1), np.array([[threshold]], dtype=np.float32)
        )
        threshold_const.name = f"{prefix}_threshold_{node_index}"

        # Compare feature value with threshold
        comparison = network.add_elementwise(
            feature_value, threshold_const.get_output(0), trt.ElementWiseOperation.LESS
        )
        comparison.name = f"{prefix}_compare_{node_index}"

        # Build left and right subtrees
        left_node_index = node_index * 2 + 1
        right_node_index = node_index * 2 + 2

        left_output = self._build_tree_network(
            network,
            input_tensor,
            node["left_child"],
            feature_indices_tensor,
            thresholds_tensor,
            leaf_values_tensor,
            left_node_index,
            f"{prefix}_left",
        )

        right_output = self._build_tree_network(
            network,
            input_tensor,
            node["right_child"],
            feature_indices_tensor,
            thresholds_tensor,
            leaf_values_tensor,
            right_node_index,
            f"{prefix}_right",
        )

        # Select left or right based on comparison
        select_layer = network.add_select(
            comparison.get_output(0), left_output, right_output
        )
        select_layer.name = f"{prefix}_select_{node_index}"

        return select_layer.get_output(0)

    def _convert_tree_to_tensorrt(self, network, input_tensor, tree, tree_idx):
        """
        Convert a single decision tree to TensorRT layers.

        Args:
            network: TensorRT network
            input_tensor: Input tensor
            tree: Tree structure
            tree_idx: Tree index

        Returns:
            Output tensor for the tree
        """
        # Start with the root node
        root_node = tree["nodes"]

        # Collect constants from the tree
        thresholds = []
        leaf_values = []
        feature_indices = []
        self._collect_constants(root_node, thresholds, leaf_values, feature_indices)

        # Build the tree network
        return self._build_tree_network(
            network,
            input_tensor,
            root_node,
            feature_indices,
            thresholds,
            leaf_values,
            0,
            f"tree_{tree_idx}",
        )

    def _combine_tree_outputs(self, network, tree_outputs, init_score, learning_rate):
        """
        Combine outputs from all trees according to GBDT algorithm.

        Args:
            network: TensorRT network
            tree_outputs: List of tree output tensors
            init_score: Initial score
            learning_rate: Learning rate

        Returns:
            Final output tensor
        """
        # Start with the base score
        base_score = network.add_constant(
            (1, 1), np.array([[init_score]], dtype=np.float32)
        )
        base_score.name = "base_score"
        current_sum = base_score.get_output(0)

        # Add all tree contributions
        for i, tree_output in enumerate(tree_outputs):
            sum_layer = network.add_elementwise(
                current_sum, tree_output, trt.ElementWiseOperation.SUM
            )
            sum_layer.name = f"tree_sum_{i}"
            current_sum = sum_layer.get_output(0)

        # Apply learning rate scaling
        if learning_rate != 1.0:
            lr_constant = network.add_constant(
                (1, 1), np.array([[learning_rate]], dtype=np.float32)
            )
            lr_constant.name = "learning_rate"

            scaled = network.add_elementwise(
                current_sum, lr_constant.get_output(0), trt.ElementWiseOperation.PROD
            )
            scaled.name = "scaled_output"
            return scaled.get_output(0)

        return current_sum

    class LightGBMCalibrator(trt.IInt8EntropyCalibrator2):
        """INT8 calibrator for LightGBM models."""

        def __init__(
            self, calibration_data, batch_size, cache_file="calibration.cache"
        ):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.calibration_data = calibration_data
            self.batch_size = batch_size
            self.cache_file = cache_file
            self.current_index = 0
            self.device_input = None

            # Allocate device memory for calibration data
            if len(calibration_data) > 0:
                self.device_input = cuda_driver.mem_alloc(
                    calibration_data[0].nbytes * batch_size
                )

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.current_index >= len(self.calibration_data):
                return None

            batch_end = min(
                self.current_index + self.batch_size, len(self.calibration_data)
            )
            batch = self.calibration_data[self.current_index : batch_end]
            self.current_index = batch_end

            if len(batch) == 0:
                return None

            # Copy calibration data to device
            batch_array = np.ascontiguousarray(np.vstack(batch), dtype=np.float32)
            cuda_driver.memcpy_htod(self.device_input, batch_array)

            return [int(self.device_input)]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)

    def _convert_to_tensorrt(self):
        """
        Convert LightGBM model directly to TensorRT for optimal GH200 performance.
        This enables true INT8 inference without ONNX as an intermediate step.
        """
        if not self.use_tensorrt or not HAS_TENSORRT or not self.booster:
            return False

        self.logger.info(
            "Converting LightGBM model directly to TensorRT for INT8 inference"
        )

        try:
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )

            # Define input tensor
            input_tensor = network.add_input(
                name="input",
                dtype=trt.float32,
                shape=(self.max_batch_size, len(self.feature_names)),
            )

            # Extract LightGBM model structure
            model_json = self.booster.dump_model()

            # Parse trees from model JSON
            trees = self._parse_lightgbm_trees(model_json)
            self.logger.info(f"Parsed {len(trees)} trees from LightGBM model")

            # Get initial score (bias)
            init_score = 0.0
            if "init_score" in model_json:
                init_score = float(model_json["init_score"])

            # Convert each tree to TensorRT layers
            tree_outputs = []
            for i, tree in enumerate(trees):
                self.logger.debug(
                    f"Converting tree {i} with {tree['num_leaves']} leaves"
                )
                tree_output = self._convert_tree_to_tensorrt(
                    network, input_tensor, tree, i
                )
                tree_outputs.append(tree_output)

            # Combine tree outputs according to boosting algorithm
            final_output = self._combine_tree_outputs(
                network, tree_outputs, init_score, self.learning_rate
            )

            # Apply sigmoid for binary classification if needed
            if self.objective == "binary":
                # Add sigmoid activation
                sigmoid_layer = network.add_activation(
                    final_output, trt.ActivationType.SIGMOID
                )
                sigmoid_layer.name = "sigmoid"
                final_output = sigmoid_layer.get_output(0)

            # Mark output tensor
            final_output.name = "output"
            network.mark_output(final_output)

            # Create optimization config
            config = builder.create_builder_config()
            config.max_workspace_size = self.trt_workspace_size

            # Enable INT8 precision if requested
            if self.use_int8:
                config.set_flag(trt.BuilderFlag.INT8)

                # For INT8 calibration, we should use actual calibration data
                # This comment serves as a reminder that in production, you should:
                # 1. Collect representative data from your production environment
                # 2. Use this data for calibration instead of synthetic data
                # 3. Store calibration data with the model for consistent results

                # Check if calibration data is provided
                if (
                    not hasattr(self, "calibration_data")
                    or self.calibration_data is None
                ):
                    self.logger.warning(
                        "No calibration data provided for INT8 precision. "
                        "In production, you should provide actual calibration data "
                        "for optimal INT8 quantization results."
                    )

                    # As a fallback only, generate synthetic data
                    if self.feature_means is not None and self.feature_stds is not None:
                        self.logger.info(
                            "Generating synthetic calibration data as a FALLBACK. "
                            "This is not recommended for production use."
                        )
                        num_samples = 1000
                        calibration_data = []
                        for _ in range(num_samples):
                            # Generate random data following feature distribution
                            sample = np.random.normal(
                                self.feature_means, self.feature_stds
                            ).astype(np.float32)
                            calibration_data.append(sample.reshape(1, -1))
                        self.calibration_data = calibration_data
                    else:
                        self.logger.warning(
                            "No feature statistics available for calibration data generation"
                        )

                # Create and set calibrator if calibration data is available
                if (
                    hasattr(self, "calibration_data")
                    and self.calibration_data is not None
                ):
                    calibrator = self.LightGBMCalibrator(
                        self.calibration_data,
                        min(len(self.calibration_data), self.max_batch_size),
                        f"{self.model_path}.calibration.cache",
                    )
                    config.int8_calibrator = calibrator
                    self.logger.info("Using calibration data for INT8 precision")
                else:
                    self.logger.warning(
                        "No calibration data provided for INT8 precision. "
                        "Consider using Treelite with Forest Inference Library (FIL) "
                        "as an alternative for GPU-accelerated tree model inference."
                    )

                self.logger.info("Enabled INT8 precision for TensorRT")

            # Build engine
            self.logger.info("Building TensorRT engine - this may take a while...")
            serialized_engine = builder.build_serialized_network(network, config)
            self.logger.info("TensorRT engine built successfully")

            # Create runtime and engine
            runtime = trt.Runtime(TRT_LOGGER)
            self.trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.trt_context = self.trt_engine.create_execution_context()

            # Prepare CUDA stream for inference - use existing context
            if not hasattr(self, "cuda_initialized") or not self.cuda_initialized:
                # Initialize CUDA if not already done
                cuda_driver.init()
                self.cuda_device = cuda_driver.Device(self.gpu_device_id)
                self.cuda_context = self.cuda_device.make_context()
                self.cuda_initialized = True
                self.logger.info("Initialized CUDA context for TensorRT")

            # Create stream in the current context
            self.trt_stream = cuda_driver.Stream()

            # Prepare bindings
            self.trt_bindings = []
            for i in range(self.trt_engine.num_bindings):
                binding_shape = self.trt_engine.get_binding_shape(i)
                binding_size = (
                    trt.volume(binding_shape)
                    * self.max_batch_size
                    * np.dtype(np.float32).itemsize
                )
                device_mem = cuda_driver.mem_alloc(binding_size)
                self.trt_bindings.append(int(device_mem))

            self.logger.info("Successfully converted LightGBM model to TensorRT")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to convert LightGBM to TensorRT: {e}", exc_info=True
            )
            self.use_tensorrt = False
            return False

    def _save_trt_engine_to_cache(self):
        """Save TensorRT engine to disk for faster loading next time."""
        if not self.trt_engine:
            self.logger.warning("No TensorRT engine to save")
            return False

        try:
            # Serialize the engine
            engine_data = self.trt_engine.serialize()

            # Save to file
            with open(self.trt_engine_path, "wb") as f:
                f.write(engine_data)

            self.logger.info(f"Saved TensorRT engine to {self.trt_engine_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save TensorRT engine: {e}", exc_info=True)
            return False

    def _load_trt_engine_from_cache(self):
        """Load TensorRT engine from disk to avoid rebuilding."""
        if not self.trt_engine_path.exists():
            self.logger.warning("No cached TensorRT engine found")
            return False

        try:
            # Create TensorRT runtime
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)

            # Load engine data
            with open(self.trt_engine_path, "rb") as f:
                engine_data = f.read()

            # Deserialize the engine
            self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
            self.trt_context = self.trt_engine.create_execution_context()

            # Create stream
            self.trt_stream = cuda_driver.Stream()

            # Prepare bindings
            self.trt_bindings = []
            for i in range(self.trt_engine.num_bindings):
                binding_shape = self.trt_engine.get_binding_shape(i)
                binding_size = (
                    trt.volume(binding_shape)
                    * self.max_batch_size
                    * np.dtype(np.float32).itemsize
                )
                device_mem = cuda_driver.mem_alloc(binding_size)
                self.trt_bindings.append(int(device_mem))

            self.logger.info(f"Loaded TensorRT engine from {self.trt_engine_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load TensorRT engine: {e}", exc_info=True)
            return False

    def predict_with_tensorrt(self, features_array: np.ndarray) -> np.ndarray:
        """
        Perform inference using TensorRT engine.
        When INT8 is enabled, TensorRT handles quantization internally based on calibration.

        Args:
            features_array: Input features in NumPy array

        Returns:
            Prediction results in NumPy array
        """
        if not self.use_tensorrt or not self.trt_engine or not self.trt_context:
            raise RuntimeError("TensorRT engine not initialized")

        start_time = time.perf_counter()

        # Reshape input if needed
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)

        # Check if we already have preprocessed features from the predict() method
        if (
            hasattr(self, "last_processed_features")
            and self.last_processed_features is not None
        ):
            # Use the already preprocessed features (normalized float32)
            self.logger.debug(
                "Using pre-processed float32 features for TensorRT inference"
            )

            # Convert CuPy array to numpy for TensorRT
            processed_features = cp.asnumpy(self.last_processed_features)

            # Clear the cached processed features to avoid stale data in future calls
            self.last_processed_features = None
        else:
            # If we don't have pre-processed features, do the preprocessing here
            # Ensure features_array is C-contiguous float32
            features_array = np.ascontiguousarray(features_array, dtype=np.float32)

            # Just normalize the features, don't quantize
            if self.feature_means is not None and self.feature_stds is not None:
                processed_features = (features_array - self.feature_means) / (
                    self.feature_stds + 1e-7
                )
            else:
                processed_features = features_array

            self.logger.debug("Normalized features for TensorRT without quantization")

        # Get input and output binding indices
        input_idx = self.trt_engine.get_binding_index("input")
        output_idx = self.trt_engine.get_binding_index("output")

        # Prepare output buffer
        output_shape = self.trt_context.get_binding_shape(output_idx)
        output_size = trt.volume(output_shape) * features_array.shape[0]
        output_buffer = np.empty(output_size, dtype=np.float32)

        # Copy input data to device
        cuda_driver.memcpy_htod_async(
            self.trt_bindings[input_idx], processed_features, self.trt_stream
        )

        # Execute inference
        self.trt_context.execute_async_v2(
            bindings=self.trt_bindings, stream_handle=self.trt_stream.handle
        )

        # Copy output data back to host
        cuda_driver.memcpy_dtoh_async(
            output_buffer, self.trt_bindings[output_idx], self.trt_stream
        )

        # Synchronize stream
        self.trt_stream.synchronize()

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Record metrics
        self._record_inference_metrics(features_array.shape[0], latency_ms)

        self.logger.debug(
            f"TensorRT inference time: {latency_ms:.3f} ms for batch size {features_array.shape[0]}"
        )

        # Reshape to (batch_size, 1) for consistency
        return output_buffer.reshape(-1, 1)

    def predict_with_treelite(self, features_array: np.ndarray) -> np.ndarray:
        """
        Perform inference using Treelite predictor.

        Args:
            features_array: Input features in NumPy array

        Returns:
            Prediction results in NumPy array
        """
        if not self.predictor:
            raise RuntimeError("Treelite predictor not initialized")

        start_time = time.perf_counter()

        # Reshape input if needed
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)

        # Convert to appropriate format
        if self.use_soa_layout:
            # Convert to Structure of Arrays layout
            features_soa = np.ascontiguousarray(features_array.transpose())
            # Create DMatrix with SoA layout
            dmat = treelite_runtime.DMatrix(
                features_soa.reshape(-1),
                num_row=features_array.shape[0],
                num_col=features_array.shape[1],
                format="soa",
            )
        else:
            # Use standard Array of Structures layout
            dmat = treelite_runtime.DMatrix(features_array)

        # Make prediction
        results = self.predictor.predict(dmat)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Record metrics
        self._record_inference_metrics(features_array.shape[0], latency_ms)

        self.logger.debug(
            f"Treelite inference time: {latency_ms:.3f} ms for batch size {features_array.shape[0]}"
        )

        # Reshape to (batch_size, 1) for consistency
        return results.reshape(-1, 1)

    def predict(self, features_array: np.ndarray) -> np.ndarray:
        """
        Make predictions using the most appropriate backend.

        Args:
            features_array: Input features in NumPy array

        Returns:
            Prediction results in NumPy array
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Validate input
        if features_array.ndim == 1:  # Single sample
            features_array = features_array.reshape(1, -1)

        if features_array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input feature count mismatch. Expected {len(self.feature_names)}, got {features_array.shape[1]}"
            )

        # Use the most appropriate inference method
        try:
            # First try Treelite if available (fastest)
            if self.predictor and self.use_treelite:
                return self.predict_with_treelite(features_array)

            # Then try TensorRT if available
            if self.trt_engine and self.use_tensorrt:
                # For TensorRT, we need to preprocess the features first
                if self.use_gpu:
                    # Get appropriate stream
                    with self._get_stream(features_array.shape[0]) as stream:
                        # Preprocess features
                        self.last_processed_features = self._preprocess_batch_gpu(
                            features_array, stream
                        )

                # Run TensorRT inference
                return self.predict_with_tensorrt(features_array)

            # Fallback to LightGBM
            return self._predict_with_lightgbm(features_array)

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}", exc_info=True)

            # Try fallback methods
            if (
                not (self.predictor and self.use_treelite)
                and self.trt_engine
                and self.use_tensorrt
            ):
                self.logger.info("Falling back to TensorRT inference")
                try:
                    if self.use_gpu:
                        with self._get_stream(features_array.shape[0]) as stream:
                            self.last_processed_features = self._preprocess_batch_gpu(
                                features_array, stream
                            )
                    return self.predict_with_tensorrt(features_array)
                except Exception as e2:
                    self.logger.error(f"TensorRT fallback failed: {e2}", exc_info=True)

            # Final fallback to LightGBM
            self.logger.info("Falling back to LightGBM inference")
            try:
                return self._predict_with_lightgbm(features_array)
            except Exception as e3:
                self.logger.error(f"All inference methods failed: {e3}", exc_info=True)
                raise RuntimeError("All prediction methods failed") from e3

    def _predict_with_lightgbm(self, features_array: np.ndarray) -> np.ndarray:
        """
        Make predictions using LightGBM booster.

        Args:
            features_array: Input features in NumPy array

        Returns:
            Prediction results in NumPy array
        """
        if not self.booster:
            raise RuntimeError("LightGBM booster not available")

        start_time = time.perf_counter()
        num_samples = features_array.shape[0]

        # Ensure features_array is C-contiguous float32 for LightGBM
        features_array = np.ascontiguousarray(features_array, dtype=np.float32)

        # GPU inference path
        if self.use_gpu and HAS_CUPY:
            # Use batch size to determine how many streams to use
            batch_size = num_samples

            # For large batches, split processing across multiple streams
            if batch_size > 1 and self.use_cuda_streams and len(self.cuda_streams) > 1:
                # Split the batch into chunks for parallel processing
                chunk_size = max(1, batch_size // len(self.cuda_streams))
                # No need to store chunks in a list since we process them immediately
                results = []

                # Process each chunk with a different stream
                for i in range(0, batch_size, chunk_size):
                    end_idx = min(i + chunk_size, batch_size)
                    chunk = features_array[i:end_idx]

                    # Get appropriate stream for this chunk
                    with self._get_stream(end_idx - i) as stream:
                        # Process this chunk with its own stream
                        d_chunk_processed = self._preprocess_batch_gpu(chunk, stream)

                        # Apply INT8 quantization if needed (for memory efficiency)
                        if self.use_int8:
                            d_quantized = self._quantize_features_gpu(
                                d_chunk_processed, stream
                            )
                            d_dequantized = self._dequantize_results_gpu(
                                d_quantized, stream
                            )
                            d_input = d_dequantized
                        else:
                            d_input = d_chunk_processed

                        # Convert to numpy for LightGBM
                        input_np = cp.asnumpy(d_input)

                        # Get predictions for this chunk
                        chunk_predictions = self.booster.predict(
                            input_np, num_iteration=self.booster.current_iteration()
                        )

                        # Store result for this chunk
                        results.append((chunk_predictions, i, end_idx))

                # Combine results in the correct order
                final_predictions = np.zeros((batch_size, 1), dtype=np.float32)
                for chunk_pred, start_idx, end_idx in results:
                    if chunk_pred.ndim == 1:
                        chunk_pred = chunk_pred.reshape(-1, 1)
                    final_predictions[start_idx:end_idx] = chunk_pred

                final_predictions_np = final_predictions
            else:
                # For small batches, use a single stream
                with self._get_stream(batch_size) as stream:
                    # Preprocess on GPU
                    d_processed_features = self._preprocess_batch_gpu(
                        features_array, stream
                    )

                    # Apply INT8 quantization if needed
                    if self.use_int8:
                        d_quantized_features = self._quantize_features_gpu(
                            d_processed_features, stream
                        )
                        d_dequantized = self._dequantize_results_gpu(
                            d_quantized_features, stream
                        )
                        d_input_for_lgbm = d_dequantized
                    else:
                        d_input_for_lgbm = d_processed_features

                    # Convert to numpy for LightGBM
                    input_for_lgbm_np = cp.asnumpy(d_input_for_lgbm)

                    # Make prediction
                    raw_predictions = self.booster.predict(
                        input_for_lgbm_np,
                        num_iteration=self.booster.current_iteration(),
                    )

                    # LightGBM outputs are already numpy arrays
                    final_predictions_np = raw_predictions
        else:
            # CPU inference path
            if self.feature_means is not None and self.feature_stds is not None:
                # Normalize features
                processed_features = (features_array - self.feature_means) / (
                    self.feature_stds + 1e-7
                )
            else:
                processed_features = features_array

            # Make prediction
            final_predictions_np = self.booster.predict(
                processed_features, num_iteration=self.booster.current_iteration()
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Record metrics
        self._record_inference_metrics(batch_size, latency_ms)

        self.logger.debug(
            f"LightGBM inference time: {latency_ms:.3f} ms for batch size {batch_size}"
        )

        # Reshape to (batch_size, 1) for consistency
        return final_predictions_np.reshape(-1, 1)

    def _record_inference_metrics(self, batch_size: int, latency_ms: float):
        """
        Record inference metrics for monitoring.

        Args:
            batch_size: Size of the batch
            latency_ms: Inference latency in milliseconds
        """
        with self.stats_lock:
            # Update inference stats
            self.inference_count += 1
            self.total_inference_time_ms += latency_ms

            # Update latency buffer for percentile calculations
            self.latency_buffer.append(latency_ms)

            # Trim buffer if it gets too large
            if len(self.latency_buffer) > self.max_latency_buffer_size:
                self.latency_buffer = self.latency_buffer[
                    -self.max_latency_buffer_size :
                ]

            # Recalculate percentiles if we have enough data
            if len(self.latency_buffer) >= 10:
                self.latency_p50 = np.percentile(self.latency_buffer, 50)
                self.latency_p99 = np.percentile(self.latency_buffer, 99)

        # Record metrics if enabled
        if self.metrics:
            self.metrics.record_histogram(
                "inference_latency_ms", latency_ms, tags={"batch_size": str(batch_size)}
            )
            self.metrics.increment_counter("inference_requests")
            self.metrics.increment_counter("samples_processed", batch_size)

            # Every 100 inferences, update summary stats
            if self.inference_count % 100 == 0:
                with self.stats_lock:
                    avg_latency = (
                        self.total_inference_time_ms / self.inference_count
                        if self.inference_count > 0
                        else 0
                    )
                    self.metrics.record_gauge("average_latency_ms", avg_latency)
                    self.metrics.record_gauge("p50_latency_ms", self.latency_p50)
                    self.metrics.record_gauge("p99_latency_ms", self.latency_p99)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"error": "Model not loaded"}

        info = {
            "model_path": self.model_path,
            "model_version": self.model_version,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "params": {},
            "use_gpu": self.use_gpu,
            "gh200_optimizations": self.gh200_optimizations,
            "use_int8": self.use_int8,
            "use_channel_quantization": self.use_channel_quantization,
            "use_soa_layout": self.use_soa_layout,
            "inference_engine": "unknown",
        }

        # Add engine-specific info
        if self.predictor:
            info["inference_engine"] = "Treelite"
            if hasattr(self, "treelite_model_info"):
                info.update(self.treelite_model_info)
        elif self.trt_engine:
            info["inference_engine"] = "TensorRT"
            info["use_tensorrt"] = True
        elif self.booster:
            info["inference_engine"] = "LightGBM"
            info["num_trees"] = self.booster.num_trees()
            info["current_iteration"] = self.booster.current_iteration()
            info["params"] = self.booster.params

        # Add performance stats
        with self.stats_lock:
            if self.inference_count > 0:
                info["inference_stats"] = {
                    "inference_count": self.inference_count,
                    "avg_latency_ms": self.total_inference_time_ms
                    / self.inference_count,
                    "p50_latency_ms": self.latency_p50,
                    "p99_latency_ms": self.latency_p99,
                }

        return info

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        return self.feature_importance

    def hot_swap_model(
        self, new_model_path: str, feature_stats_path: Optional[str] = None
    ) -> bool:
        """
        Hot-swap the model without interrupting service.
        This allows for model updates without restarting the application.

        Args:
            new_model_path: Path to the new model file
            feature_stats_path: Optional path to feature statistics

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(
            f"Hot-swapping model from {self.model_path} to {new_model_path}"
        )

        try:
            # Create a copy of the current model for rollback if needed
            old_booster = self.booster
            old_predictor = self.predictor
            old_trt_engine = self.trt_engine
            old_trt_context = self.trt_context
            old_trt_bindings = self.trt_bindings
            old_model_path = self.model_path
            old_model_version = self.model_version

            # Create temporary directories for the new model
            temp_dir = Path(f"/tmp/model_swap_{int(time.time())}")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Load the new model in a temporary instance
            temp_model = GH200GBDTModel(self.config)
            temp_model.load(new_model_path, feature_stats_path)

            # Verify the new model
            if not temp_model.is_loaded:
                raise ValueError("Failed to load new model")

            # Swap model components under lock to ensure thread safety
            with self.model_lock:
                self.model_path = new_model_path
                self.model_version = self.model_version + 1
                self.booster = temp_model.booster
                self.predictor = temp_model.predictor
                self.treelite_model = temp_model.treelite_model
                self.trt_engine = temp_model.trt_engine
                self.trt_context = temp_model.trt_context
                self.trt_bindings = temp_model.trt_bindings
                self.feature_names = temp_model.feature_names
                self.feature_importance = temp_model.feature_importance
                self.feature_means = temp_model.feature_means
                self.feature_stds = temp_model.feature_stds
                self.quant_scales = temp_model.quant_scales
                self.quant_zero_points = temp_model.quant_zero_points

                # Update metrics labels
                if self.metrics:
                    self.metrics.update_labels({"version": str(self.model_version)})

                self.logger.info(
                    f"Successfully hot-swapped to model {new_model_path} (v{self.model_version})"
                )

            # Clean up old resources in background
            def cleanup_old_resources():
                nonlocal \
                    old_booster, \
                    old_predictor, \
                    old_trt_engine, \
                    old_trt_context, \
                    old_trt_bindings, \
                    old_model_path, \
                    old_model_version
                try:
                    # Give time for any in-flight requests to complete
                    time.sleep(5)

                    # Clean up old resources more thoroughly
                    if old_booster is not None:
                        # For LightGBM, explicitly free model memory
                        if hasattr(old_booster, "free_dataset"):
                            old_booster.free_dataset()
                        # Release Python reference
                        del old_booster

                    # For Treelite predictor
                    if old_predictor is not None:
                        # Ensure any GPU memory is released first
                        if hasattr(old_predictor, "free_device_memory"):
                            old_predictor.free_device_memory()
                        del old_predictor

                    # For TensorRT engine
                    if old_trt_engine is not None:
                        # Clean up associated resources
                        if old_trt_context is not None:
                            del old_trt_context
                        # Release any CUDA memory allocations
                        if old_trt_bindings:
                            try:
                                for binding in old_trt_bindings:
                                    if binding != 0:  # Skip null pointers
                                        cuda_driver.mem_free(binding)
                            except Exception as binding_error:
                                self.logger.warning(
                                    f"Error freeing TensorRT bindings: {binding_error}"
                                )
                        # Finally delete the engine
                        del old_trt_engine

                    # Explicitly call garbage collection to ensure memory is released
                    import gc

                    gc.collect()

                    # Force CUDA garbage collection if running on GPU
                    if HAS_CUPY:
                        cp.get_default_memory_pool().free_all_blocks()

                    self.logger.info(
                        f"Cleaned up resources from old model {old_model_path} (v{old_model_version})"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error cleaning up old model resources: {e}", exc_info=True
                    )

            # Start cleanup in background thread
            threading.Thread(target=cleanup_old_resources, daemon=True).start()

            return True

        except Exception as e:
            self.logger.error(f"Failed to hot-swap model: {e}", exc_info=True)
            return False

    def initialize_components(self, config: Optional[Dict] = None) -> bool:
        """
        Initialize all components for the integrated trading system.

        Args:
            config: Configuration dictionary (uses model config if None)

        Returns:
            True if initialization was successful, False otherwise
        """
        if config is None:
            # Load HFT model specific settings from dedicated config file
            hft_config_path = (
                "/home/ubuntu/gh200-trading-system/config/hft_model_settings.json"
            )
            try:
                with open(hft_config_path, "r") as f:
                    config = json.load(f)
                self.logger.info(f"Loaded HFT model settings from {hft_config_path}")
            except Exception as e:
                self.logger.error(
                    f"Failed to load HFT model settings from {hft_config_path}: {e}"
                )
                # No fallback, raise error if config can't be loaded
                raise RuntimeError(
                    f"Could not load required configuration from {hft_config_path}"
                )

        try:
            # Initialize market data provider
            self.logger.info("Initializing Polygon Market Data Provider for HFT model")
            # Load polygon provider specific settings from dedicated config file
            polygon_config_path = "/home/ubuntu/gh200-trading-system/config/polygon_provider_settings.json"
            try:
                with open(polygon_config_path, "r") as f:
                    polygon_config = json.load(f)
                self.logger.info(
                    f"Loaded polygon provider settings from {polygon_config_path}"
                )
                self.market_data_provider = PolygonMarketDataProvider(polygon_config)
            except Exception as e:
                self.logger.error(
                    f"Failed to load polygon provider settings from {polygon_config_path}: {e}"
                )
                # Fall back to using the general config
                self.logger.warning(
                    "Falling back to general config for polygon provider"
                )
                self.market_data_provider = PolygonMarketDataProvider(config)

            # Initialize fast exit strategy
            self.logger.info("Initializing Fast Exit Strategy for HFT model")
            # Load fast exit strategy specific settings from dedicated config file
            exit_strategy_config_path = "/home/ubuntu/gh200-trading-system/config/fast_exit_strategy_settings.json"
            try:
                with open(exit_strategy_config_path, "r") as f:
                    exit_strategy_config = json.load(f)
                self.logger.info(
                    f"Loaded fast exit strategy settings from {exit_strategy_config_path}"
                )
                self.exit_strategy = FastExitStrategy(exit_strategy_config)
            except Exception as e:
                self.logger.error(
                    f"Failed to load fast exit strategy settings from {exit_strategy_config_path}: {e}"
                )
                # Fall back to using the general config
                self.logger.warning(
                    "Falling back to general config for fast exit strategy"
                )
                self.exit_strategy = FastExitStrategy(config)
            if not self.exit_strategy.initialize():
                self.logger.error("Failed to initialize Fast Exit Strategy")
                return False

            # Initialize active positions tracking
            self.active_positions = {}  # Dict[str, Signal]
            self.positions_lock = threading.Lock()

            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}", exc_info=True)
            return False

    def fetch_and_predict(self, symbols: List[str]) -> Dict[str, np.ndarray]:
        """
        Fetch market data and make predictions in a single optimized pipeline.

        Args:
            symbols: List of symbols to fetch data for

        Returns:
            Dictionary mapping symbols to prediction results
        """
        if not hasattr(self, "market_data_provider"):
            raise RuntimeError(
                "Components not initialized. Call initialize_components() first."
            )
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Use the new memory-efficient method to get model-ready data
        # This handles normalization, memory reuse, and optimal layout
        model_ready_data = self.market_data_provider.get_model_ready_data(symbols)

        # For multiple symbols, we can leverage parallel streams
        # by batching predictions instead of processing one symbol at a time
        if len(symbols) > 1 and self.use_cuda_streams and len(self.cuda_streams) > 1:
            # Prepare batch input
            all_features = []
            symbol_indices = {}
            current_idx = 0

            # Collect all features into a single batch
            for symbol, features in model_ready_data.items():
                # Handle both dictionary format (from SOA layout) and direct arrays
                if isinstance(features, dict) and "data" in features:
                    feature_array = features["data"]
                else:
                    feature_array = features

                if feature_array.ndim == 1:
                    feature_array = feature_array.reshape(1, -1)

                # Track which indices in the batch belong to which symbol
                symbol_indices[symbol] = (
                    current_idx,
                    current_idx + feature_array.shape[0],
                )
                current_idx += feature_array.shape[0]
                all_features.append(feature_array)

            # Combine into a single batch
            if all_features:
                # Check if we're using CuPy arrays
                if any(
                    isinstance(f, cp.ndarray) for f in all_features if f is not None
                ):
                    # Use CuPy for batch combination
                    batch_features = cp.vstack(
                        [
                            f if isinstance(f, cp.ndarray) else cp.asarray(f)
                            for f in all_features
                            if f is not None
                        ]
                    )
                else:
                    # Use NumPy for batch combination
                    batch_features = np.vstack(
                        [f for f in all_features if f is not None]
                    )

                # Make batch prediction using multiple streams
                batch_predictions = self.predict(batch_features)

                # Split results back to individual symbols
                predictions = {}
                for symbol, (start_idx, end_idx) in symbol_indices.items():
                    predictions[symbol] = batch_predictions[start_idx:end_idx]
            else:
                predictions = {}
        else:
            # Make predictions for each symbol individually
            predictions = {}
            for symbol, features in model_ready_data.items():
                try:
                    # Handle both dictionary format (from SOA layout) and direct arrays
                    if isinstance(features, dict) and "data" in features:
                        feature_array = features["data"]
                    else:
                        feature_array = features

                    # Reshape features to match model input if needed
                    if feature_array.ndim == 1:
                        feature_array = feature_array.reshape(1, -1)

                    # Make prediction
                    pred = self.predict(feature_array)
                    predictions[symbol] = pred
                except Exception as e:
                    self.logger.error(f"Error predicting for symbol {symbol}: {e}")

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        self.logger.info(
            f"Fetched data and made predictions for {len(predictions)} symbols in {total_time_ms:.2f}ms"
        )

        # Record metrics
        if self.metrics:
            self.metrics.record_histogram(
                "batch_prediction_time_ms",
                total_time_ms,
                tags={"symbols": str(len(symbols))},
            )
            self.metrics.increment_counter("batch_predictions")

        return predictions

    def process_trading_cycle(
        self, symbols: List[str]
    ) -> Tuple[List[Signal], List[Signal]]:
        """
        Process a complete trading cycle: fetch data, make predictions, generate signals, and manage exits.

        Args:
            symbols: List of symbols to process

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        if not hasattr(self, "market_data_provider") or not hasattr(
            self, "exit_strategy"
        ):
            raise RuntimeError(
                "Components not initialized. Call initialize_components() first."
            )

        start_time = time.perf_counter()

        # Fetch market data
        market_data = self.market_data_provider.fetch_and_process_market_data(symbols)

        # Generate entry signals from predictions
        entry_signals = self._generate_entry_signals(symbols, market_data)

        # Get active positions
        active_positions = self._get_active_positions()

        # Generate exit signals
        exit_signals = self.exit_strategy.optimize_exits(active_positions, market_data)

        # Execute exit trades
        if exit_signals:
            self.exit_strategy.execute_exit_trades(exit_signals)

            # Update active positions
            self._update_positions_after_exits(exit_signals)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        self.logger.info(
            f"Processed trading cycle for {len(symbols)} symbols in {total_time_ms:.2f}ms, "
            f"generated {len(entry_signals)} entry signals and {len(exit_signals)} exit signals"
        )

        # Record metrics
        if self.metrics:
            self.metrics.record_histogram("trading_cycle_time_ms", total_time_ms)
            self.metrics.increment_counter("trading_cycles")
            self.metrics.increment_counter("entry_signals", len(entry_signals))
            self.metrics.increment_counter("exit_signals", len(exit_signals))

        return entry_signals, exit_signals

    def _generate_entry_signals(
        self, symbols: List[str], market_data: Dict[str, Any]
    ) -> List[Signal]:
        """
        Generate entry signals based on model predictions.

        Args:
            symbols: List of symbols to generate signals for
            market_data: Market data for the symbols

        Returns:
            List of entry signals
        """
        # Fetch market data optimized for HFT model
        feature_arrays = {}
        for symbol in symbols:
            if symbol in market_data.get("symbol_data", {}):
                symbol_data = market_data["symbol_data"][symbol]
                features = np.array(
                    [
                        symbol_data.get("last_price", 0.0),
                        symbol_data.get("bid_price", 0.0),
                        symbol_data.get("ask_price", 0.0),
                        symbol_data.get("bid_ask_spread", 0.0),
                        symbol_data.get("high_price", 0.0),
                        symbol_data.get("low_price", 0.0),
                        symbol_data.get("volume", 0),
                        symbol_data.get("vwap", 0.0),
                        symbol_data.get("rsi_14", 50.0),
                        symbol_data.get("macd", 0.0),
                        symbol_data.get("bb_upper", 0.0),
                        symbol_data.get("bb_middle", 0.0),
                        symbol_data.get("bb_lower", 0.0),
                        symbol_data.get("volume_acceleration", 0.0),
                        symbol_data.get("price_change_5m", 0.0),
                        symbol_data.get("momentum_1m", 0.0),
                    ],
                    dtype=np.float32,
                )
                feature_arrays[symbol] = features

        # Make predictions for each symbol
        entry_signals = []
        for symbol, features in feature_arrays.items():
            try:
                # Reshape features to match model input if needed
                if features.ndim == 1:
                    features = features.reshape(1, -1)

                # Make prediction
                pred = self.predict(features)

                # Generate signal if prediction exceeds threshold
                if pred[0][0] > 0.7:  # Entry threshold
                    # Create entry signal
                    signal = Signal(
                        symbol=symbol,
                        type="BUY",  # Simplified to BUY for now
                        price=market_data["symbol_data"][symbol].get("last_price", 0.0),
                        position_size=10000.0,  # Fixed position size for now
                        stop_loss=market_data["symbol_data"][symbol].get(
                            "last_price", 0.0
                        )
                        * 0.98,  # 2% stop loss
                        take_profit=market_data["symbol_data"][symbol].get(
                            "last_price", 0.0
                        )
                        * 1.03,  # 3% take profit
                        confidence=float(pred[0][0]),
                        timestamp=market_data.get(
                            "timestamp", int(time.time() * 1_000_000_000)
                        ),
                    )

                    # Add to signals
                    entry_signals.append(signal)

                    # Update active positions
                    self._add_active_position(signal)
            except Exception as e:
                self.logger.error(f"Error generating entry signal for {symbol}: {e}")

        return entry_signals

    def _get_active_positions(self) -> List[Signal]:
        """
        Get active positions.

        Returns:
            List of active positions
        """
        with self.positions_lock:
            return list(self.active_positions.values())

    def _add_active_position(self, signal: Signal) -> None:
        """
        Add an active position.

        Args:
            signal: Entry signal
        """
        with self.positions_lock:
            self.active_positions[signal.symbol] = signal

    def _update_positions_after_exits(self, exit_signals: List[Signal]) -> None:
        """
        Update active positions after exits.

        Args:
            exit_signals: List of exit signals
        """
        with self.positions_lock:
            for signal in exit_signals:
                if signal.symbol in self.active_positions:
                    del self.active_positions[signal.symbol]

    def cleanup_all(self) -> None:
        """
        Clean up all resources including components.
        This method should be called when the model is no longer needed.
        """
        try:
            self._is_closing = True
            self.logger.info("Cleaning up all resources...")

            # Clean up CUDA resources
            if HAS_CUPY and self.use_gpu:
                try:
                    for stream in self.cuda_streams:
                        stream.synchronize()

                    # Clean up memory pools
                    if self.memory_pool:
                        self.memory_pool.free_all_blocks()
                    if self.pinned_memory_pool:
                        self.pinned_memory_pool.free_all_blocks()

                    # Clean up other CuPy resources
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

                    self.logger.info("CUDA memory resources cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up CUDA memory resources: {e}")

            # Clean up TensorRT resources
            if self.use_tensorrt and self.trt_engine:
                try:
                    del self.trt_context
                    del self.trt_engine
                    self.trt_context = None
                    self.trt_engine = None
                    self.logger.info("TensorRT resources cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up TensorRT resources: {e}")

            # Clean up Treelite resources
            if self.use_treelite and self.predictor:
                try:
                    del self.predictor
                    del self.treelite_model
                    self.predictor = None
                    self.treelite_model = None
                    self.logger.info("Treelite resources cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up Treelite resources: {e}")

            # Clean up market data provider if initialized
            if hasattr(self, "market_data_provider"):
                try:
                    self.market_data_provider.cleanup()
                    self.logger.info("Market data provider cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up market data provider: {e}")

            # Clean up exit strategy if initialized
            if hasattr(self, "exit_strategy"):
                try:
                    self.exit_strategy.cleanup()
                    self.logger.info("Exit strategy cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up exit strategy: {e}")

            # Clean up CUDA context if we initialized it explicitly
            if (
                hasattr(self, "cuda_initialized")
                and self.cuda_initialized
                and hasattr(self, "cuda_context")
            ):
                try:
                    self.cuda_context.pop()
                    self.cuda_context.detach()
                    self.cuda_initialized = False
                    self.logger.info("CUDA context cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up CUDA context: {e}")

            # Clean up LightGBM resources
            self.booster = None

            # Final log message
            self.logger.info("All resources cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
