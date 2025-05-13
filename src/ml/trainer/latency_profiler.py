"""
Latency Profiler for ML Model Training and Inference

This module provides utilities for measuring and tracking latency metrics
for both training and inference phases of ML models.
"""

import time
import json
import numpy as np
import threading
from typing import Dict, Optional, Any
from pathlib import Path
from functools import wraps
from collections import defaultdict

from src.monitoring.log import logging as log


class LatencyProfiler:
    """
    Latency profiler for ML model training and inference.

    This class provides utilities for measuring and tracking latency metrics
    for both training and inference phases of ML models. It supports tracking
    of average latencies as well as percentile latencies (p50, p95, p99).
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the latency profiler.

        Args:
            name: Name of the component being profiled
            config: Configuration dictionary (optional)
        """
        self.name = name
        self.logger = log.setup_logger(f"latency_profiler_{name}")
        self.logger.info(f"Initializing latency profiler for {name}")

        # Initialize metrics storage
        self.metrics = defaultdict(list)
        self.metrics_lock = threading.Lock()

        # Configuration
        self.config = config or {}

        # Tracking state
        self.current_phase = None
        self.phase_start_time = None
        self.active_timers = {}

    def start_phase(self, phase_name: str) -> None:
        """
        Start timing a new phase.

        Args:
            phase_name: Name of the phase to start timing
        """
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        self.logger.debug(f"Started phase: {phase_name}")

    def end_phase(self) -> float:
        """
        End timing the current phase and record the elapsed time.

        Returns:
            Elapsed time in milliseconds
        """
        if self.phase_start_time is None:
            self.logger.warning("end_phase called without a corresponding start_phase")
            return 0.0

        elapsed_time = (time.time() - self.phase_start_time) * 1000  # Convert to ms

        with self.metrics_lock:
            self.metrics[f"{self.current_phase}_ms"].append(elapsed_time)

        self.logger.debug(
            f"Ended phase: {self.current_phase}, elapsed: {elapsed_time:.2f} ms"
        )

        self.phase_start_time = None
        self.current_phase = None

        return elapsed_time

    def time_function(self, func_name: str = None):
        """
        Decorator to time a function execution.

        Args:
            func_name: Optional name for the function (defaults to function.__name__)

        Returns:
            Decorated function
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                self.start_phase(name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_phase()

            return wrapper

        return decorator

    def start_timer(self, timer_name: str) -> None:
        """
        Start a named timer.

        Args:
            timer_name: Name of the timer to start
        """
        self.active_timers[timer_name] = time.time()
        self.logger.debug(f"Started timer: {timer_name}")

    def stop_timer(self, timer_name: str) -> float:
        """
        Stop a named timer and record the elapsed time.

        Args:
            timer_name: Name of the timer to stop

        Returns:
            Elapsed time in milliseconds
        """
        if timer_name not in self.active_timers:
            self.logger.warning(f"stop_timer called for unknown timer: {timer_name}")
            return 0.0

        elapsed_time = (
            time.time() - self.active_timers[timer_name]
        ) * 1000  # Convert to ms

        with self.metrics_lock:
            self.metrics[f"{timer_name}_ms"].append(elapsed_time)

        self.logger.debug(
            f"Stopped timer: {timer_name}, elapsed: {elapsed_time:.2f} ms"
        )

        del self.active_timers[timer_name]

        return elapsed_time

    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Record a custom metric value.

        Args:
            metric_name: Name of the metric
            value: Value to record
        """
        with self.metrics_lock:
            self.metrics[metric_name].append(value)

        self.logger.debug(f"Recorded metric: {metric_name}={value}")

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of all recorded metrics.

        Returns:
            Dictionary with metric summaries
        """
        summary = {}

        with self.metrics_lock:
            for metric_name, values in self.metrics.items():
                if not values:
                    continue

                values_array = np.array(values)

                # Calculate statistics
                summary[metric_name] = {
                    "count": len(values),
                    "mean": float(np.mean(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "std": float(np.std(values_array)),
                    "p50": float(np.percentile(values_array, 50)),
                    "p95": float(np.percentile(values_array, 95)),
                    "p99": float(np.percentile(values_array, 99)),
                }

        return summary

    def reset_metrics(self) -> None:
        """Reset all recorded metrics."""
        with self.metrics_lock:
            self.metrics.clear()

        self.logger.debug("Reset all metrics")

    def save_metrics(self, file_path: str) -> None:
        """
        Save metrics to a JSON file.

        Args:
            file_path: Path to save the metrics to
        """
        summary = self.get_metrics_summary()

        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Saved metrics to {file_path}")

    def log_metrics_summary(self) -> None:
        """Log a summary of all recorded metrics."""
        summary = self.get_metrics_summary()

        self.logger.info(f"Latency metrics summary for {self.name}:")

        for metric_name, stats in summary.items():
            self.logger.info(f"  {metric_name}:")
            self.logger.info(f"    count: {stats['count']}")
            self.logger.info(f"    mean: {stats['mean']:.2f}")
            self.logger.info(f"    p50: {stats['p50']:.2f}")
            self.logger.info(f"    p95: {stats['p95']:.2f}")
            self.logger.info(f"    p99: {stats['p99']:.2f}")
            self.logger.info(f"    min: {stats['min']:.2f}")
            self.logger.info(f"    max: {stats['max']:.2f}")

    def create_latency_profile(self) -> Dict[str, Any]:
        """
        Create a latency profile for model metadata.

        Returns:
            Dictionary with latency profile
        """
        summary = self.get_metrics_summary()

        # Extract training and inference metrics
        training_metrics = {}
        inference_metrics = {}

        for metric_name, stats in summary.items():
            if metric_name.startswith("train_") or metric_name.startswith("training_"):
                training_metrics[metric_name] = stats
            elif metric_name.startswith("infer_") or metric_name.startswith(
                "inference_"
            ):
                inference_metrics[metric_name] = stats

        # Create profile
        profile = {
            "training_latency": training_metrics,
            "inference_latency": inference_metrics,
            "timestamp": time.time(),
            "hardware_info": self._get_hardware_info(),
        }

        return profile

    def _get_hardware_info(self) -> Dict[str, str]:
        """
        Get hardware information.

        Returns:
            Dictionary with hardware information
        """
        # In a real implementation, this would detect actual hardware
        # For now, return placeholder values
        return {
            "cpu": "Unknown",
            "gpu": "NVIDIA GH200",
            "memory": "Unknown",
        }

    def __enter__(self):
        """Context manager entry."""
        self.start_phase("context_block")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_phase()
