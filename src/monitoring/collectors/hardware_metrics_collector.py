"""
Hardware metrics collector for the system exporter

This module provides a collector for hardware metrics including GPU, temperature, and power consumption.
"""

import os
import time
import threading
import logging
import subprocess
from typing import Dict, Any


class HardwareMetricsCollector:
    """Collector for hardware metrics (GPU, temperature, power)"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hardware metrics collector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.collection_thread = None
        self.collection_interval = config.get("hardware_metrics", {}).get(
            "collection_interval", 10
        )  # seconds
        self.metrics = {}
        self.metrics_lock = threading.Lock()

        # Check for NVIDIA tools
        self.has_nvidia_smi = self._check_nvidia_smi()
        if not self.has_nvidia_smi:
            logging.warning("nvidia-smi not found, GPU metrics will be limited")

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def start(self):
        """Start the collector"""
        if self.running:
            logging.warning("Hardware metrics collector already running")
            return

        self.running = True

        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True
        )
        self.collection_thread.start()

        logging.info("Hardware metrics collector started")

    def stop(self):
        """Stop the collector"""
        if not self.running:
            logging.warning("Hardware metrics collector not running")
            return

        self.running = False

        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)

        logging.info("Hardware metrics collector stopped")

    def _collection_loop(self):
        """Collection loop for periodic collection"""
        while self.running:
            try:
                self._collect_metrics()
            except Exception as e:
                logging.error(f"Error collecting hardware metrics: {str(e)}")

            # Sleep until next collection
            time.sleep(self.collection_interval)

    def _collect_metrics(self):
        """Collect hardware metrics"""
        metrics = {
            "timestamp": int(time.time()),
            "gpu": self._collect_gpu_metrics(),
            "temperature": self._collect_temperature_metrics(),
            "power": self._collect_power_metrics(),
        }

        # Update metrics
        with self.metrics_lock:
            self.metrics = metrics

    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics using nvidia-smi"""
        if not self.has_nvidia_smi:
            return {"error": "nvidia-smi not available"}

        try:
            # Run nvidia-smi with JSON output
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # Parse CSV output
            gpu_metrics = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                values = [v.strip() for v in line.split(",")]
                if len(values) >= 10:
                    gpu_metrics.append(
                        {
                            "index": int(values[0]),
                            "name": values[1],
                            "temperature": float(values[2]),
                            "gpu_utilization": float(values[3]),
                            "memory_utilization": float(values[4]),
                            "memory_total": float(values[5]),
                            "memory_free": float(values[6]),
                            "memory_used": float(values[7]),
                            "power_draw": float(values[8]),
                            "power_limit": float(values[9]),
                        }
                    )

            return {"devices": gpu_metrics}

        except subprocess.SubprocessError as e:
            logging.error(f"Error running nvidia-smi: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logging.error(f"Error parsing nvidia-smi output: {str(e)}")
            return {"error": str(e)}

    def _collect_temperature_metrics(self) -> Dict[str, Any]:
        """Collect temperature metrics"""
        temperature_metrics = {}

        # Try to get CPU temperature on Linux
        try:
            if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = int(f.read().strip()) / 1000.0
                    temperature_metrics["cpu"] = temp
        except Exception as e:
            logging.debug(f"Error reading CPU temperature: {str(e)}")

        # Try to get fan speeds on Linux
        try:
            result = subprocess.run(
                ["sensors"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode == 0:
                # Parse fan speeds from sensors output
                fan_speeds = []
                for line in result.stdout.split("\n"):
                    if "fan" in line.lower() and "rpm" in line.lower():
                        parts = line.split(":")
                        if len(parts) >= 2:
                            fan_name = parts[0].strip()
                            fan_value = parts[1].strip().split()[0]
                            try:
                                fan_speeds.append(
                                    {"name": fan_name, "rpm": float(fan_value)}
                                )
                            except ValueError:
                                pass

                if fan_speeds:
                    temperature_metrics["fans"] = fan_speeds
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return temperature_metrics

    def _collect_power_metrics(self) -> Dict[str, Any]:
        """Collect power consumption metrics"""
        power_metrics = {}

        # Try to get power consumption on Linux
        try:
            # Check if RAPL (Running Average Power Limit) is available
            rapl_dir = "/sys/class/powercap/intel-rapl"
            if os.path.exists(rapl_dir):
                domains = []

                # Iterate through RAPL domains
                for domain in os.listdir(rapl_dir):
                    if domain.startswith("intel-rapl:"):
                        domain_path = os.path.join(rapl_dir, domain)

                        # Get domain name
                        name_path = os.path.join(domain_path, "name")
                        if os.path.exists(name_path):
                            with open(name_path, "r") as f:
                                name = f.read().strip()
                        else:
                            name = domain

                        # Get energy consumption
                        energy_path = os.path.join(domain_path, "energy_uj")
                        if os.path.exists(energy_path):
                            with open(energy_path, "r") as f:
                                energy_uj = int(f.read().strip())
                        else:
                            energy_uj = 0

                        domains.append({"name": name, "energy_uj": energy_uj})

                if domains:
                    power_metrics["rapl_domains"] = domains
        except Exception as e:
            logging.debug(f"Error reading power consumption: {str(e)}")

        return power_metrics

    def get_metrics(self):
        """Get the latest metrics"""
        with self.metrics_lock:
            return self.metrics.copy()
