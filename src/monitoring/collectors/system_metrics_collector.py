"""
System metrics collector for the system exporter

This module provides a collector for system metrics including CPU, memory, disk, and network.
"""

import os
import time
import threading
import logging
import psutil
from typing import Dict, Any


class SystemMetricsCollector:
    """Collector for system metrics (CPU, memory, disk, network)"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the system metrics collector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.collection_thread = None
        self.collection_interval = config.get("system_metrics", {}).get(
            "collection_interval", 5
        )  # seconds
        self.metrics = {}
        self.metrics_lock = threading.Lock()

    def start(self):
        """Start the collector"""
        if self.running:
            logging.warning("System metrics collector already running")
            return

        self.running = True

        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True
        )
        self.collection_thread.start()

        logging.info("System metrics collector started")

    def stop(self):
        """Stop the collector"""
        if not self.running:
            logging.warning("System metrics collector not running")
            return

        self.running = False

        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)

        logging.info("System metrics collector stopped")

    def _collection_loop(self):
        """Collection loop for periodic collection"""
        while self.running:
            try:
                self._collect_metrics()
            except Exception as e:
                logging.error(f"Error collecting system metrics: {str(e)}")

            # Sleep until next collection
            time.sleep(self.collection_interval)

    def _collect_metrics(self):
        """Collect system metrics"""
        metrics = {
            "timestamp": int(time.time()),
            "cpu": self._collect_cpu_metrics(),
            "memory": self._collect_memory_metrics(),
            "disk": self._collect_disk_metrics(),
            "network": self._collect_network_metrics(),
            "process": self._collect_process_metrics(),
        }

        # Update metrics
        with self.metrics_lock:
            self.metrics = metrics

    def _collect_cpu_metrics(self):
        """Collect CPU metrics"""
        cpu_metrics = {
            "percent": psutil.cpu_percent(interval=1, percpu=True),
            "count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0],
            "frequency": psutil.cpu_freq()._asdict()
            if psutil.cpu_freq()
            else {"current": 0, "min": 0, "max": 0},
        }

        # Add CPU times
        cpu_times = psutil.cpu_times_percent()
        cpu_metrics["times"] = {
            "user": cpu_times.user,
            "system": cpu_times.system,
            "idle": cpu_times.idle,
            "iowait": cpu_times.iowait if hasattr(cpu_times, "iowait") else 0,
            "irq": cpu_times.irq if hasattr(cpu_times, "irq") else 0,
            "softirq": cpu_times.softirq if hasattr(cpu_times, "softirq") else 0,
            "steal": cpu_times.steal if hasattr(cpu_times, "steal") else 0,
            "guest": cpu_times.guest if hasattr(cpu_times, "guest") else 0,
        }

        return cpu_metrics

    def _collect_memory_metrics(self):
        """Collect memory metrics"""
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()

        memory_metrics = {
            "virtual": {
                "total": virtual_memory.total,
                "available": virtual_memory.available,
                "used": virtual_memory.used,
                "free": virtual_memory.free,
                "percent": virtual_memory.percent,
            },
            "swap": {
                "total": swap_memory.total,
                "used": swap_memory.used,
                "free": swap_memory.free,
                "percent": swap_memory.percent,
            },
        }

        return memory_metrics

    def _collect_disk_metrics(self):
        """Collect disk metrics"""
        disk_metrics = {}

        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_metrics[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                    "fstype": partition.fstype,
                    "device": partition.device,
                }
            except (PermissionError, FileNotFoundError):
                # Skip partitions that can't be accessed
                pass

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_metrics["io"] = {
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_time": disk_io.read_time,
                "write_time": disk_io.write_time,
            }

        return disk_metrics

    def _collect_network_metrics(self):
        """Collect network metrics"""
        network_metrics = {}

        # Network I/O
        net_io = psutil.net_io_counters(pernic=True)
        for interface, counters in net_io.items():
            network_metrics[interface] = {
                "bytes_sent": counters.bytes_sent,
                "bytes_recv": counters.bytes_recv,
                "packets_sent": counters.packets_sent,
                "packets_recv": counters.packets_recv,
                "errin": counters.errin,
                "errout": counters.errout,
                "dropin": counters.dropin,
                "dropout": counters.dropout,
            }

        # Network connections
        connections = psutil.net_connections()
        connection_stats = {
            "established": 0,
            "listen": 0,
            "time_wait": 0,
            "close_wait": 0,
            "total": len(connections),
        }

        for conn in connections:
            if conn.status == "ESTABLISHED":
                connection_stats["established"] += 1
            elif conn.status == "LISTEN":
                connection_stats["listen"] += 1
            elif conn.status == "TIME_WAIT":
                connection_stats["time_wait"] += 1
            elif conn.status == "CLOSE_WAIT":
                connection_stats["close_wait"] += 1

        network_metrics["connections"] = connection_stats

        return network_metrics

    def _collect_process_metrics(self):
        """Collect process metrics for the current process"""
        process = psutil.Process()

        process_metrics = {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "memory_info": process.memory_info()._asdict(),
            "num_threads": process.num_threads(),
            "create_time": process.create_time(),
            "status": process.status(),
        }

        return process_metrics

    def get_metrics(self):
        """Get the latest metrics"""
        with self.metrics_lock:
            return self.metrics.copy()
