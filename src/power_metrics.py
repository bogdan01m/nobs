"""Power and resource utilization metrics collector.

Logs system power, CPU utilization, RAM usage, and battery drain
during benchmark execution. Calculates p50 and p95 statistics at the end.
"""

import json
import logging
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import psutil

from src.device_info import get_device_info

# Setup logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("power_metrics")
logger.setLevel(logging.INFO)

# File handler
log_file = LOGS_DIR / "power_metrics.log"
file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class PowerMetrics:
    """Collects power and utilization metrics during benchmark execution."""

    def __init__(self, interval: float = 1.0):
        """
        Initialize power metrics collector.

        Args:
            interval: Sampling interval in seconds (default: 1.0)
        """
        self.interval = interval
        self.running = False
        self.thread: threading.Thread | None = None
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.timestamps: list[float] = []

        # Detect device type and platform
        device_info = get_device_info()
        self.device = device_info.get("device", "cpu")
        self.platform = device_info.get("platform", "N/A")

        # Check capabilities
        self.can_monitor_power = self._check_power_monitoring()
        self.can_monitor_battery = self._check_battery_monitoring()

        # Battery tracking
        self.battery_start: float | None = None
        self.battery_end: float | None = None

        logger.info(
            f"PowerMetrics initialized - Device: {self.device}, Platform: {self.platform}, "
            f"Power monitoring: {self.can_monitor_power}, Battery monitoring: {self.can_monitor_battery}"
        )

    def _check_power_monitoring(self) -> bool:
        """Check if power monitoring is available."""
        if self.device == "cuda":
            # Check for nvidia-smi
            try:
                subprocess.run(
                    ["nvidia-smi", "-L"],
                    capture_output=True,
                    check=True,
                    timeout=2,
                )
                return True
            except Exception:
                return False
        elif self.platform == "Darwin":
            # macOS - check for ioreg
            try:
                subprocess.run(
                    ["ioreg", "-rw0", "-c", "AppleSmartBattery"],
                    capture_output=True,
                    check=True,
                    timeout=2,
                )
                return True
            except Exception:
                return False
        return False

    def _check_battery_monitoring(self) -> bool:
        """Check if battery monitoring is available."""
        try:
            battery = psutil.sensors_battery()
            return battery is not None
        except Exception:
            return False

    def _get_battery_percent(self) -> float | None:
        """Get current battery percentage."""
        if not self.can_monitor_battery:
            return None
        try:
            battery = psutil.sensors_battery()
            return battery.percent if battery else None
        except Exception:
            return None

    def _sample_nvidia_power(self) -> dict[str, float]:
        """Sample NVIDIA GPU power, VRAM usage, utilization, and temperature."""
        metrics = {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                values = [v.strip() for v in lines[0].split(",")]
                if len(values) >= 5:
                    metrics["system_watts"] = float(values[0])
                    metrics["gpu_vram_used_mb"] = float(values[1])
                    metrics["gpu_vram_total_mb"] = float(values[2])
                    metrics["gpu_util_percent"] = float(values[3])
                    metrics["gpu_temp_celsius"] = float(values[4])
        except Exception:
            pass
        return metrics

    def _sample_macos_power(self) -> dict[str, float]:
        """Sample macOS system power via battery without sudo."""
        metrics = {}
        try:
            result = subprocess.run(
                ["ioreg", "-rw0", "-c", "AppleSmartBattery"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            # Parse InstantAmperage and Voltage from output
            amperage = None
            voltage = None
            for line in result.stdout.split("\n"):
                if '"InstantAmperage"' in line:
                    # Format: "InstantAmperage" = 18446744073709551064
                    parts = line.split("=")
                    if len(parts) == 2:
                        val = int(parts[1].strip())
                        # Handle unsigned int wrap-around (negative values)
                        if val > 2**63:
                            val = val - 2**64
                        amperage = val  # in mA
                elif '"Voltage"' in line:
                    # Format: "Voltage" = 11692
                    parts = line.split("=")
                    if len(parts) == 2:
                        voltage = int(parts[1].strip())  # in mV

            if amperage is not None and voltage is not None:
                # Power = Voltage * Current
                # Voltage is in mV, Amperage is in mA
                # Power in watts = (mV * mA) / 1000000
                power_watts = (voltage * amperage) / 1_000_000.0
                metrics["system_watts"] = abs(power_watts)  # Absolute value

        except Exception:
            pass
        return metrics

    def _sample_system(self) -> dict[str, float]:
        """Sample system-wide metrics (CPU, RAM)."""
        metrics = {}
        try:
            metrics["cpu_util_percent"] = psutil.cpu_percent(interval=None)
            metrics["ram_used_gb"] = psutil.virtual_memory().used / (1024**3)
        except Exception:
            pass
        return metrics

    def _sample_once(self):
        """Take a single sample of all available metrics."""
        timestamp = time.time()
        sample = {}

        # Power metrics
        if self.can_monitor_power:
            if self.device == "cuda":
                sample.update(self._sample_nvidia_power())
            elif self.platform == "Darwin":
                sample.update(self._sample_macos_power())

        # System metrics
        sample.update(self._sample_system())

        # Store metrics
        self.timestamps.append(timestamp)
        for key, value in sample.items():
            self.metrics[key].append(value)

    def _monitoring_loop(self):
        """Background thread loop for continuous monitoring."""
        # Prime CPU percent (first call returns 0.0)
        psutil.cpu_percent(interval=None)
        time.sleep(0.1)

        while self.running:
            start = time.time()
            self._sample_once()
            elapsed = time.time() - start
            sleep_time = max(0, self.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        """Start monitoring metrics in background thread."""
        if self.running:
            return

        self.running = True
        self.metrics.clear()
        self.timestamps.clear()

        # Record battery at start
        if self.can_monitor_battery:
            self.battery_start = self._get_battery_percent()
            logger.info(f"Monitoring started - Battery at start: {self.battery_start}%")
        else:
            logger.info("Monitoring started")

        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        """
        Stop monitoring and return aggregated results.

        Returns:
            Dictionary with p50, p95 statistics and battery drain
        """
        if not self.running:
            return {}

        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

        # Record battery at end
        if self.can_monitor_battery:
            self.battery_end = self._get_battery_percent()

        # Calculate statistics
        results = {}

        # p50 and p95 for each metric
        for key, values in self.metrics.items():
            if values:
                sorted_values = sorted(values)
                n = len(sorted_values)
                p50_idx = int(n * 0.50)
                p95_idx = int(n * 0.95)

                results[f"{key}_p50"] = round(sorted_values[p50_idx], 2)
                results[f"{key}_p95"] = round(sorted_values[p95_idx], 2)

        # Battery drain
        if self.battery_start is not None and self.battery_end is not None:
            results["battery_start_percent"] = round(self.battery_start, 1)
            results["battery_end_percent"] = round(self.battery_end, 1)
            results["battery_drain_percent"] = round(
                self.battery_start - self.battery_end, 1
            )

        # Metadata
        results["samples_collected"] = len(self.timestamps)
        results["monitoring_duration_seconds"] = (
            round(self.timestamps[-1] - self.timestamps[0], 2)
            if len(self.timestamps) >= 2
            else 0.0
        )

        # Log results
        logger.info(
            f"Monitoring stopped - Collected {results['samples_collected']} samples over {results['monitoring_duration_seconds']}s"
        )
        if self.battery_start is not None and self.battery_end is not None:
            logger.info(
                f"Battery drain: {results['battery_drain_percent']}% ({self.battery_start}% -> {self.battery_end}%)"
            )
        logger.info(f"Results: {json.dumps(results, indent=2)}")

        return results

    def reset(self):
        """Reset all metrics (for starting a new measurement)."""
        self.metrics.clear()
        self.timestamps.clear()
        self.battery_start = None
        self.battery_end = None


# Context manager for easy usage
class PowerMonitor:
    """Context manager for power monitoring during benchmarks."""

    def __init__(self, interval: float = 1.0):
        self.monitor = PowerMetrics(interval=interval)
        self.results: dict[str, Any] = {}

    def __enter__(self):
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.results = self.monitor.stop()


# Example usage
if __name__ == "__main__":
    print("Starting 10-second power monitoring test...")
    print(f"Device: {get_device_info().get('device')}")
    print(f"Platform: {get_device_info().get('platform')}")

    with PowerMonitor(interval=1.0) as pm:
        # Simulate some work
        for i in range(10):
            print(f"Working... {i+1}/10")
            time.sleep(1)

    # Results are stored in pm.results after __exit__
    print("\nResults:")
    print(json.dumps(pm.results, indent=2))
