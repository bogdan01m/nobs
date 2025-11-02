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

from src.system_info.device_info import get_device_info

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

    def __init__(self, interval: float = 1.0, use_sudo_powermetrics: bool = False):
        """
        Initialize power metrics collector.

        Args:
            interval: Sampling interval in seconds (default: 1.0)
            use_sudo_powermetrics: Enable sudo powermetrics on macOS for GPU/CPU power (default: False)
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

        # Powermetrics (macOS sudo)
        self.use_sudo_powermetrics = use_sudo_powermetrics and self.platform == "Darwin"
        self.powermetrics_process: subprocess.Popen | None = None
        self.powermetrics_log: Path | None = None

        # Don't log on init - defer until start() is called
        self._init_info = (
            f"Device: {self.device}, Platform: {self.platform}, "
            f"Power monitoring: {self.can_monitor_power}, Battery monitoring: {self.can_monitor_battery}, "
            f"Sudo powermetrics: {self.use_sudo_powermetrics}"
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

                # Negative amperage = discharging (power consumption)
                # Positive amperage = charging (skip power metric)
                if amperage < 0:
                    metrics["system_watts"] = abs(power_watts)
                    metrics["is_charging"] = 0.0
                else:
                    # On AC power, skip system_watts metric
                    metrics["is_charging"] = 1.0

        except Exception:
            pass
        return metrics

    def _sample_macos_gpu(self) -> dict[str, float]:
        """Sample macOS GPU metrics via ioreg (Apple Silicon)."""
        metrics = {}
        try:
            import plistlib

            result = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator", "-a"],
                capture_output=True,
                timeout=2,
                check=True,
            )

            # Parse plist output
            data = plistlib.loads(result.stdout)

            # Extract PerformanceStatistics from first accelerator
            for item in data:
                if "PerformanceStatistics" in item:
                    stats = item["PerformanceStatistics"]

                    # GPU utilization (only Device Utilization for simplicity)
                    if "Device Utilization %" in stats:
                        metrics["gpu_util_percent"] = float(
                            stats["Device Utilization %"]
                        )

                    # GPU memory (bytes to MB) - only "In use" for simplicity
                    if "In use system memory" in stats:
                        metrics["gpu_vram_used_mb"] = float(
                            stats["In use system memory"]
                        ) / (1024**2)

                # Extract total VRAM from IOAcceleratorMemoryInfo
                if "IOAcceleratorMemoryInfo" in item:
                    mem_info = item["IOAcceleratorMemoryInfo"]
                    if "Total" in mem_info:
                        metrics["gpu_vram_total_mb"] = float(mem_info["Total"]) / (
                            1024**2
                        )

                break

        except Exception:
            pass

        # Get GPU temperature via powermetrics (single sample, no sudo needed)
        try:
            temp_result = subprocess.run(
                [
                    "sudo",
                    "-n",
                    "powermetrics",
                    "--samplers",
                    "thermal",
                    "-i",
                    "1",
                    "-n",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            # Parse temperature from output (example: "GPU die temperature: 45.2 C")
            for line in temp_result.stdout.split("\n"):
                if "GPU die temperature:" in line:
                    temp_str = line.split(":")[1].strip().split()[0]
                    try:
                        metrics["gpu_temp_celsius"] = float(temp_str)
                    except ValueError:
                        pass
                    break
        except Exception:
            # Temperature is optional, don't fail if unavailable
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

    def _start_powermetrics(self, print_message: bool = True):
        """Start sudo powermetrics process in background (macOS only).

        Args:
            print_message: If True, print user-facing message before sudo prompt
        """
        if not self.use_sudo_powermetrics:
            return

        try:
            # Create log file for powermetrics output
            self.powermetrics_log = LOGS_DIR / f"powermetrics_{int(time.time())}.log"

            # Print user-facing message before sudo prompt (optional)
            if print_message:
                print("\n⚡ Starting power monitoring with sudo powermetrics...")
                print("   (sudo password required for GPU/CPU power metrics)\n")

            logger.info("Starting sudo powermetrics process...")

            # Start powermetrics in background
            self.powermetrics_process = subprocess.Popen(
                [
                    "sudo",
                    "powermetrics",
                    "--samplers",
                    "cpu_power,gpu_power",
                    "-i",
                    str(int(self.interval * 1000)),  # Convert to milliseconds
                    "-o",
                    str(self.powermetrics_log),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # Wait a bit to ensure it started
            time.sleep(1)

            # Check if process is still running
            if self.powermetrics_process.poll() is not None:
                # Process died, check stderr
                _, stderr = self.powermetrics_process.communicate(timeout=1)
                logger.error(f"powermetrics failed to start: {stderr.decode()}")
                self.use_sudo_powermetrics = False
            else:
                logger.info(
                    f"powermetrics started successfully (PID: {self.powermetrics_process.pid})"
                )

        except Exception as e:
            logger.error(f"Failed to start powermetrics: {e}")
            self.use_sudo_powermetrics = False

    def _stop_powermetrics(self):
        """Stop powermetrics process and parse results."""
        if not self.powermetrics_process:
            return

        try:
            # Terminate powermetrics
            self.powermetrics_process.terminate()
            self.powermetrics_process.wait(timeout=5)
            logger.info("powermetrics process stopped")

        except Exception as e:
            logger.error(f"Error stopping powermetrics: {e}")
            try:
                self.powermetrics_process.kill()
            except Exception:
                pass

    def _parse_powermetrics_log(self) -> dict[str, float]:
        """Parse powermetrics log and extract GPU power metrics."""
        if not self.powermetrics_log or not self.powermetrics_log.exists():
            return {}

        try:
            gpu_power_samples = []
            cpu_power_samples = []

            with open(self.powermetrics_log, "r") as f:
                content = f.read()

            # Parse GPU power (looking for "GPU Power" in mW)
            for line in content.split("\n"):
                # Example line: "GPU Power: 1234 mW"
                if "GPU Power:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        value_str = (
                            parts[1].strip().split()[0]
                        )  # Get first word (number)
                        try:
                            gpu_power_samples.append(float(value_str))
                        except ValueError:
                            pass

                # Example line: "CPU Power: 5678 mW"
                elif "CPU Power:" in line and "Package" not in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        value_str = parts[1].strip().split()[0]
                        try:
                            cpu_power_samples.append(float(value_str))
                        except ValueError:
                            pass

            # Calculate statistics (only report in watts, not milliwatts)
            results = {}
            if gpu_power_samples:
                sorted_gpu = sorted(gpu_power_samples)
                n = len(sorted_gpu)
                gpu_power_mw_p50 = sorted_gpu[int(n * 0.50)]
                gpu_power_mw_p95 = sorted_gpu[int(n * 0.95)]
                # Convert to watts
                results["gpu_power_watts_p50"] = round(gpu_power_mw_p50 / 1000, 2)
                results["gpu_power_watts_p95"] = round(gpu_power_mw_p95 / 1000, 2)

            if cpu_power_samples:
                sorted_cpu = sorted(cpu_power_samples)
                n = len(sorted_cpu)
                cpu_power_mw_p50 = sorted_cpu[int(n * 0.50)]
                cpu_power_mw_p95 = sorted_cpu[int(n * 0.95)]
                # Convert to watts
                results["cpu_power_watts_p50"] = round(cpu_power_mw_p50 / 1000, 2)
                results["cpu_power_watts_p95"] = round(cpu_power_mw_p95 / 1000, 2)

            logger.info(
                f"Parsed {len(gpu_power_samples)} GPU power samples, {len(cpu_power_samples)} CPU power samples"
            )

            return results

        except Exception as e:
            logger.error(f"Error parsing powermetrics log: {e}")
            return {}

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

        # GPU metrics (macOS Apple Silicon)
        if self.platform == "Darwin" and self.device == "mps":
            sample.update(self._sample_macos_gpu())

        # System metrics
        sample.update(self._sample_system())

        # Store metrics
        self.timestamps.append(timestamp)
        for key, value in sample.items():
            self.metrics[key].append(value)

        # Log the sample immediately
        if sample:
            sample_log = {
                "timestamp": round(timestamp, 3),
                **{k: round(v, 2) for k, v in sample.items()},
            }
            logger.info(f"Sample: {json.dumps(sample_log)}")

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

        # Log initialization info on first start
        logger.info(f"PowerMetrics initialized - {self._init_info}")

        self.running = True
        self.metrics.clear()
        self.timestamps.clear()

        # Start powermetrics if enabled (macOS sudo)
        if self.use_sudo_powermetrics:
            self._start_powermetrics(print_message=False)

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

        # Stop powermetrics if running
        if self.use_sudo_powermetrics:
            self._stop_powermetrics()

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

        # Parse powermetrics log if available
        if self.use_sudo_powermetrics:
            powermetrics_results = self._parse_powermetrics_log()
            results.update(powermetrics_results)

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

    def __init__(
        self,
        interval: float = 1.0,
        use_sudo_powermetrics: bool | None = None,
        ask_user: bool = True,
    ):
        """
        Initialize power monitor.

        Args:
            interval: Sampling interval in seconds
            use_sudo_powermetrics: Enable sudo powermetrics (if None and ask_user=True, will prompt)
            ask_user: If True and use_sudo_powermetrics is None, ask user interactively
        """
        # Determine if we should use sudo powermetrics
        if use_sudo_powermetrics is None and ask_user:
            # Detect platform first
            device_info = get_device_info()
            platform = device_info.get("platform", "")

            # Only ask on macOS
            if platform == "Darwin":
                print("\n⚡ Power Monitoring Setup")
                print("━" * 50)
                print("Do you want to collect detailed GPU/CPU power metrics?")
                print("This requires sudo access (you'll be prompted for password).")
                print()
                print("Without sudo: CPU/GPU utilization, RAM, battery drain")
                print("With sudo:    All above + detailed power consumption (watts)")
                print("━" * 50)

                while True:
                    response = (
                        input("Enable sudo powermetrics? [y/N]: ").strip().lower()
                    )
                    if response in ["y", "yes"]:
                        use_sudo_powermetrics = True
                        break
                    elif response in ["n", "no", ""]:
                        use_sudo_powermetrics = False
                        break
                    else:
                        print("Please enter 'y' or 'n'")

                print()  # Empty line for better formatting
            else:
                # Not macOS, disable sudo powermetrics
                use_sudo_powermetrics = False
        elif use_sudo_powermetrics is None:
            # ask_user=False, default to False
            use_sudo_powermetrics = False

        # If sudo powermetrics enabled, request sudo password NOW (before starting monitoring)
        # This prevents the password prompt from appearing during active monitoring
        if use_sudo_powermetrics:
            print("⚡ Requesting sudo access for powermetrics...")
            print("   (You may be prompted for your password now)\n")
            try:
                # Warm up sudo - this will cache credentials
                result = subprocess.run(
                    ["sudo", "-v"],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    print(
                        "⚠️  Sudo authentication failed. Continuing without sudo powermetrics.\n"
                    )
                    use_sudo_powermetrics = False
                else:
                    print("✓ Sudo access granted\n")
            except Exception as e:
                print(
                    f"⚠️  Failed to authenticate: {e}. Continuing without sudo powermetrics.\n"
                )
                use_sudo_powermetrics = False

        self.monitor = PowerMetrics(
            interval=interval, use_sudo_powermetrics=use_sudo_powermetrics
        )
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
    print()

    # PowerMonitor will ask user interactively about sudo powermetrics
    with PowerMonitor(interval=1.0) as pm:
        print("Monitoring active...")
        # Simulate some work
        for i in range(10):
            print(f"Working... {i+1}/10")
            time.sleep(1)

    # Results are stored in pm.results after __exit__
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(pm.results, indent=2))
