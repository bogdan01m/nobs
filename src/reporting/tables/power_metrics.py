"""Power metrics table generator."""

from typing import Any

from ..extractors import PowerMetricsExtractor
from ..formatters import format_battery_info, format_percentile_pair
from .base import BaseTableGenerator


class PowerMetricsTableGenerator(BaseTableGenerator):
    """Generate power metrics table for all devices."""

    def generate(self) -> str:
        """Generate power metrics table.

        Returns:
            Markdown table string or empty string if no power data
        """
        if not self._has_power_data():
            return ""

        lines = ["### ⚡ Power Metrics\n"]
        lines.extend(
            self._build_header(
                [
                    "Device",
                    "CPU Usage (p50/p95)",
                    "RAM Used GB (p50/p95)",
                    "VRAM Used GB (p50/p95)",
                    "GPU Usage (p50/p95)",
                    "GPU Temp (p50/p95)",
                    "Battery (start/end/Δ)",
                    "GPU Power (p50/p95)",
                    "CPU Power (p50/p95)",
                ]
            )
        )

        for result in self.results:
            if power := result.get("power_metrics"):
                device = result["device_info"]["host"]
                lines.append(self._format_row(device, power))

        lines.append("")
        lines.append("!!! Note\n")
        lines.append(
            "    For devices with unified memory (e.g. Apple Silicon), **VRAM usage** represents the portion of shared RAM allocated to the GPU — "
            "it does not indicate a separate dedicated memory pool as on discrete GPUs.\n"
        )

        return "\n".join(lines) + "\n"

    def _has_power_data(self) -> bool:
        """Check if any result has power metrics.

        Returns:
            True if at least one result contains power_metrics
        """
        return any("power_metrics" in r for r in self.results)

    def _format_row(self, device: str, power: dict[str, Any]) -> str:
        """Format a single power metrics row.

        Args:
            device: Device name/host
            power: Power metrics dictionary

        Returns:
            Markdown table row string
        """
        extractor = PowerMetricsExtractor()

        # Extract all metrics
        cpu_p50, cpu_p95 = extractor.get_cpu_metrics(power)
        ram_p50, ram_p95 = extractor.get_ram_metrics(power)
        gpu_util_p50, gpu_util_p95 = extractor.get_gpu_utilization_metrics(power)
        gpu_temp_p50, gpu_temp_p95 = extractor.get_gpu_temperature_metrics(power)
        vram_p50, vram_p95 = extractor.get_vram_metrics(power)
        battery_start, battery_end, battery_delta = extractor.get_battery_metrics(power)
        gpu_power_p50, gpu_power_p95 = extractor.get_gpu_power_metrics(power)
        cpu_power_p50, cpu_power_p95 = extractor.get_cpu_power_metrics(power)

        # Format metrics
        cpu_usage = format_percentile_pair(cpu_p50, cpu_p95, "%")
        ram_usage = format_percentile_pair(ram_p50, ram_p95, "GB")
        vram_usage = format_percentile_pair(vram_p50, vram_p95, "GB")
        gpu_usage = format_percentile_pair(gpu_util_p50, gpu_util_p95, "%")
        gpu_temp = format_percentile_pair(gpu_temp_p50, gpu_temp_p95, "°C")
        battery_info = format_battery_info(battery_start, battery_end, battery_delta)
        gpu_power = format_percentile_pair(gpu_power_p50, gpu_power_p95, "W")
        cpu_power = format_percentile_pair(cpu_power_p50, cpu_power_p95, "W")

        return (
            f"| {device} | {cpu_usage} | {ram_usage} | {vram_usage} | {gpu_usage} | "
            f"{gpu_temp} | {battery_info} | {gpu_power} | {cpu_power} |"
        )
