"""Metric extraction logic from benchmark results."""

from typing import Any


class TaskTimingExtractor:
    """Extract timing data from benchmark results."""

    @staticmethod
    def extract_embeddings_time(result: dict[str, Any]) -> float | None:
        """Extract embeddings task time.

        Args:
            result: Benchmark result dictionary

        Returns:
            Time in seconds or None if not found
        """
        for task in result.get("tasks", []):
            if task.get("task") == "embeddings":
                return task.get("total_time_seconds")
        return None

    @staticmethod
    def extract_llm_times_by_backend(
        result: dict[str, Any],
    ) -> dict[str, float | None]:
        """Extract LLM task times grouped by backend.

        Args:
            result: Benchmark result dictionary

        Returns:
            Dict with backend keys (LM_STUDIO, OLLAMA) and times
        """
        times: dict[str, float | None] = {"LM_STUDIO": None, "OLLAMA": None}
        for task in result.get("tasks", []):
            if task.get("task") == "llms":
                backend = task.get("backend", "").upper()
                if backend in times:
                    times[backend] = task.get("total_time_seconds")
        return times

    @staticmethod
    def extract_vlm_times_by_backend(
        result: dict[str, Any],
    ) -> dict[str, float | None]:
        """Extract VLM task times grouped by backend.

        Args:
            result: Benchmark result dictionary

        Returns:
            Dict with backend keys (LM_STUDIO, OLLAMA) and times
        """
        times: dict[str, float | None] = {"LM_STUDIO": None, "OLLAMA": None}
        for task in result.get("tasks", []):
            if task.get("task") == "vlms":
                backend = task.get("backend", "").upper()
                if backend in times:
                    times[backend] = task.get("total_time_seconds")
        return times

    @staticmethod
    def calculate_total_time(result: dict[str, Any]) -> float:
        """Calculate total time across all tasks.

        Args:
            result: Benchmark result dictionary

        Returns:
            Total time in seconds
        """
        total = 0.0

        # Embeddings
        if emb_time := TaskTimingExtractor.extract_embeddings_time(result):
            total += emb_time

        # LLMs
        llm_times = TaskTimingExtractor.extract_llm_times_by_backend(result)
        for time in llm_times.values():
            if time:
                total += time

        # VLMs
        vlm_times = TaskTimingExtractor.extract_vlm_times_by_backend(result)
        for time in vlm_times.values():
            if time:
                total += time

        return total


class PowerMetricsExtractor:
    """Extract power consumption metrics from benchmark results."""

    @staticmethod
    def get_cpu_metrics(power: dict[str, Any]) -> tuple[Any, Any]:
        """Extract CPU utilization metrics.

        Args:
            power: Power metrics dictionary

        Returns:
            Tuple of (p50, p95) CPU utilization percentages
        """
        return (
            power.get("cpu_util_percent_p50", "N/A"),
            power.get("cpu_util_percent_p95", "N/A"),
        )

    @staticmethod
    def get_ram_metrics(power: dict[str, Any]) -> tuple[Any, Any]:
        """Extract RAM usage metrics.

        Args:
            power: Power metrics dictionary

        Returns:
            Tuple of (p50, p95) RAM usage in GB
        """
        return (
            power.get("ram_used_gb_p50", "N/A"),
            power.get("ram_used_gb_p95", "N/A"),
        )

    @staticmethod
    def get_gpu_utilization_metrics(power: dict[str, Any]) -> tuple[Any, Any]:
        """Extract GPU utilization metrics.

        Args:
            power: Power metrics dictionary

        Returns:
            Tuple of (p50, p95) GPU utilization percentages
        """
        return (
            power.get("gpu_util_percent_p50", "N/A"),
            power.get("gpu_util_percent_p95", "N/A"),
        )

    @staticmethod
    def get_gpu_temperature_metrics(power: dict[str, Any]) -> tuple[Any, Any]:
        """Extract GPU temperature metrics.

        Args:
            power: Power metrics dictionary

        Returns:
            Tuple of (p50, p95) GPU temperature in Celsius
        """
        return (
            power.get("gpu_temp_celsius_p50", "N/A"),
            power.get("gpu_temp_celsius_p95", "N/A"),
        )

    @staticmethod
    def get_battery_metrics(power: dict[str, Any]) -> tuple[Any, Any, Any]:
        """Extract battery metrics.

        Args:
            power: Power metrics dictionary

        Returns:
            Tuple of (start%, end%, delta%)
        """
        return (
            power.get("battery_start_percent", "N/A"),
            power.get("battery_end_percent", "N/A"),
            power.get("battery_drain_percent", "N/A"),
        )

    @staticmethod
    def get_gpu_power_metrics(power: dict[str, Any]) -> tuple[Any, Any]:
        """Extract GPU power consumption metrics.

        Args:
            power: Power metrics dictionary

        Returns:
            Tuple of (p50, p95) GPU power in watts
        """
        # Try dedicated GPU power first, fall back to system power if not available
        gpu_p50 = power.get("gpu_power_watts_p50")
        gpu_p95 = power.get("gpu_power_watts_p95")

        if gpu_p50 is None or gpu_p95 is None:
            # Fall back to gpu_watts for NVIDIA systems
            gpu_p50 = power.get("gpu_watts_p50", "N/A")
            gpu_p95 = power.get("gpu_watts_p95", "N/A")

        return (gpu_p50, gpu_p95)

    @staticmethod
    def get_cpu_power_metrics(power: dict[str, Any]) -> tuple[Any, Any]:
        """Extract CPU power consumption metrics.

        Args:
            power: Power metrics dictionary

        Returns:
            Tuple of (p50, p95) CPU power in watts
        """
        # Try dedicated CPU power first, fall back to system power if not available
        cpu_p50 = power.get("cpu_power_watts_p50")
        cpu_p95 = power.get("cpu_power_watts_p95")

        if cpu_p50 is None or cpu_p95 is None:
            # No fallback for CPU power - only show if explicitly measured
            cpu_p50 = "N/A"
            cpu_p95 = "N/A"

        return (cpu_p50, cpu_p95)

    @staticmethod
    def get_vram_metrics(power: dict[str, Any]) -> tuple[Any, Any]:
        """Extract GPU VRAM usage metrics.

        Args:
            power: Power metrics dictionary

        Returns:
            Tuple of (p50, p95) VRAM usage in GB
        """
        vram_mb_p50 = power.get("gpu_vram_used_mb_p50")
        vram_mb_p95 = power.get("gpu_vram_used_mb_p95")

        # Convert MB to GB
        if vram_mb_p50 is not None and isinstance(vram_mb_p50, (int, float)):
            vram_p50 = round(vram_mb_p50 / 1024, 2)
        else:
            vram_p50 = "N/A"

        if vram_mb_p95 is not None and isinstance(vram_mb_p95, (int, float)):
            vram_p95 = round(vram_mb_p95 / 1024, 2)
        else:
            vram_p95 = "N/A"

        return (vram_p50, vram_p95)

    @staticmethod
    def get_monitoring_duration(power: dict[str, Any]) -> float | None:
        """Extract monitoring duration in seconds.

        Args:
            power: Power metrics dictionary

        Returns:
            Duration in seconds or None if not available
        """
        duration = power.get("monitoring_duration_seconds")
        if duration is not None and isinstance(duration, (int, float)):
            return duration
        return None


class ModelMetricsExtractor:
    """Extract model-specific performance metrics."""

    @staticmethod
    def extract_embeddings_models(result: dict[str, Any]) -> dict[str, Any]:
        """Extract embeddings model metrics.

        Args:
            result: Benchmark result dictionary

        Returns:
            Dict of model names to their metrics
        """
        for task in result.get("tasks", []):
            if task.get("task") == "embeddings":
                return task.get("models", {})
        return {}

    @staticmethod
    def extract_inference_tasks(
        result: dict[str, Any], task_type: str
    ) -> list[dict[str, Any]]:
        """Extract all inference tasks of a given type (llms or vlms).

        Args:
            result: Benchmark result dictionary
            task_type: Either "llms" or "vlms"

        Returns:
            List of task dictionaries
        """
        return [t for t in result.get("tasks", []) if t.get("task") == task_type]


class SummaryMetricsExtractor:
    """Extract summary metrics for performance overview table."""

    @staticmethod
    def extract_embeddings_rps_p50(result: dict[str, Any]) -> float | None:
        """Extract embeddings RPS P50 metric.

        Args:
            result: Benchmark result dictionary

        Returns:
            RPS P50 value or None if not available
        """
        models = ModelMetricsExtractor.extract_embeddings_models(result)
        # Get first model's RPS (usually only one embeddings model per benchmark)
        for model_data in models.values():
            return model_data.get("final_mean_rps")
        return None

    @staticmethod
    def extract_llm_tps_p50_by_backend(
        result: dict[str, Any],
    ) -> dict[str, float | None]:
        """Extract LLM TPS P50 metric grouped by backend.

        Args:
            result: Benchmark result dictionary

        Returns:
            Dict with backend keys (LM_STUDIO, OLLAMA) and TPS P50 values
        """
        tps_by_backend: dict[str, float | None] = {"LM_STUDIO": None, "OLLAMA": None}
        llm_tasks = ModelMetricsExtractor.extract_inference_tasks(result, "llms")

        for task in llm_tasks:
            backend = task.get("backend", "").upper()
            if backend not in tps_by_backend:
                continue

            model = task.get("model", {})
            runs = model.get("runs", [])
            if runs:
                tps_values = [run.get("p50_tps") for run in runs if run.get("p50_tps")]
                if tps_values:
                    tps_by_backend[backend] = sum(tps_values) / len(tps_values)

        return tps_by_backend

    @staticmethod
    def extract_vlm_tps_p50_by_backend(
        result: dict[str, Any],
    ) -> dict[str, float | None]:
        """Extract VLM TPS P50 metric grouped by backend.

        Args:
            result: Benchmark result dictionary

        Returns:
            Dict with backend keys (LM_STUDIO, OLLAMA) and TPS P50 values
        """
        tps_by_backend: dict[str, float | None] = {"LM_STUDIO": None, "OLLAMA": None}
        vlm_tasks = ModelMetricsExtractor.extract_inference_tasks(result, "vlms")

        for task in vlm_tasks:
            backend = task.get("backend", "").upper()
            if backend not in tps_by_backend:
                continue

            model = task.get("model", {})
            runs = model.get("runs", [])
            if runs:
                tps_values = [run.get("p50_tps") for run in runs if run.get("p50_tps")]
                if tps_values:
                    tps_by_backend[backend] = sum(tps_values) / len(tps_values)

        return tps_by_backend

    @staticmethod
    def extract_gpu_watts_p50(result: dict[str, Any]) -> float | None:
        """Extract GPU power consumption P50 across all tasks.

        Args:
            result: Benchmark result dictionary

        Returns:
            GPU watts P50 value or None if not available
        """
        # Power metrics are at the root level, not per-task
        power = result.get("power_metrics", {})
        gpu_p50, _ = PowerMetricsExtractor.get_gpu_power_metrics(power)
        if gpu_p50 != "N/A" and isinstance(gpu_p50, (int, float)):
            return gpu_p50
        return None

    @staticmethod
    def extract_cpu_watts_p50(result: dict[str, Any]) -> float | None:
        """Extract CPU power consumption P50 across all tasks.

        Args:
            result: Benchmark result dictionary

        Returns:
            CPU watts P50 value or None if not available
        """
        # Power metrics are at the root level, not per-task
        power = result.get("power_metrics", {})
        cpu_p50, _ = PowerMetricsExtractor.get_cpu_power_metrics(power)
        if cpu_p50 != "N/A" and isinstance(cpu_p50, (int, float)):
            return cpu_p50
        return None
