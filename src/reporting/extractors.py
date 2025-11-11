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
