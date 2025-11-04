"""Unified inference table generator for LLMs and VLMs."""

from typing import Any

from ..extractors import ModelMetricsExtractor
from ..formatters import format_time_with_std
from .base import BaseTableGenerator


class InferenceTableGenerator(BaseTableGenerator):
    """Unified generator for LLM and VLM inference tables."""

    def __init__(self, results: list[dict[str, Any]], task_type: str):
        """Initialize with results and task type.

        Args:
            results: List of benchmark result dictionaries
            task_type: Either "llms" or "vlms"
        """
        super().__init__(results)
        self.task_type = task_type

    def generate(self) -> str:
        """Generate inference table for the specified task type.

        Returns:
            Markdown table string or empty string if no data
        """
        if not self._has_inference_data():
            return ""

        backend_results = self._group_by_backend()

        lines = [self._get_title()]

        for backend in ["LM_STUDIO", "OLLAMA", "UNKNOWN"]:
            if backend_data := backend_results.get(backend):
                lines.extend(self._generate_backend_section(backend, backend_data))

        return "\n".join(lines) + "\n"

    def _get_title(self) -> str:
        """Get section title based on task type.

        Returns:
            Section title string
        """
        if self.task_type == "llms":
            return "#### LLM Inference (3 prompts from awesome-chatgpt-prompts)\n"
        else:
            return "#### VLM Inference (3 questions from Hallucination_COCO)\n"

    def _has_inference_data(self) -> bool:
        """Check if any result has inference data for this task type.

        Returns:
            True if at least one result contains the task
        """
        return any(
            any(t["task"] == self.task_type for t in result.get("tasks", []))
            for result in self.results
        )

    def _group_by_backend(self) -> dict[str, list[dict[str, Any]]]:
        """Group results by backend.

        Returns:
            Dict with backend keys and lists of device/task pairs
        """
        backend_results: dict[str, list] = {
            "LM_STUDIO": [],
            "OLLAMA": [],
            "UNKNOWN": [],
        }
        extractor = ModelMetricsExtractor()

        for result in self.results:
            device = result["device_info"]["host"]
            tasks = extractor.extract_inference_tasks(result, self.task_type)

            for task in tasks:
                if "model" in task:
                    backend = task.get("backend", "UNKNOWN")
                    backend_results[backend].append({"device": device, "task": task})

        return backend_results

    def _generate_backend_section(
        self, backend: str, data: list[dict[str, Any]]
    ) -> list[str]:
        """Generate table section for a specific backend.

        Args:
            backend: Backend name (LM_STUDIO, OLLAMA, UNKNOWN)
            data: List of device/task pairs for this backend

        Returns:
            List of markdown lines for this section
        """
        lines = []

        # Add backend header
        backend_display = backend.replace("_", " ")
        lines.append(f"\n**{backend_display}**\n")

        # Table header
        lines.extend(
            self._build_header(
                [
                    "Device",
                    "Model",
                    "E2E TPS",
                    "TTFT (s)",
                    "TG (s)",
                    "E2E Latency (s)",
                    "Input Tokens",
                    "Output Tokens",
                ]
            )
        )

        # Table rows
        for item in data:
            lines.append(self._format_row(item))

        return lines

    def _format_row(self, item: dict[str, Any]) -> str:
        """Format a single inference row.

        Args:
            item: Dict with device and task data

        Returns:
            Markdown table row string
        """
        device = item["device"]
        model_data = item["task"]["model"]

        # Get token counts from first run
        first_run = model_data["runs"][0] if model_data["runs"] else {}
        input_tokens = first_run.get("total_input_tokens", "-")
        output_tokens = first_run.get("total_output_tokens", "-")

        # Format E2E TPS with std (try new key first, fallback to old for compatibility)
        tps_median = model_data.get("final_50p_e2e_tps") or model_data.get(
            "final_50p_tokens_per_sec"
        )
        tps_std = model_data.get("final_std_e2e_tps") or model_data.get(
            "final_std_tokens_per_sec", 0
        )
        tps_str = format_time_with_std(tps_median, tps_std) if tps_median else "N/A"

        # Format TTFT with std
        ttft_median = model_data.get("final_50p_ttft_s")
        ttft_std = model_data.get("final_std_ttft_s", 0)
        ttft_str = format_time_with_std(ttft_median, ttft_std) if ttft_median else "N/A"

        # Format TG with std
        tg_median = model_data.get("final_50p_tg_s")
        tg_std = model_data.get("final_std_tg_s", 0)
        tg_str = format_time_with_std(tg_median, tg_std) if tg_median else "N/A"

        # Format E2E Latency with std (try new key first, fallback to old for compatibility)
        lat_median = model_data.get("final_50p_e2e_latency_s") or model_data.get(
            "final_50p_latency_s"
        )
        lat_std = model_data.get("final_std_e2e_latency_s") or model_data.get(
            "final_std_latency_s", 0
        )
        lat_str = format_time_with_std(lat_median, lat_std) if lat_median else "N/A"

        return (
            f"| {device} | {model_data['model_name']} | "
            f"{tps_str} | "
            f"{ttft_str} | "
            f"{tg_str} | "
            f"{lat_str} | "
            f"{input_tokens} | {output_tokens} |"
        )
