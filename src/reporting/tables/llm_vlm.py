"""Unified inference table generator for LLMs and VLMs."""

from typing import Any

from ..extractors import ModelMetricsExtractor, SummaryMetricsExtractor
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
            return "#### LLM Inference (10 prompts from awesome-chatgpt-prompts)\n"
        else:
            return "#### VLM Inference (10 questions from Hallucination_COCO)\n"

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
                    "TPS P50",
                    "TPS P95",
                    "TTFT P50 (s)",
                    "TTFT P95 (s)",
                    "TG P50 (s)",
                    "TG P95 (s)",
                    "Latency P50 (s)",
                    "Latency P95 (s)",
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

        # Helper function to format value with std
        def fmt_with_std(val, std_val):
            if isinstance(val, (int, float)) and isinstance(std_val, (int, float)):
                return f"{val:.2f} ± {std_val:.2f}"
            elif isinstance(val, (int, float)):
                return f"{val:.2f}"
            return "-"

        # Get P50 and P95 with std for TPS
        tps_p50 = model_data.get("final_p50_tps")
        tps_p50_std = model_data.get("final_p50_tps_std")
        tps_p95 = model_data.get("final_p95_tps")
        tps_p95_std = model_data.get("final_p95_tps_std")

        # Get P50 and P95 with std for TTFT
        ttft_p50 = model_data.get("final_p50_ttft_s")
        ttft_p50_std = model_data.get("final_p50_ttft_std_s")
        ttft_p95 = model_data.get("final_p95_ttft_s")
        ttft_p95_std = model_data.get("final_p95_ttft_std_s")

        # Get P50 and P95 with std for TG
        tg_p50 = model_data.get("final_p50_tg_s")
        tg_p50_std = model_data.get("final_p50_tg_std_s")
        tg_p95 = model_data.get("final_p95_tg_s")
        tg_p95_std = model_data.get("final_p95_tg_std_s")

        # Get P50 and P95 with std for E2E Latency
        lat_p50 = model_data.get("final_p50_e2e_latency_s")
        lat_p50_std = model_data.get("final_p50_e2e_latency_std_s")
        lat_p95 = model_data.get("final_p95_e2e_latency_s")
        lat_p95_std = model_data.get("final_p95_e2e_latency_std_s")

        return (
            f"| {device} | {model_data['model_name']} | "
            f"{fmt_with_std(tps_p50, tps_p50_std)} | {fmt_with_std(tps_p95, tps_p95_std)} | "
            f"{fmt_with_std(ttft_p50, ttft_p50_std)} | {fmt_with_std(ttft_p95, ttft_p95_std)} | "
            f"{fmt_with_std(tg_p50, tg_p50_std)} | {fmt_with_std(tg_p95, tg_p95_std)} | "
            f"{fmt_with_std(lat_p50, lat_p50_std)} | {fmt_with_std(lat_p95, lat_p95_std)} | "
            f"{input_tokens} | {output_tokens} |"
        )


class InferenceEfficiencyTableGenerator(BaseTableGenerator):
    """Generate inference performance per watt table for LLMs and VLMs."""

    def __init__(self, results: list[dict[str, Any]], task_type: str):
        """Initialize with results and task type.

        Args:
            results: List of benchmark result dictionaries
            task_type: Either "llms" or "vlms"
        """
        super().__init__(results)
        self.task_type = task_type

    def generate(self) -> str:
        """Generate inference efficiency table.

        Returns:
            Markdown table string or empty string if no data
        """
        if not self._has_data():
            return ""

        backend_results = self._extract_metrics_by_backend()
        if not any(backend_results.values()):
            return ""

        lines = [self._get_title()]

        for backend in ["LM_STUDIO", "OLLAMA"]:
            if metrics := backend_results.get(backend):
                lines.extend(self._generate_backend_section(backend, metrics))

        return "\n".join(lines) + "\n"

    def _get_title(self) -> str:
        """Get section title based on task type.

        Returns:
            Section title string
        """
        task_name = "LLM" if self.task_type == "llms" else "VLM"
        return (
            f"\n#### {task_name} Performance per Watt (GPU)\n"
            "_Higher values indicate better performance per watt._\n"
        )

    def _extract_metrics_by_backend(self) -> dict[str, list[dict[str, Any]]]:
        """Extract efficiency metrics grouped by backend.

        Returns:
            Dict with backend keys and lists of efficiency metrics
        """
        backend_results: dict[str, list] = {
            "LM_STUDIO": [],
            "OLLAMA": [],
        }
        extractor = SummaryMetricsExtractor()

        for result in self.results:
            device = result["device_info"]["host"]
            gpu_watts = extractor.extract_gpu_watts_p50(result)

            if not gpu_watts:
                continue

            # Extract TPS for the specific backend and task type
            if self.task_type == "llms":
                tps_by_backend = extractor.extract_llm_tps_p50_by_backend(result)
            else:
                tps_by_backend = extractor.extract_vlm_tps_p50_by_backend(result)

            for backend in ["LM_STUDIO", "OLLAMA"]:
                tps = tps_by_backend.get(backend)
                if tps:
                    efficiency = tps / gpu_watts
                    backend_results[backend].append(
                        {
                            "device": device,
                            "tps": tps,
                            "gpu_watts": gpu_watts,
                            "efficiency": efficiency,
                        }
                    )

        return backend_results

    def _generate_backend_section(
        self, backend: str, metrics: list[dict[str, Any]]
    ) -> list[str]:
        """Generate table section for a specific backend.

        Args:
            backend: Backend name (LM_STUDIO, OLLAMA)
            metrics: List of efficiency metrics for this backend

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
                    "TPS P50",
                    "GPU Power P50 (W)",
                    "Efficiency (TPS/W)",
                ]
            )
        )

        # Table rows
        for item in metrics:
            lines.append(self._format_row(item))

        return lines

    def _format_row(self, item: dict[str, Any]) -> str:
        """Format a single efficiency table row.

        Args:
            item: Dict with device and efficiency metrics

        Returns:
            Markdown table row string
        """
        return (
            f"| {item['device']} | "
            f"{item['tps']:.1f} | "
            f"{item['gpu_watts']:.1f} | "
            f"{item['efficiency']:.2f} |"
        )
