"""Summary ranking table generator."""

from typing import Any

from ..extractors import SummaryMetricsExtractor
from ..formatters import format_platform, format_vram
from .base import BaseTableGenerator


class SummaryTableGenerator(BaseTableGenerator):
    """Generate summary comparison table across all devices."""

    def generate(self) -> str:
        """Generate summary table with performance metrics.

        Returns:
            Markdown table string or fallback message if no data
        """
        if not self._has_data():
            return "_No benchmark results available yet._\n"

        metrics_results = self._extract_metrics()

        lines = self._build_header(
            [
                "Device",
                "Platform",
                "GPU",
                "VRAM",
                "Emb RPS P50",
                "LLM TPS P50 (lms)",
                "LLM TPS P50 (ollama)",
                "VLM TPS P50 (lms)",
                "VLM TPS P50 (ollama)",
                "GPU Power P50",
                "CPU Power P50",
            ]
        )

        for item in metrics_results:
            lines.append(self._format_row(item))

        lines.extend(self._add_footnotes())
        return "\n".join(lines) + "\n"

    def _extract_metrics(self) -> list[dict[str, Any]]:
        """Extract performance metrics for each result.

        Returns:
            List of dicts with result and extracted metrics
        """
        metrics_results = []
        extractor = SummaryMetricsExtractor()

        for result in self.results:
            embeddings_rps = extractor.extract_embeddings_rps_p50(result)
            llm_tps_by_backend = extractor.extract_llm_tps_p50_by_backend(result)
            vlm_tps_by_backend = extractor.extract_vlm_tps_p50_by_backend(result)
            gpu_watts = extractor.extract_gpu_watts_p50(result)
            cpu_watts = extractor.extract_cpu_watts_p50(result)

            metrics_results.append(
                {
                    "result": result,
                    "embeddings_rps": embeddings_rps,
                    "llm_tps_lms": llm_tps_by_backend["LM_STUDIO"],
                    "llm_tps_ollama": llm_tps_by_backend["OLLAMA"],
                    "vlm_tps_lms": vlm_tps_by_backend["LM_STUDIO"],
                    "vlm_tps_ollama": vlm_tps_by_backend["OLLAMA"],
                    "gpu_watts": gpu_watts,
                    "cpu_watts": cpu_watts,
                }
            )

        return metrics_results

    def _format_row(self, item: dict[str, Any]) -> str:
        """Format a single table row.

        Args:
            item: Dict with result and metrics data

        Returns:
            Markdown table row string
        """
        result = item["result"]
        device_info = result["device_info"]

        # Format metric values
        emb_rps_str = f"{item['embeddings_rps']:.1f}" if item["embeddings_rps"] else "-"
        llm_tps_lms_str = f"{item['llm_tps_lms']:.1f}" if item["llm_tps_lms"] else "-"
        llm_tps_ollama_str = (
            f"{item['llm_tps_ollama']:.1f}" if item["llm_tps_ollama"] else "-"
        )
        vlm_tps_lms_str = f"{item['vlm_tps_lms']:.1f}" if item["vlm_tps_lms"] else "-"
        vlm_tps_ollama_str = (
            f"{item['vlm_tps_ollama']:.1f}" if item["vlm_tps_ollama"] else "-"
        )
        gpu_watts_str = f"{item['gpu_watts']:.1f} W" if item["gpu_watts"] else "-"
        cpu_watts_str = f"{item['cpu_watts']:.1f} W" if item["cpu_watts"] else "-"

        # Format device info
        vram_str = format_vram(device_info.get("gpu_memory_gb", "N/A"))
        platform_str = format_platform(device_info.get("platform", "Unknown"))

        return (
            f"| {device_info['host']} | {platform_str} | "
            f"{device_info['gpu_name']} | {vram_str} | "
            f"{emb_rps_str} | {llm_tps_lms_str} | {llm_tps_ollama_str} | "
            f"{vlm_tps_lms_str} | {vlm_tps_ollama_str} | "
            f"{gpu_watts_str} | {cpu_watts_str} |"
        )

    def _add_footnotes(self) -> list[str]:
        """Add footnotes explaining metrics.

        Returns:
            List of footnote strings
        """
        return [
            "",
            "*RPS - Requests Per Second (embeddings throughput)*\n",
            "*TPS - Tokens Per Second (generation speed)*\n",
            "*W - Watts (power consumption)*\n",
        ]
