"""Summary ranking table generator."""

from typing import Any

from ..extractors import TaskTimingExtractor
from ..formatters import format_platform, format_rank, format_vram
from .base import BaseTableGenerator


class SummaryTableGenerator(BaseTableGenerator):
    """Generate summary comparison table across all devices."""

    def generate(self) -> str:
        """Generate summary table with device rankings.

        Returns:
            Markdown table string or fallback message if no data
        """
        if not self._has_data():
            return "_No benchmark results available yet._\n"

        time_results = self._calculate_times()
        time_results.sort(
            key=lambda x: x["total_time"] if x["total_time"] > 0 else float("inf")
        )

        lines = self._build_header(
            [
                "Rank",
                "Device",
                "Platform",
                "CPU",
                "RAM",
                "GPU",
                "VRAM",
                "Embeddings, sts (s)",
                "LLM, lms (s)",
                "LLM, ollama (s)",
                "VLM, lms (s)",
                "VLM, ollama (s)",
                "Total Time (s)",
            ]
        )

        for rank, item in enumerate(time_results, 1):
            lines.append(self._format_row(rank, item))

        lines.extend(self._add_footnotes())
        return "\n".join(lines) + "\n"

    def _calculate_times(self) -> list[dict[str, Any]]:
        """Extract and calculate all timing metrics for each result.

        Returns:
            List of dicts with result and extracted times
        """
        time_results = []
        extractor = TaskTimingExtractor()

        for result in self.results:
            embeddings_time = extractor.extract_embeddings_time(result)
            llm_times = extractor.extract_llm_times_by_backend(result)
            vlm_times = extractor.extract_vlm_times_by_backend(result)
            total_time = extractor.calculate_total_time(result)

            time_results.append(
                {
                    "result": result,
                    "embeddings_time": embeddings_time,
                    "llm_lm_studio_time": llm_times["LM_STUDIO"],
                    "llm_ollama_time": llm_times["OLLAMA"],
                    "vlm_lm_studio_time": vlm_times["LM_STUDIO"],
                    "vlm_ollama_time": vlm_times["OLLAMA"],
                    "total_time": total_time,
                }
            )

        return time_results

    def _format_row(self, rank: int, item: dict[str, Any]) -> str:
        """Format a single table row.

        Args:
            rank: Device ranking position
            item: Dict with result and timing data

        Returns:
            Markdown table row string
        """
        result = item["result"]
        device_info = result["device_info"]

        # Format time values
        emb_str = f"{item['embeddings_time']:.2f}" if item["embeddings_time"] else "-"
        llm_lms_str = (
            f"{item['llm_lm_studio_time']:.2f}" if item["llm_lm_studio_time"] else "-"
        )
        llm_ollama_str = (
            f"{item['llm_ollama_time']:.2f}" if item["llm_ollama_time"] else "-"
        )
        vlm_lms_str = (
            f"{item['vlm_lm_studio_time']:.2f}" if item["vlm_lm_studio_time"] else "-"
        )
        vlm_ollama_str = (
            f"{item['vlm_ollama_time']:.2f}" if item["vlm_ollama_time"] else "-"
        )
        total_str = f"{item['total_time']:.2f}" if item["total_time"] > 0 else "-"

        # Format device info
        vram_str = format_vram(device_info.get("gpu_memory_gb", "N/A"))
        platform_str = format_platform(device_info.get("platform", "Unknown"))
        rank_str = format_rank(rank)

        return (
            f"| {rank_str} | {device_info['host']} | {platform_str} | "
            f"{device_info['processor']} | {device_info['ram_gb']:.0f} GB | "
            f"{device_info['gpu_name']} | {vram_str} | "
            f"{emb_str} | {llm_lms_str} | {llm_ollama_str} | {vlm_lms_str} | {vlm_ollama_str} | **{total_str}** |"
        )

    def _add_footnotes(self) -> list[str]:
        """Add footnotes explaining abbreviations.

        Returns:
            List of footnote strings
        """
        return [
            "",
            "*sts - sentence transformers*\n",
            "*lms - lm stuido*\n",
            "*ollama - ollama*\n",
        ]
