"""GPU vendor-grouped tables generator."""

from typing import Any

from ..extractors import TaskTimingExtractor
from ..formatters import (
    VENDOR_EMOJIS,
    format_platform,
    format_rank,
    format_vram,
    get_gpu_vendor,
)
from .base import BaseTableGenerator


class GPUGroupedTableGenerator(BaseTableGenerator):
    """Generate tables grouped by GPU vendor in collapsible sections."""

    def generate(self) -> str:
        """Generate GPU vendor-grouped tables.

        Returns:
            Markdown with collapsible sections for each vendor
        """
        if not self._has_data():
            return ""

        vendor_groups = self._group_by_vendor()
        sections = []

        for vendor in ["Apple", "NVIDIA", "AMD", "Intel", "Other"]:
            if items := vendor_groups.get(vendor):
                sections.append(self._generate_vendor_section(vendor, items))

        return "\n".join(sections)

    def _group_by_vendor(self) -> dict[str, list[dict[str, Any]]]:
        """Group results by GPU vendor.

        Returns:
            Dict mapping vendor names to lists of result items
        """
        vendor_groups: dict[str, list[dict[str, Any]]] = {}
        extractor = TaskTimingExtractor()

        for result in self.results:
            device_info = result["device_info"]
            tasks = result["tasks"]

            # Extract times - sum all tasks of each type (handles BOTH backend case)
            embeddings_time = extractor.extract_embeddings_time(result)

            # Sum ALL llm task times (in case of BOTH backend)
            llm_times = [t["total_time_seconds"] for t in tasks if t["task"] == "llms"]
            llm_time = sum(llm_times) if llm_times else None

            # Calculate total
            total_time = 0.0
            if embeddings_time:
                total_time += embeddings_time
            if llm_time:
                total_time += llm_time

            vendor = get_gpu_vendor(device_info["gpu_name"])

            if vendor not in vendor_groups:
                vendor_groups[vendor] = []

            vendor_groups[vendor].append(
                {
                    "result": result,
                    "embeddings_time": embeddings_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                }
            )

        # Sort each vendor group by time (lowest first - faster is better)
        for vendor in vendor_groups:
            vendor_groups[vendor].sort(
                key=lambda x: x["total_time"] if x["total_time"] > 0 else float("inf")
            )

        return vendor_groups

    def _generate_vendor_section(self, vendor: str, items: list[dict[str, Any]]) -> str:
        """Generate collapsible section for a vendor.

        Args:
            vendor: Vendor name (Apple, NVIDIA, etc.)
            items: List of result items for this vendor

        Returns:
            Markdown collapsible section string
        """
        emoji = VENDOR_EMOJIS.get(vendor, "")
        lines = []

        # Start collapsible section (open by default)
        lines.append("<details open>")
        lines.append(
            f'<summary><b>{emoji} {vendor}</b> ({len(items)} device{"s" if len(items) > 1 else ""})</summary>'
        )
        lines.append("")  # Empty line after summary

        # Table header
        lines.append(
            "| Rank | Device | Platform | CPU | RAM | GPU | VRAM | Embeddings (s) | LLM (s) | Total Time (s) |"
        )
        lines.append(
            "|------|--------|----------|-----|-----|-----|------|----------------|---------|----------------|"
        )

        # Table rows
        for rank, item in enumerate(items, 1):
            lines.append(self._format_row(rank, item))

        lines.append("")  # Empty line before closing tag
        lines.append("</details>")
        lines.append("")  # Empty line after section

        return "\n".join(lines)

    def _format_row(self, rank: int, item: dict[str, Any]) -> str:
        """Format a single table row within a vendor group.

        Args:
            rank: Ranking within the vendor group
            item: Result item with timing data

        Returns:
            Markdown table row string
        """
        result = item["result"]
        device_info = result["device_info"]

        # Format values
        emb_str = f"{item['embeddings_time']:.2f}" if item["embeddings_time"] else "-"
        llm_str = f"{item['llm_time']:.2f}" if item["llm_time"] else "-"
        total_str = f"{item['total_time']:.2f}" if item["total_time"] > 0 else "-"

        # Format device info
        vram_str = format_vram(device_info.get("gpu_memory_gb", "N/A"))
        platform_str = format_platform(device_info.get("platform", "Unknown"))
        rank_str = format_rank(rank)

        return (
            f"| {rank_str} | {device_info['host']} | {platform_str} | "
            f"{device_info['processor']} | {device_info['ram_gb']:.0f} GB | "
            f"{device_info['gpu_name']} | {vram_str} | "
            f"{emb_str} | {llm_str} | **{total_str}** |"
        )
