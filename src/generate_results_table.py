"""Generate markdown tables from benchmark results JSON files."""

import json
from pathlib import Path
from typing import Any


def load_results(results_dir: Path = Path("results")) -> list[dict[str, Any]]:
    """Load all result JSON files from the results directory."""
    results = []
    for json_file in results_dir.glob("report_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    return sorted(results, key=lambda x: x["device_info"]["host"])


def generate_summary_table(results: list[dict[str, Any]]) -> str:
    """Generate summary comparison table across all devices."""
    if not results:
        return "_No benchmark results available yet._\n"

    # Calculate scores for each result
    scored_results = []
    for result in results:
        device_info = result["device_info"]
        tasks = result["tasks"]

        # Extract scores
        embeddings_score = next(
            (t["task_score"] for t in tasks if t["task"] == "embeddings"), None
        )
        llm_score = next((t["task_score"] for t in tasks if t["task"] == "llms"), None)

        # Calculate total
        total_score = 0
        if embeddings_score:
            total_score += embeddings_score
        if llm_score:
            total_score += llm_score

        scored_results.append(
            {
                "result": result,
                "embeddings_score": embeddings_score,
                "llm_score": llm_score,
                "total_score": total_score,
            }
        )

    # Sort by total score (highest first)
    scored_results.sort(key=lambda x: x["total_score"], reverse=True)

    lines = [
        "| Rank | Device | CPU | RAM | GPU | VRAM | Embeddings | LLM | Total Score |",
        "|------|--------|-----|-----|-----|------|------------|-----|-------------|",
    ]

    for rank, item in enumerate(scored_results, 1):
        result = item["result"]
        device_info = result["device_info"]

        # Format values
        emb_str = f"{item['embeddings_score']:.2f}" if item["embeddings_score"] else "-"
        llm_str = f"{item['llm_score']:.2f}" if item["llm_score"] else "-"
        total_str = f"{item['total_score']:.2f}" if item["total_score"] > 0 else "-"

        # Format GPU memory
        gpu_mem = device_info.get("gpu_memory_gb", "N/A")
        if isinstance(gpu_mem, (int, float)):
            vram_str = f"{gpu_mem:.0f} GB"
        else:
            vram_str = str(gpu_mem)

        # Add medal emoji for top 3
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        rank_str = f"{rank_emoji} {rank}" if rank_emoji else str(rank)

        lines.append(
            f"| {rank_str} | {device_info['host']} | {device_info['processor']} | "
            f"{device_info['ram_gb']:.0f} GB | {device_info['gpu_name']} | {vram_str} | "
            f"{emb_str} | {llm_str} | **{total_str}** |"
        )

    return "\n".join(lines) + "\n"


def generate_embeddings_table(results: list[dict[str, Any]]) -> str:
    """Generate detailed embeddings performance table."""
    if not results:
        return ""

    # Collect all unique models across all results
    all_models = set()
    for result in results:
        embeddings_task = next(
            (t for t in result["tasks"] if t["task"] == "embeddings"), None
        )
        if embeddings_task:
            all_models.update(embeddings_task["models"].keys())

    if not all_models:
        return ""

    lines = [
        "#### Text Embeddings (100 IMDB samples)\n",
        "| Device | Model | Rows/sec | Time (s) | Embedding Dim | Batch Size |",
        "|--------|-------|----------|----------|---------------|------------|",
    ]

    for result in results:
        device = result["device_info"]["host"]
        embeddings_task = next(
            (t for t in result["tasks"] if t["task"] == "embeddings"), None
        )

        if embeddings_task:
            for model_name in sorted(all_models):
                if model_name in embeddings_task["models"]:
                    model_data = embeddings_task["models"][model_name]
                    lines.append(
                        f"| {device} | {model_name} | "
                        f"{model_data['median_rows_per_second']:.2f} | "
                        f"{model_data['median_encoding_time_seconds']:.2f} | "
                        f"{model_data['embedding_dimension']} | "
                        f"{model_data['batch_size']} |"
                    )

    return "\n".join(lines) + "\n"


def generate_llm_table(results: list[dict[str, Any]]) -> str:
    """Generate detailed LLM inference performance table."""
    if not results:
        return ""

    has_llm_results = any(
        any(t["task"] == "llms" for t in result["tasks"]) for result in results
    )

    if not has_llm_results:
        return ""

    lines = [
        "#### LLM Inference (3 prompts from awesome-chatgpt-prompts)\n",
        "| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |",
        "|--------|-------|------------|----------|-------------|--------------|---------------|",
    ]

    for result in results:
        device = result["device_info"]["host"]
        llm_task = next((t for t in result["tasks"] if t["task"] == "llms"), None)

        if llm_task and "model" in llm_task:
            model_data = llm_task["model"]
            # Get token counts from first run
            first_run = model_data["runs"][0] if model_data["runs"] else {}
            input_tokens = first_run.get("total_input_tokens", "-")
            output_tokens = first_run.get("total_output_tokens", "-")

            lines.append(
                f"| {device} | {model_data['model_name']} | "
                f"{model_data['final_median_tokens_per_sec']:.2f} | "
                f"{model_data['final_median_ttft_s']:.2f} | "
                f"{model_data['final_median_latency_s']:.2f} | "
                f"{input_tokens} | {output_tokens} |"
            )

    return "\n".join(lines) + "\n"


def get_gpu_vendor(gpu_name: str) -> str:
    """Determine GPU vendor from GPU name."""
    gpu_lower = gpu_name.lower()
    if (
        "apple" in gpu_lower
        or "m1" in gpu_lower
        or "m2" in gpu_lower
        or "m3" in gpu_lower
        or "m4" in gpu_lower
    ):
        return "Apple"
    elif (
        "nvidia" in gpu_lower
        or "rtx" in gpu_lower
        or "gtx" in gpu_lower
        or "tesla" in gpu_lower
        or "a100" in gpu_lower
    ):
        return "NVIDIA"
    elif "amd" in gpu_lower or "radeon" in gpu_lower or "rx" in gpu_lower:
        return "AMD"
    elif "intel" in gpu_lower or "arc" in gpu_lower or "uhd" in gpu_lower:
        return "Intel"
    else:
        return "Other"


def generate_gpu_grouped_tables(results: list[dict[str, Any]]) -> str:
    """Generate tables grouped by GPU vendor in collapsible sections."""
    if not results:
        return ""

    # Calculate scores and group by vendor
    vendor_groups: dict[str, list[dict[str, Any]]] = {}

    for result in results:
        device_info = result["device_info"]
        tasks = result["tasks"]

        # Extract scores
        embeddings_score = next(
            (t["task_score"] for t in tasks if t["task"] == "embeddings"), None
        )
        llm_score = next((t["task_score"] for t in tasks if t["task"] == "llms"), None)

        # Calculate total
        total_score = 0
        if embeddings_score:
            total_score += embeddings_score
        if llm_score:
            total_score += llm_score

        vendor = get_gpu_vendor(device_info["gpu_name"])

        if vendor not in vendor_groups:
            vendor_groups[vendor] = []

        vendor_groups[vendor].append(
            {
                "result": result,
                "embeddings_score": embeddings_score,
                "llm_score": llm_score,
                "total_score": total_score,
            }
        )

    # Sort each vendor group by score
    for vendor in vendor_groups:
        vendor_groups[vendor].sort(key=lambda x: x["total_score"], reverse=True)

    # Vendor emojis
    vendor_emojis = {
        "Apple": "🍎",
        "NVIDIA": "🟢",
        "AMD": "🔴",
        "Intel": "🔵",
        "Other": "⚪",
    }

    # Generate sections for each vendor
    sections = []
    for vendor in ["Apple", "NVIDIA", "AMD", "Intel", "Other"]:
        if vendor not in vendor_groups:
            continue

        emoji = vendor_emojis.get(vendor, "")
        items = vendor_groups[vendor]

        # Start collapsible section (open by default)
        sections.append("<details open>")
        sections.append(
            f'<summary><b>{emoji} {vendor}</b> ({len(items)} device{"s" if len(items) > 1 else ""})</summary>'
        )
        sections.append("")  # Empty line after summary

        # Table header
        sections.append(
            "| Rank | Device | CPU | RAM | GPU | VRAM | Embeddings | LLM | Total Score |"
        )
        sections.append(
            "|------|--------|-----|-----|-----|------|------------|-----|-------------|"
        )

        # Table rows
        for rank, item in enumerate(items, 1):
            result = item["result"]
            device_info = result["device_info"]

            # Format values
            emb_str = (
                f"{item['embeddings_score']:.2f}" if item["embeddings_score"] else "-"
            )
            llm_str = f"{item['llm_score']:.2f}" if item["llm_score"] else "-"
            total_str = f"{item['total_score']:.2f}" if item["total_score"] > 0 else "-"

            # Format GPU memory
            gpu_mem = device_info.get("gpu_memory_gb", "N/A")
            if isinstance(gpu_mem, (int, float)):
                vram_str = f"{gpu_mem:.0f} GB"
            else:
                vram_str = str(gpu_mem)

            # Add medal emoji for top 3 within vendor
            rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
            rank_str = f"{rank_emoji} {rank}" if rank_emoji else str(rank)

            sections.append(
                f"| {rank_str} | {device_info['host']} | {device_info['processor']} | "
                f"{device_info['ram_gb']:.0f} GB | {device_info['gpu_name']} | {vram_str} | "
                f"{emb_str} | {llm_str} | **{total_str}** |"
            )

        sections.append("")  # Empty line before closing tag
        sections.append("</details>")
        sections.append("")  # Empty line after section

    return "\n".join(sections)


def generate_full_results_section(results_dir: Path = Path("results")) -> str:
    """Generate complete results section for README."""
    results = load_results(results_dir)

    if not results:
        return "## Benchmark Results\n\n_No results available yet. Run benchmarks with `uv run python main.py`_\n"

    # Get latest timestamp
    latest = max(results, key=lambda x: x["timestamp"])
    timestamp = latest["timestamp"].split("T")[0]

    sections = [
        "## Benchmark Results\n",
        f"> **Last Updated**: {timestamp}\n",
        "### 🏆 Overall Ranking\n",
        generate_summary_table(results),
        "\n### 📊 By GPU Vendor\n",
        generate_gpu_grouped_tables(results),
        "\n### 📈 Detailed Performance\n",
        generate_embeddings_table(results),
    ]

    # Add LLM table if available
    llm_table = generate_llm_table(results)
    if llm_table:
        sections.append("\n" + llm_table)

    # Add notes
    sections.extend(
        [
            "\n---\n",
            "_All metrics are median values across 3 runs. ",
            "Scores calculated as: `num_tasks * 3600 / total_time_seconds`._\n",
        ]
    )

    return "\n".join(sections)


def update_readme(
    results_dir: Path = Path("results"), readme_path: Path = Path("README.md")
) -> None:
    """Update README.md with generated results section."""
    results_section = generate_full_results_section(results_dir)

    # Read existing README
    if readme_path.exists():
        with open(readme_path) as f:
            content = f.read()

        # Find and replace results section
        start_marker = "## Benchmark Results"
        end_marker = "\n## "  # Next section

        if start_marker in content:
            # Replace existing section
            start_idx = content.find(start_marker)
            remaining = content[start_idx + len(start_marker) :]

            # Find next section
            end_idx = remaining.find(end_marker)
            if end_idx != -1:
                # Replace between markers
                new_content = (
                    content[:start_idx]
                    + results_section
                    + "\n"
                    + remaining[end_idx + 1 :]
                )
            else:
                # Replace to end of file
                new_content = content[:start_idx] + results_section
        else:
            # Append to end
            new_content = content.rstrip() + "\n\n" + results_section
    else:
        # Create new README with just results
        new_content = results_section

    # Write updated README
    with open(readme_path, "w") as f:
        f.write(new_content)

    print(f"✅ Updated {readme_path} with benchmark results")


if __name__ == "__main__":
    update_readme()
