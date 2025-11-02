"""Generate markdown tables from benchmark results JSON files."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from plots.plot_embeddings_metrics import plot_embeddings_performance
from plots.plot_llm_metrics import plot_llm_performance
from plots.plot_token_metrics import (
    plot_tg_vs_output_tokens,
    plot_ttft_vs_input_tokens,
)
from plots.plot_vlm_metrics import plot_vlm_performance
import seaborn as sns

sns.set_style("darkgrid")


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

    # Calculate times for each result
    time_results = []
    for result in results:
        device_info = result["device_info"]
        tasks = result["tasks"]

        # Extract times - sum all tasks of each type (handles BOTH backend case)
        embeddings_time = next(
            (t["total_time_seconds"] for t in tasks if t["task"] == "embeddings"), None
        )

        # Sum ALL llm task times (in case of BOTH backend)
        llm_times = [t["total_time_seconds"] for t in tasks if t["task"] == "llms"]
        llm_time = sum(llm_times) if llm_times else None

        # Sum ALL vlm task times (in case of BOTH backend)
        vlm_times = [t["total_time_seconds"] for t in tasks if t["task"] == "vlms"]
        vlm_time = sum(vlm_times) if vlm_times else None

        # Calculate total time
        total_time = 0
        if embeddings_time:
            total_time += embeddings_time
        if llm_time:
            total_time += llm_time
        if vlm_time:
            total_time += vlm_time

        time_results.append(
            {
                "result": result,
                "embeddings_time": embeddings_time,
                "llm_time": llm_time,
                "vlm_time": vlm_time,
                "total_time": total_time,
            }
        )

    # Sort by total time (lowest first - faster is better)
    time_results.sort(
        key=lambda x: x["total_time"] if x["total_time"] > 0 else float("inf")
    )

    lines = [
        "| Rank | Device | Platform | CPU | RAM | GPU | VRAM | Embeddings (s) | LLM (s) | VLM (s) | Total Time (s) |",
        "|------|--------|----------|-----|-----|-----|------|----------------|---------|---------|----------------|",
    ]

    for rank, item in enumerate(time_results, 1):
        result = item["result"]
        device_info = result["device_info"]

        # Format values
        emb_str = f"{item['embeddings_time']:.2f}" if item["embeddings_time"] else "-"
        llm_str = f"{item['llm_time']:.2f}" if item["llm_time"] else "-"
        vlm_str = f"{item['vlm_time']:.2f}" if item["vlm_time"] else "-"
        total_str = f"{item['total_time']:.2f}" if item["total_time"] > 0 else "-"

        # Format GPU memory
        gpu_mem = device_info.get("gpu_memory_gb", "N/A")
        if isinstance(gpu_mem, (int, float)):
            vram_str = f"{gpu_mem:.0f} GB"
        else:
            vram_str = str(gpu_mem)

        # Format platform with emoji
        platform = device_info.get("platform", "Unknown")
        platform_emoji_map = {
            "Darwin": "🍏 macOS",
            "Linux": "🐧 Linux",
            "Windows": "🪟 Windows",
        }
        platform_str = platform_emoji_map.get(platform, platform)

        # Add medal emoji for top 3
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        rank_str = f"{rank_emoji} {rank}" if rank_emoji else str(rank)

        lines.append(
            f"| {rank_str} | {device_info['host']} | {platform_str} | "
            f"{device_info['processor']} | {device_info['ram_gb']:.0f} GB | "
            f"{device_info['gpu_name']} | {vram_str} | "
            f"{emb_str} | {llm_str} | {vlm_str} | **{total_str}** |"
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

                    # Format rows/sec with std
                    rps_median = model_data["median_rows_per_second"]
                    rps_std = model_data.get("std_rows_per_second", 0)
                    rps_str = f"{rps_median:.2f} ± {rps_std:.2f}"

                    # Format time with std
                    time_median = model_data["median_encoding_time_seconds"]
                    time_std = model_data.get("std_encoding_time_seconds", 0)
                    time_str = f"{time_median:.2f} ± {time_std:.2f}"

                    lines.append(
                        f"| {device} | {model_name} | "
                        f"{rps_str} | "
                        f"{time_str} | "
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

    # Collect all LLM tasks grouped by backend
    backend_results: dict[str, list] = {"LM_STUDIO": [], "OLLAMA": [], "UNKNOWN": []}

    for result in results:
        device = result["device_info"]["host"]
        # Get ALL llm tasks (not just first)
        llm_tasks = [t for t in result["tasks"] if t["task"] == "llms"]

        for llm_task in llm_tasks:
            if "model" in llm_task:
                backend = llm_task.get("backend", "UNKNOWN")
                backend_results[backend].append({"device": device, "task": llm_task})

    lines = ["#### LLM Inference (3 prompts from awesome-chatgpt-prompts)\n"]

    # Process backends in order: LM_STUDIO, then OLLAMA
    for backend_name in ["LM_STUDIO", "OLLAMA", "UNKNOWN"]:
        backend_data = backend_results[backend_name]
        if not backend_data:
            continue

        # Add backend header
        backend_display = backend_name.replace("_", " ")
        lines.append(f"\n**{backend_display}**\n")

        # Table header
        lines.extend(
            [
                "| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |",
                "|--------|-------|------------|----------|-------------|--------------|---------------|",
            ]
        )

        for item in backend_data:
            device = item["device"]
            model_data = item["task"]["model"]

            # Get token counts from first run
            first_run = model_data["runs"][0] if model_data["runs"] else {}
            input_tokens = first_run.get("total_input_tokens", "-")
            output_tokens = first_run.get("total_output_tokens", "-")

            # Format tokens/sec with std
            tps_median = model_data["final_50p_tokens_per_sec"]
            tps_std = model_data.get("final_std_tokens_per_sec", 0)
            tps_str = f"{tps_median:.2f} ± {tps_std:.2f}"

            # Format TTFT with std
            ttft_median = model_data["final_50p_ttft_s"]
            ttft_std = model_data.get("final_std_ttft_s", 0)
            ttft_str = f"{ttft_median:.2f} ± {ttft_std:.2f}"

            # Format latency with std
            lat_median = model_data["final_50p_latency_s"]
            lat_std = model_data.get("final_std_latency_s", 0)
            lat_str = f"{lat_median:.2f} ± {lat_std:.2f}"

            lines.append(
                f"| {device} | {model_data['model_name']} | "
                f"{tps_str} | "
                f"{ttft_str} | "
                f"{lat_str} | "
                f"{input_tokens} | {output_tokens} |"
            )

    return "\n".join(lines) + "\n"


def generate_vlm_table(results: list[dict[str, Any]]) -> str:
    """Generate detailed VLM inference performance table."""
    if not results:
        return ""

    has_vlm_results = any(
        any(t["task"] == "vlms" for t in result["tasks"]) for result in results
    )

    if not has_vlm_results:
        return ""

    # Collect all VLM tasks grouped by backend
    backend_results: dict[str, list] = {"LM_STUDIO": [], "OLLAMA": [], "UNKNOWN": []}

    for result in results:
        device = result["device_info"]["host"]
        # Get ALL vlm tasks (not just first)
        vlm_tasks = [t for t in result["tasks"] if t["task"] == "vlms"]

        for vlm_task in vlm_tasks:
            if "model" in vlm_task:
                backend = vlm_task.get("backend", "UNKNOWN")
                backend_results[backend].append({"device": device, "task": vlm_task})

    lines = ["#### VLM Inference (3 questions from Hallucination_COCO)\n"]

    # Process backends in order: LM_STUDIO, then OLLAMA
    for backend_name in ["LM_STUDIO", "OLLAMA", "UNKNOWN"]:
        backend_data = backend_results[backend_name]
        if not backend_data:
            continue

        # Add backend header
        backend_display = backend_name.replace("_", " ")
        lines.append(f"\n**{backend_display}**\n")

        # Table header
        lines.extend(
            [
                "| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |",
                "|--------|-------|------------|----------|-------------|--------------|---------------|",
            ]
        )

        for item in backend_data:
            device = item["device"]
            model_data = item["task"]["model"]

            # Get token counts from first run
            first_run = model_data["runs"][0] if model_data["runs"] else {}
            input_tokens = first_run.get("total_input_tokens", "-")
            output_tokens = first_run.get("total_output_tokens", "-")

            # Format tokens/sec with std
            tps_median = model_data.get("final_50p_tokens_per_sec")
            tps_std = model_data.get("final_std_tokens_per_sec", 0)
            tps_str = f"{tps_median:.2f} ± {tps_std:.2f}" if tps_median else "N/A"

            # Format TTFT with std
            ttft_median = model_data.get("final_50p_ttft_s")
            ttft_std = model_data.get("final_std_ttft_s", 0)
            ttft_str = f"{ttft_median:.2f} ± {ttft_std:.2f}" if ttft_median else "N/A"

            # Format latency with std
            lat_median = model_data.get("final_50p_latency_s")
            lat_std = model_data.get("final_std_latency_s", 0)
            lat_str = f"{lat_median:.2f} ± {lat_std:.2f}" if lat_median else "N/A"

            lines.append(
                f"| {device} | {model_data['model_name']} | "
                f"{tps_str} | "
                f"{ttft_str} | "
                f"{lat_str} | "
                f"{input_tokens} | {output_tokens} |"
            )

    return "\n".join(lines) + "\n"


def collect_prompt_details_by_task(
    results_dir: Path,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group prompt-level metrics by task type and device."""
    task_device_map: dict[str, defaultdict[str, list[dict[str, Any]]]] = {}

    for result_file in sorted(results_dir.glob("report_*.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"⚠️  Failed to read {result_file.name}: {exc}")
            continue

        device_info = data.get("device_info", {})
        host = device_info.get("host")
        gpu_name = device_info.get("gpu_name", "Unknown GPU")

        for task in data.get("tasks", []):
            task_type = task.get("task")
            if task_type not in {"llms", "vlms"}:
                continue

            backend_name = (task.get("backend") or "UNKNOWN").upper()
            backend_display = {
                "LM_STUDIO": "LM Studio",
                "OLLAMA": "Ollama",
            }.get(backend_name, backend_name.title())

            parts = []
            if host:
                parts.append(host)
            if gpu_name:
                parts.append(gpu_name)
            parts.append(backend_display)
            device_label = " | ".join(parts)

            model_data = task.get("model") or {}
            prompt_details = model_data.get("all_prompt_details") or []
            if not prompt_details:
                continue

            device_map = task_device_map.setdefault(task_type, defaultdict(list))
            device_map[device_label].extend(prompt_details)

    aggregated: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task_type, devices in task_device_map.items():
        filtered_devices = {
            device: details for device, details in devices.items() if details
        }
        if filtered_devices:
            aggregated[task_type] = filtered_devices

    return aggregated


def generate_token_metric_visualizations(
    results_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Create TTFT and generation-time plots for LLM and VLM prompt details."""
    prompt_details = collect_prompt_details_by_task(results_dir)
    plots: dict[str, dict[str, Path]] = {}
    plots_dir = results_dir / "plots"

    for task_type, devices in prompt_details.items():
        if not devices:
            continue

        prefix = "llm" if task_type == "llms" else "vlm"
        ttft_path = plots_dir / f"{prefix}_ttft_vs_input_tokens.png"
        tg_path = plots_dir / f"{prefix}_tg_vs_output_tokens.png"

        try:
            plot_ttft_vs_input_tokens(devices, ttft_path)
            plot_tg_vs_output_tokens(devices, tg_path)
            plots[task_type] = {"ttft": ttft_path, "tg": tg_path}
        except Exception as exc:
            print(f"⚠️  Failed to generate {task_type.upper()} token plots: {exc}")

    return plots


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

    # Calculate times and group by vendor
    vendor_groups: dict[str, list[dict[str, Any]]] = {}

    for result in results:
        device_info = result["device_info"]
        tasks = result["tasks"]

        # Extract times - sum all tasks of each type (handles BOTH backend case)
        embeddings_time = next(
            (t["total_time_seconds"] for t in tasks if t["task"] == "embeddings"), None
        )

        # Sum ALL llm task times (in case of BOTH backend)
        llm_times = [t["total_time_seconds"] for t in tasks if t["task"] == "llms"]
        llm_time = sum(llm_times) if llm_times else None

        # Calculate total
        total_time = 0
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

    # Vendor emojis
    vendor_emojis = {
        "Apple": "⚫",
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
            "| Rank | Device | Platform | CPU | RAM | GPU | VRAM | Embeddings (s) | LLM (s) | Total Time (s) |"
        )
        sections.append(
            "|------|--------|----------|-----|-----|-----|------|----------------|---------|----------------|"
        )

        # Table rows
        for rank, item in enumerate(items, 1):
            result = item["result"]
            device_info = result["device_info"]

            # Format values
            emb_str = (
                f"{item['embeddings_time']:.2f}" if item["embeddings_time"] else "-"
            )
            llm_str = f"{item['llm_time']:.2f}" if item["llm_time"] else "-"
            total_str = f"{item['total_time']:.2f}" if item["total_time"] > 0 else "-"

            # Format GPU memory
            gpu_mem = device_info.get("gpu_memory_gb", "N/A")
            if isinstance(gpu_mem, (int, float)):
                vram_str = f"{gpu_mem:.0f} GB"
            else:
                vram_str = str(gpu_mem)

            # Format platform with emoji
            platform = device_info.get("platform", "Unknown")
            platform_emoji_map = {
                "Darwin": "🍏 macOS",
                "Linux": "🐧 Linux",
                "Windows": "🪟 Windows",
            }
            platform_str = platform_emoji_map.get(platform, platform)

            # Add medal emoji for top 3 within vendor
            rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
            rank_str = f"{rank_emoji} {rank}" if rank_emoji else str(rank)

            sections.append(
                f"| {rank_str} | {device_info['host']} | {platform_str} | "
                f"{device_info['processor']} | {device_info['ram_gb']:.0f} GB | "
                f"{device_info['gpu_name']} | {vram_str} | "
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
    ]

    has_embeddings_results = any(
        any(t["task"] == "embeddings" for t in result["tasks"]) for result in results
    )
    has_llm_results = any(
        any(t["task"] == "llms" for t in result["tasks"]) for result in results
    )
    has_vlm_results = any(
        any(t["task"] == "vlms" for t in result["tasks"]) for result in results
    )

    token_plots: dict[str, dict[str, Path]] = {}
    if has_llm_results or has_vlm_results:
        token_plots = generate_token_metric_visualizations(results_dir)

    # Generate embeddings performance plot if embeddings results exist
    if has_embeddings_results:
        try:
            # Generate the plot
            plot_embeddings_performance(results_dir)

            # Add plot to README
            sections.append("\n#### Embeddings Performance Visualization\n")
            sections.append(
                "![Embeddings Performance Profile](results/plots/embeddings_performance.png)\n"
            )
            sections.append(
                "*Throughput comparison for different embedding models across hardware. "
                "Higher values indicate better performance.*\n"
            )
        except Exception as e:
            print(f"⚠️  Failed to generate embeddings plot: {e}")

    # Embeddings section
    embeddings_section = generate_embeddings_table(results)
    if embeddings_section:
        sections.append("\n### 📈 Embeddings\n")
        sections.append(embeddings_section)
        if has_embeddings_results:
            sections.append(
                "![Embeddings Performance Profile](results/plots/embeddings_performance.png)\n"
            )
            sections.append(
                "*Throughput comparison for different embedding models across hardware. "
                "Higher values indicate better performance.*\n"
            )

    # LLM section
    llm_table = generate_llm_table(results)
    if llm_table:
        sections.append("\n### 🧠 LLMs\n")
        sections.append(llm_table)
        llm_token_plots = token_plots.get("llms")
        if llm_token_plots:
            sections.append(
                f"![LLM TTFT vs Input Tokens]({llm_token_plots['ttft'].as_posix()})\n"
            )
            sections.append(
                "*Time To First Token across prompt lengths. Lower values mean faster first responses.*\n\n"
            )
            sections.append(
                f"![LLM Generation Time vs Output Tokens]({llm_token_plots['tg'].as_posix()})\n"
            )
            sections.append(
                "*Generation time growth relative to output length. Lower values reflect faster completions.*\n"
            )

        if has_llm_results:
            try:
                plot_llm_performance(results_dir)
                sections.append("![LLM TTFT Performance](results/plots/llm_ttft.png)\n")
                sections.append(
                    "*Time To First Token (TTFT) - Lower is better. "
                    "Measures response latency.*\n\n"
                )
                sections.append(
                    "![LLM Throughput Performance](results/plots/llm_tps.png)\n"
                )
                sections.append(
                    "*Token Generation per second (TG) - Higher is better. "
                    "Measures token generation.*\n"
                )
            except Exception as e:
                print(f"⚠️  Failed to generate LLM plots: {e}")

    # VLM section
    vlm_table = generate_vlm_table(results)
    if vlm_table:
        sections.append("\n### 👁️ VLMs\n")
        sections.append(vlm_table)
        vlm_token_plots = token_plots.get("vlms")
        if vlm_token_plots:
            sections.append(
                f"![VLM TTFT vs Input Tokens]({vlm_token_plots['ttft'].as_posix()})\n"
            )
            sections.append(
                "*TTFT behaviour for multimodal prompts. Lower values mean faster first visual-token outputs.*\n\n"
            )
            sections.append(
                f"![VLM Generation Time vs Output Tokens]({vlm_token_plots['tg'].as_posix()})\n"
            )
            sections.append(
                "*Generation time vs output token count for multimodal responses. Lower values are faster.*\n"
            )

    if has_vlm_results:
        try:
            # Generate the plots
            plot_vlm_performance(results_dir)

            sections.append("![VLM TTFT Performance](results/plots/vlm_ttft.png)\n")
            sections.append(
                "*Time To First Token (TTFT) - Lower is better. "
                "Measures response latency.*\n\n"
            )
            sections.append(
                "![VLM Throughput Performance](results/plots/vlm_tps.png)\n"
            )
            sections.append(
                "*Token Generation per second (TG) - Higher is better. "
                "Measures token generation.*\n"
            )
        except Exception as e:
            print(f"⚠️  Failed to generate VLM plots: {e}")

    # Add notes
    sections.extend(
        [
            "\n---\n",
            "_All metrics are shown as median ± standard deviation across 3 runs. ",
            "Lower times are better (faster performance)._\n",
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
