"""Token metric visualization generation."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from plots.plot_token_metrics import (
    plot_tg_vs_output_tokens,
    plot_ttft_vs_input_tokens,
)


def collect_prompt_details_by_task(
    results_dir: Path,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group prompt-level metrics by task type and device.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dict mapping task types to device labels to prompt details
    """
    task_device_map: dict[str, defaultdict[str, list[dict[str, Any]]]] = {}

    for result_file in sorted(results_dir.glob("report_*.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"⚠️  Failed to read {result_file.name}: {exc}")
            continue

        device_info = data.get("device_info", {})
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

            # Build device label - GPU name + backend (no hostname)
            device_label = f"{gpu_name} | {backend_display}"

            model_data = task.get("model") or {}
            prompt_details = model_data.get("all_prompt_details") or []
            if not prompt_details:
                continue

            device_map = task_device_map.setdefault(task_type, defaultdict(list))
            device_map[device_label].extend(prompt_details)

    # Filter out empty device lists
    aggregated: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task_type, devices in task_device_map.items():
        filtered_devices = {
            device: details for device, details in devices.items() if details
        }
        if filtered_devices:
            aggregated[task_type] = filtered_devices

    return aggregated


def build_device_gpu_mapping(results_dir: Path) -> dict[str, str]:
    """Build mapping from device labels to GPU names.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dict mapping device labels to GPU names
    """
    device_gpu_mapping: dict[str, str] = {}

    for result_file in sorted(results_dir.glob("report_*.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
            device_info = data.get("device_info", {})
            gpu_name = device_info.get("gpu_name", "Unknown GPU")

            # Match the device_label format from collect_prompt_details_by_task
            for task in data.get("tasks", []):
                if task.get("task") not in {"llms", "vlms"}:
                    continue

                backend_name = (task.get("backend") or "UNKNOWN").upper()
                backend_display = {
                    "LM_STUDIO": "LM Studio",
                    "OLLAMA": "Ollama",
                }.get(backend_name, backend_name.title())

                # Build device label - GPU name + backend (no hostname)
                device_label = f"{gpu_name} | {backend_display}"
                device_gpu_mapping[device_label] = gpu_name
        except Exception:
            continue

    return device_gpu_mapping


def generate_token_metric_visualizations(
    results_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Create TTFT and generation-time plots for LLM and VLM prompt details.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dict mapping task types to plot paths (ttft, tg)
    """
    prompt_details = collect_prompt_details_by_task(results_dir)
    plots: dict[str, dict[str, Path]] = {}
    plots_dir = results_dir / "plots"

    device_gpu_mapping = build_device_gpu_mapping(results_dir)

    for task_type, devices in prompt_details.items():
        if not devices:
            continue

        prefix = "llm" if task_type == "llms" else "vlm"
        ttft_path = plots_dir / f"{prefix}_ttft_vs_input_tokens.png"
        tg_path = plots_dir / f"{prefix}_tg_vs_output_tokens.png"

        try:
            plot_ttft_vs_input_tokens(devices, device_gpu_mapping, ttft_path)
            plot_tg_vs_output_tokens(devices, device_gpu_mapping, tg_path)
            plots[task_type] = {"ttft": ttft_path, "tg": tg_path}
        except Exception as exc:
            print(f"⚠️  Failed to generate {task_type.upper()} token plots: {exc}")

    return plots
