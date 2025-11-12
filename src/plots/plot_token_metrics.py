"""
Visualize prompt processing and generation metrics from benchmark results.

This script reads JSON benchmark results and creates plots showing:
1. Prompt processing time (ttft_s) vs input length (input_tokens)
2. Text generation time (tg_s) vs output length (output_tokens)
"""

import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from .vendor_color import get_gpu_vendor_color
import colorsys
from collections import defaultdict

# Get project root directory (2 levels up from src/plots/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# ---- Палитра: сразу ~20 контрастных цветов (tab20) + несколько доп. хексов ----
TAB20 = [mpl.colormaps["tab20"](i) for i in range(20)]  # RGBA из colormap
EXTRA_HEX = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
DEVICE_COLORS = TAB20 + [mpl.colors.to_rgba(x) for x in EXTRA_HEX]  # ~30 цветов


# Функция на случай, если устройств больше 30 — генерим по HSV равномерно
def get_color(idx: int, total: int | None = None):
    if idx < len(DEVICE_COLORS):
        return DEVICE_COLORS[idx]
    # равномерно по окружности
    h = (idx % 360) / 360.0
    return mpl.colors.hsv_to_rgb((h, 0.55, 0.9))


# ---- Маркеры: 20+ вариантов (только «хорошо читаемые» в легенде) ----
DEVICE_MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "<",
    ">",
    "D",
    "p",
    "*",
    "X",
    "P",
    "h",
    "H",
    "8",
    "d",
    "1",
    "2",
    "3",
    "4",
    "|",
    "_",
]


def get_marker(idx: int):
    return DEVICE_MARKERS[idx % len(DEVICE_MARKERS)]


# Helper to adjust color lightness
def _adjust_lightness(color_in, factor: float):
    """Lighten/darken a color by multiplying its HLS lightness by `factor`."""
    rgba = mpl.colors.to_rgba(color_in)
    r, g, b, a = rgba
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    lightness = max(0.0, min(1.0, lightness * factor))
    nr, ng, nb = colorsys.hls_to_rgb(h, lightness, s)
    return (nr, ng, nb, a)


def load_prompt_details(result_file: str):
    """
    Load prompt details from a benchmark result file with device info.
    Aggregates multiple runs by calculating mean for each prompt position.
    Returns separate entries for each model+backend combination.

    Args:
        result_file: Path to JSON result file

    Returns:
        list: List of dictionaries with device_label, gpu_name and prompt_details
    """
    with open(result_file) as f:
        data = json.load(f)

    # Extract device info
    device_info = data.get("device_info", {})
    gpu_name = device_info.get("gpu_name", "Unknown GPU")

    result_entries = []

    for task in data.get("tasks", []):
        task_type = task.get("task")

        if task_type == "llms" or task_type == "vlms":
            model_data = task.get("model", {})
            all_details = model_data.get("all_prompt_details", [])
            model_name = model_data.get("model_name", "Unknown Model")

            # Determine backend from model name
            if "/" in model_name:
                backend = "LM Studio"
            else:
                backend = "Ollama"

            # Create device label with GPU + Backend + Task type
            device_label = f"{gpu_name} | {backend}"

            # Group details by runs (assuming each run has same number of prompts)
            num_runs = model_data.get("num_runs", 3)
            num_prompts = model_data.get("num_prompts", 10)

            if len(all_details) != num_runs * num_prompts:
                print(
                    f"Warning: Expected {num_runs * num_prompts} details, got {len(all_details)} for {task_type}"
                )
                # Fallback to old behavior if structure doesn't match
                result_entries.append(
                    {
                        "device_label": device_label,
                        "gpu_name": gpu_name,
                        "prompt_details": all_details,
                    }
                )
                continue

            # Group by prompt index and calculate means
            prompt_aggregations = []
            for prompt_idx in range(num_prompts):
                # Collect data for this prompt from all runs
                prompt_data_across_runs = []
                for run_idx in range(num_runs):
                    detail_idx = run_idx * num_prompts + prompt_idx
                    if detail_idx < len(all_details):
                        prompt_data_across_runs.append(all_details[detail_idx])

                if not prompt_data_across_runs:
                    continue

                # Calculate mean values for this prompt
                mean_ttft = np.mean(
                    [
                        d.get("ttft_s", 0)
                        for d in prompt_data_across_runs
                        if d.get("ttft_s") is not None
                    ]
                )
                mean_tg = np.mean(
                    [
                        d.get("tg_s", 0)
                        for d in prompt_data_across_runs
                        if d.get("tg_s") is not None
                    ]
                )

                # Input/output tokens should be the same across runs for same prompt
                input_tokens = prompt_data_across_runs[0].get("input_tokens")
                output_tokens = np.mean(
                    [
                        d.get("output_tokens", 0)
                        for d in prompt_data_across_runs
                        if d.get("output_tokens") is not None
                    ]
                )

                aggregated_detail = {
                    "ttft_s": float(mean_ttft),
                    "tg_s": float(mean_tg),
                    "input_tokens": input_tokens,
                    "output_tokens": int(output_tokens),
                }

                prompt_aggregations.append(aggregated_detail)

            # Sort by input tokens (prompt length) for logical ordering
            prompt_aggregations.sort(key=lambda x: x["input_tokens"])

            result_entries.append(
                {
                    "device_label": device_label,
                    "gpu_name": gpu_name,
                    "prompt_details": prompt_aggregations,
                }
            )

    return result_entries


def bin_data(tokens: np.ndarray, times: np.ndarray, bin_size: int = 100):
    """
    Bin data by token count and calculate mean time for each bin.

    Args:
        tokens: Array of token counts
        times: Array of corresponding times
        bin_size: Size of each bin (default: 100 tokens)

    Returns:
        bin_centers: Array of bin center positions
        mean_times: Array of mean times for each bin
    """
    if len(tokens) == 0:
        return np.array([]), np.array([])

    # Create bins
    min_tokens = int(np.floor(tokens.min() / bin_size) * bin_size)
    max_tokens = int(np.ceil(tokens.max() / bin_size) * bin_size)
    bins = np.arange(min_tokens, max_tokens + bin_size, bin_size)

    # Assign each data point to a bin
    bin_indices = np.digitize(tokens, bins) - 1

    # Calculate mean for each bin
    bin_centers = []
    mean_times = []

    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.any():
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            mean_times.append(times[mask].mean())

    return np.array(bin_centers), np.array(mean_times)


def plot_ttft_vs_input_tokens(
    devices_data: dict,
    device_gpu_mapping: dict,
    output_path: Path | None = None,
    max_time: float = 30.0,
):
    """
    Plot Time To First Token vs Input Tokens for multiple devices using bar charts with bins.

    Args:
        devices_data: Dict mapping device_label to list of prompt details
        device_gpu_mapping: Dict mapping device_label to GPU name for color selection
        output_path: Path to save the plot (defaults to results/plots/ttft_vs_input_tokens.png)
        max_time: Maximum time in seconds to display on y-axis (default: 30.0)
    """
    if output_path is None:
        output_path = PLOTS_DIR / "ttft_vs_input_tokens.png"

    # Create figure with scientific style
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    has_data = False

    # Determine bin size dynamically based on data range
    all_input_tokens = []
    for details in devices_data.values():
        all_input_tokens.extend(
            [d["input_tokens"] for d in details if d.get("input_tokens") is not None]
        )

    if not all_input_tokens:
        print("No data available for TTFT vs Input Tokens plot")
        return

    token_range = max(all_input_tokens) - min(all_input_tokens)
    bin_size = max(100, int(token_range / 8))  # Larger adaptive bin size

    # Collect all binned data first to determine x positions
    binned_devices = {}
    all_bin_centers = set()

    for device_label, details in devices_data.items():
        # Filter out None values
        filtered = [
            (d["input_tokens"], d["ttft_s"])
            for d in details
            if d.get("input_tokens") is not None and d.get("ttft_s") is not None
        ]

        if not filtered:
            continue

        has_data = True

        # Extract data
        input_tokens = np.array([x[0] for x in filtered])
        ttft_times = np.array([x[1] for x in filtered])

        # Bin the data
        bin_centers, mean_times = bin_data(input_tokens, ttft_times, bin_size)

        binned_devices[device_label] = (bin_centers, mean_times)
        all_bin_centers.update(bin_centers)

    if not has_data:
        print("No data available for TTFT vs Input Tokens plot")
        return

    # Calculate bar width - make bars wider while preventing overlap
    num_devices = len(binned_devices)
    # Compute per-bin group width with padding so bars never overlap adjacent bins
    bin_padding = 0.15  # 15% empty space per bin for visual separation
    group_width = bin_size * (1 - bin_padding)
    bar_width = group_width / max(num_devices, 1)

    # Plot bars for each device
    variant_counts: defaultdict[tuple[str, str], int] = defaultdict(
        int
    )  # key: (vendor_hex, backend_key)
    for idx, (device_label, (bin_centers, mean_times)) in enumerate(
        binned_devices.items()
    ):
        # Vendor base color + backend-based shade (Ollama lighter, LM Studio darker)
        gpu_name = device_gpu_mapping.get(device_label, "Unknown GPU")
        vendor_color = get_gpu_vendor_color(gpu_name)

        backend = device_label.split(" | ", 1)[1] if " | " in device_label else ""
        backend_lower = backend.lower()
        base_factor = 1.0
        if backend_lower.startswith("ollama"):
            base_factor = 1.12  # lighter
        elif backend_lower.startswith("lm studio") or backend_lower.startswith(
            "lmstudio"
        ):
            base_factor = 0.88  # darker

        key = (vendor_color, backend_lower.split()[0] if backend_lower else "")
        idx_variant = variant_counts[key]
        variant_counts[key] += 1
        variant_ring = [0.95, 1.0, 1.05, 0.9, 1.1]
        factor = base_factor * variant_ring[idx_variant % len(variant_ring)]
        color = _adjust_lightness(vendor_color, factor)

        # Calculate x positions for this device's bars
        x_offset = (idx - num_devices / 2) * bar_width + bar_width / 2
        x_positions = bin_centers + x_offset

        # Clip values to max_time for display
        display_times = np.clip(mean_times, 0, max_time)

        # Plot bars
        ax.bar(
            x_positions,
            display_times,
            width=bar_width,
            color=color,
            alpha=0.8,
            label=device_label,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels on top of bars
        for x_pos, actual_time, display_time in zip(
            x_positions, mean_times, display_times
        ):
            # Format the time label
            if actual_time > max_time:
                time_label = f"{actual_time:.1f}s"
                y_pos = max_time
            else:
                time_label = f"{actual_time:.1f}s"
                y_pos = display_time

            # Add time label on top of bar
            ax.text(
                x_pos,
                y_pos + max_time * 0.02,
                time_label,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=color if actual_time <= max_time else "red",
            )

    # Styling
    ax.set_xlabel("Input Tokens (prompt length)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time To First Token (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Prompt Processing Performance\nTTFT vs Input Length",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Legend with vendor-colored shades
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=9,
    )

    # Set y-axis limit to max_time
    ax.set_ylim(bottom=0, top=max_time * 1.1)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    plt.tight_layout()

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ TTFT plot saved to {output_path}")
    plt.close()


def plot_tg_vs_output_tokens(
    devices_data: dict,
    device_gpu_mapping: dict,
    output_path: Path | None = None,
    max_time: float = 30.0,
):
    """
    Plot Text Generation Time vs Output Tokens for multiple devices using bar charts with bins.

    Args:
        devices_data: Dict mapping device_label to list of prompt details
        device_gpu_mapping: Dict mapping device_label to GPU name for color selection
        output_path: Path to save the plot (defaults to results/plots/tg_vs_output_tokens.png)
        max_time: Maximum time in seconds to display on y-axis (default: 30.0)
    """
    if output_path is None:
        output_path = PLOTS_DIR / "tg_vs_output_tokens.png"

    # Create figure with scientific style
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    has_data = False

    # Determine bin size dynamically based on data range
    all_output_tokens = []
    for details in devices_data.values():
        all_output_tokens.extend(
            [d["output_tokens"] for d in details if d.get("output_tokens") is not None]
        )

    if not all_output_tokens:
        print("No data available for Generation Time vs Output Tokens plot")
        return

    token_range = max(all_output_tokens) - min(all_output_tokens)
    bin_size = max(100, int(token_range / 8))  # Larger adaptive bin size

    # Collect all binned data first to determine x positions
    binned_devices = {}
    all_bin_centers = set()

    for device_label, details in devices_data.items():
        # Filter out None values
        filtered = [
            (d["output_tokens"], d["tg_s"])
            for d in details
            if d.get("output_tokens") is not None and d.get("tg_s") is not None
        ]

        if not filtered:
            continue

        has_data = True

        # Extract data
        output_tokens = np.array([x[0] for x in filtered])
        tg_times = np.array([x[1] for x in filtered])

        # Bin the data
        bin_centers, mean_times = bin_data(output_tokens, tg_times, bin_size)

        binned_devices[device_label] = (bin_centers, mean_times)
        all_bin_centers.update(bin_centers)

    if not has_data:
        print("No data available for Generation Time vs Output Tokens plot")
        return

    # Calculate bar width - make bars wider while preventing overlap
    num_devices = len(binned_devices)
    # Compute per-bin group width with padding so bars never overlap adjacent bins
    bin_padding = 0.15  # 15% empty space per bin for visual separation
    group_width = bin_size * (1 - bin_padding)
    bar_width = group_width / max(num_devices, 1)

    # Plot bars for each device
    variant_counts: defaultdict[tuple[str, str], int] = defaultdict(
        int
    )  # key: (vendor_hex, backend_key)
    for idx, (device_label, (bin_centers, mean_times)) in enumerate(
        binned_devices.items()
    ):
        # Vendor base color + backend-based shade (Ollama lighter, LM Studio darker)
        gpu_name = device_gpu_mapping.get(device_label, "Unknown GPU")
        vendor_color = get_gpu_vendor_color(gpu_name)

        backend = device_label.split(" | ", 1)[1] if " | " in device_label else ""
        backend_lower = backend.lower()
        base_factor = 1.0
        if backend_lower.startswith("ollama"):
            base_factor = 1.12  # lighter
        elif backend_lower.startswith("lm studio") or backend_lower.startswith(
            "lmstudio"
        ):
            base_factor = 0.88  # darker

        key = (vendor_color, backend_lower.split()[0] if backend_lower else "")
        idx_variant = variant_counts[key]
        variant_counts[key] += 1
        variant_ring = [0.95, 1.0, 1.05, 0.9, 1.1]
        factor = base_factor * variant_ring[idx_variant % len(variant_ring)]
        color = _adjust_lightness(vendor_color, factor)

        # Calculate x positions for this device's bars
        x_offset = (idx - num_devices / 2) * bar_width + bar_width / 2
        x_positions = bin_centers + x_offset

        # Clip values to max_time for display
        display_times = np.clip(mean_times, 0, max_time)

        # Plot bars
        ax.bar(
            x_positions,
            display_times,
            width=bar_width,
            color=color,
            alpha=0.8,
            label=device_label,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels on top of bars
        for x_pos, actual_time, display_time in zip(
            x_positions, mean_times, display_times
        ):
            # Format the time label
            if actual_time > max_time:
                time_label = f"{actual_time:.1f}s"
                y_pos = max_time
            else:
                time_label = f"{actual_time:.1f}s"
                y_pos = display_time

            # Add time label on top of bar
            ax.text(
                x_pos,
                y_pos + max_time * 0.02,
                time_label,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=color if actual_time <= max_time else "red",
            )

    # Styling
    ax.set_xlabel("Output Tokens (response length)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Text Generation Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Text Generation Performance\nGeneration Time vs Output Length",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Legend with vendor-colored shades
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=9,
    )

    # Set y-axis limit to max_time
    ax.set_ylim(bottom=0, top=max_time * 1.1)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    plt.tight_layout()

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Generation time plot saved to {output_path}")
    plt.close()


def main():
    """Main function to load data and create plots."""
    # Find all result files
    result_files = list(RESULTS_DIR.glob("report_*.json"))

    if not result_files:
        print(f"No benchmark result files found in {RESULTS_DIR}/")
        return

    print(f"Found {len(result_files)} result file(s)")

    # Group prompt details by device
    devices_data: dict[str, list] = {}
    device_gpu_mapping: dict[str, str] = {}  # Map device_label to GPU name

    for result_file in result_files:
        print(f"Loading: {result_file.name}")
        entries = load_prompt_details(str(result_file))

        for entry in entries:
            device_label = entry["device_label"]
            gpu_name = entry["gpu_name"]
            prompt_details = entry["prompt_details"]

            # Store GPU name for color mapping
            device_gpu_mapping[device_label] = gpu_name

            # Aggregate data for this device (append if device already exists)
            if device_label not in devices_data:
                devices_data[device_label] = []

            devices_data[device_label].extend(prompt_details)

    # Print summary
    total_details = 0
    print("\nDevice summary:")
    for device_label, details in devices_data.items():
        print(f"  {device_label}: {len(details)} prompt details")
        total_details += len(details)

    if total_details == 0:
        print("\nNo prompt details found in results. Run benchmarks first!")
        return

    print(f"\nTotal prompt details: {total_details}")
    print("\nGenerating plots...")

    # Create plots with device grouping and vendor-based colors
    plot_ttft_vs_input_tokens(devices_data, device_gpu_mapping)
    plot_tg_vs_output_tokens(devices_data, device_gpu_mapping)

    print(f"\nDone! Plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
