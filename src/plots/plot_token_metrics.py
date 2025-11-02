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


def load_prompt_details(result_file: str):
    """
    Load all prompt details from a benchmark result file with device info.

    Args:
        result_file: Path to JSON result file

    Returns:
        dict: Dictionary with device_info and prompt_details list
    """
    with open(result_file) as f:
        data = json.load(f)

    # Extract device info
    device_info = data.get("device_info", {})
    gpu_name = device_info.get("gpu_name", "Unknown GPU")
    host = device_info.get("host", "Unknown")
    device_label = f"{gpu_name}"

    prompt_details = []

    for task in data.get("tasks", []):
        task_type = task.get("task")

        if task_type == "llms" or task_type == "vlms":
            model_data = task.get("model", {})
            details = model_data.get("all_prompt_details", [])
            prompt_details.extend(details)

    return {
        "device_label": device_label,
        "host": host,
        "gpu_name": gpu_name,
        "prompt_details": prompt_details,
    }


def plot_ttft_vs_input_tokens(
    devices_data: dict, device_gpu_mapping: dict, output_path: Path | None = None
):
    """
    Plot Time To First Token vs Input Tokens for multiple devices.

    Args:
        devices_data: Dict mapping device_label to list of prompt details
        device_gpu_mapping: Dict mapping device_label to GPU name for color selection
        output_path: Path to save the plot (defaults to results/plots/ttft_vs_input_tokens.png)
    """
    if output_path is None:
        output_path = PLOTS_DIR / "ttft_vs_input_tokens.png"

    # Create figure with scientific style
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    has_data = False

    # Plot each device with different color/marker
    for idx, (device_label, details) in enumerate(devices_data.items()):
        # Filter out None values
        filtered = [
            (d["input_tokens"], d["ttft_s"])
            for d in details
            if d.get("input_tokens") is not None and d.get("ttft_s") is not None
        ]

        if not filtered:
            continue

        has_data = True
        input_tokens = np.array([x[0] for x in filtered])
        ttft_times = np.array([x[1] for x in filtered])

        # Get vendor-based color and marker for this device
        gpu_name = device_gpu_mapping.get(device_label, "Unknown GPU")
        vendor_color = get_gpu_vendor_color(gpu_name)
        marker = get_marker(idx)

        # For multiple devices of same vendor, adjust hue slightly
        color = mpl.colors.to_rgba(vendor_color)

        # Scatter plot
        ax.scatter(
            input_tokens,
            ttft_times,
            alpha=0.6,
            s=100,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidth=0.5,
            label=device_label,
            zorder=3,
        )

        # Add trend line for this device
        if len(input_tokens) > 1:
            z = np.polyfit(input_tokens, ttft_times, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(input_tokens.min(), input_tokens.max(), 100)
            ax.plot(
                x_trend,
                p(x_trend),
                "--",
                color=color,
                alpha=0.3,
                linewidth=1.5,
                zorder=2,
            )

    if not has_data:
        print("No data available for TTFT vs Input Tokens plot")
        return

    # Styling
    ax.set_xlabel("Input Tokens (prompt length)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time To First Token (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Prompt Processing Performance\nTTFT vs Input Length",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Legend
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=10,
    )

    # Start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ TTFT plot saved to {output_path}")
    plt.close()


def plot_tg_vs_output_tokens(
    devices_data: dict, device_gpu_mapping: dict, output_path: Path | None = None
):
    """
    Plot Text Generation Time vs Output Tokens for multiple devices.

    Args:
        devices_data: Dict mapping device_label to list of prompt details
        device_gpu_mapping: Dict mapping device_label to GPU name for color selection
        output_path: Path to save the plot (defaults to results/plots/tg_vs_output_tokens.png)
    """
    if output_path is None:
        output_path = PLOTS_DIR / "tg_vs_output_tokens.png"

    # Create figure with scientific style
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    has_data = False

    # Plot each device with different color/marker
    for idx, (device_label, details) in enumerate(devices_data.items()):
        # Filter out None values
        filtered = [
            (d["output_tokens"], d["tg_s"])
            for d in details
            if d.get("output_tokens") is not None and d.get("tg_s") is not None
        ]

        if not filtered:
            continue

        has_data = True
        output_tokens = np.array([x[0] for x in filtered])
        tg_times = np.array([x[1] for x in filtered])

        # Get vendor-based color and marker for this device
        gpu_name = device_gpu_mapping.get(device_label, "Unknown GPU")
        vendor_color = get_gpu_vendor_color(gpu_name)
        marker = get_marker(idx)

        # For multiple devices of same vendor, adjust hue slightly
        color = mpl.colors.to_rgba(vendor_color)

        # Scatter plot
        ax.scatter(
            output_tokens,
            tg_times,
            alpha=0.6,
            s=100,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidth=0.5,
            label=device_label,
            zorder=3,
        )

        # Add trend line for this device
        if len(output_tokens) > 1:
            z = np.polyfit(output_tokens, tg_times, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(output_tokens.min(), output_tokens.max(), 100)
            ax.plot(
                x_trend,
                p(x_trend),
                "--",
                color=color,
                alpha=0.3,
                linewidth=1.5,
                zorder=2,
            )

    if not has_data:
        print("No data available for Generation Time vs Output Tokens plot")
        return

    # Styling
    ax.set_xlabel("Output Tokens (response length)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Text Generation Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Text Generation Performance\nGeneration Time vs Output Length",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Legend
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=10,
    )

    # Start from 0
    ax.set_ylim(bottom=0)

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
        data = load_prompt_details(str(result_file))

        device_label = data["device_label"]
        gpu_name = data["gpu_name"]
        prompt_details = data["prompt_details"]

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
