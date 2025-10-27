"""Generate scientific performance profile plot for embeddings benchmark metrics."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path = Path("results")) -> list[dict[str, Any]]:
    """Load all result JSON files from the results directory."""
    results = []
    for json_file in results_dir.glob("report_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    return results


def plot_embeddings_performance(
    results_dir: Path = Path("results"),
    output_path: Path = Path("results/embeddings_performance.png"),
) -> None:
    """
    Generate performance profile plot for embeddings metrics.

    Shows throughput (rows/sec) for each model across different GPUs.
    Lines represent different models, sorted by average performance.
    """
    results = load_results(results_dir)

    if not results:
        print("No results found to plot")
        return

    # Extract embeddings metrics
    devices_data: dict[str, dict[str, dict[str, float]]] = {}
    all_models = set()

    for result in results:
        device_info = result["device_info"]
        embeddings_task = next(
            (t for t in result["tasks"] if t["task"] == "embeddings"), None
        )

        if embeddings_task and "models" in embeddings_task:
            gpu_name = device_info["gpu_name"]
            devices_data[gpu_name] = {}

            for model_name, model_data in embeddings_task["models"].items():
                all_models.add(model_name)
                devices_data[gpu_name][model_name] = {
                    "rps": model_data["median_rows_per_second"],
                    "rps_std": model_data.get("std_rows_per_second", 0),
                }

    if not devices_data:
        print("No embeddings results found to plot")
        return

    # Calculate average performance for sorting
    gpu_avg_perf = {}
    for gpu, models in devices_data.items():
        avg = np.mean([m["rps"] for m in models.values()])
        gpu_avg_perf[gpu] = avg

    # Sort GPUs by average performance (descending)
    sorted_gpus = sorted(
        gpu_avg_perf.keys(), key=lambda x: gpu_avg_perf[x], reverse=True
    )

    # Sort models alphabetically for consistent colors
    sorted_models = sorted(all_models)

    # Create figure with scientific style
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # X positions
    x_pos = np.arange(len(sorted_gpus))

    # Color palette for models
    colors = ["#2E86AB", "#A23B72", "#F77F00", "#06A77D"]
    markers = ["o", "s", "^", "D"]

    # Plot each model
    for idx, model_name in enumerate(sorted_models):
        rps_values = []
        rps_std_values = []

        for gpu in sorted_gpus:
            if model_name in devices_data[gpu]:
                rps_values.append(devices_data[gpu][model_name]["rps"])
                rps_std_values.append(devices_data[gpu][model_name]["rps_std"])
            else:
                rps_values.append(0)
                rps_std_values.append(0)

        # Simplify model name for legend
        display_name = model_name.split("/")[-1] if "/" in model_name else model_name

        # Plot line with error bars
        ax.errorbar(
            x_pos,
            rps_values,
            yerr=rps_std_values,
            fmt=f"{markers[idx % len(markers)]}-",
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            label=display_name,
            color=colors[idx % len(colors)],
            ecolor=colors[idx % len(colors)],
            alpha=0.9,
        )

    # Styling
    ax.set_xlabel("GPU Device (sorted by performance)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (rows/second)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Text Embeddings Performance Profile\nThroughput Comparison Across Hardware",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_gpus, rotation=20, ha="right")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="y")
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=10,
        title="Model",
    )

    # Y-axis starts from 0 for better comparison
    ax.set_ylim(bottom=0)

    # Add median line
    all_rps = []
    for gpu_models in devices_data.values():
        for model_data in gpu_models.values():
            all_rps.append(model_data["rps"])

    if all_rps:
        median_rps = np.median(all_rps)
        ax.axhline(
            median_rps,
            color="gray",
            linestyle=":",
            alpha=0.5,
            linewidth=1,
            zorder=0,
        )
        # Add median label
        ax.text(
            len(sorted_gpus) - 0.3,
            median_rps + 1,
            f"Median: {median_rps:.1f}",
            fontsize=9,
            color="gray",
            style="italic",
        )

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    print(f"✅ Embeddings performance plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_embeddings_performance()
