"""Generate scientific performance profile plot for embeddings benchmark metrics."""

import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from .vendor_color import get_gpu_vendor_color


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
    output_path: Path = Path("results/plots/embeddings_performance.png"),
) -> None:
    """
    Generate performance profile bar chart for embeddings metrics.

    Shows throughput (rows/sec) for each device/model combination.
    Bars represent different devices, sorted by performance (descending).
    Labels include: gpu_name [STS] model_name
    """
    results = load_results(results_dir)

    if not results:
        print("No results found to plot")
        return

    # Extract embeddings metrics: list of devices with their metrics
    devices = []

    for result in results:
        device_info = result["device_info"]
        embeddings_task = next(
            (t for t in result["tasks"] if t["task"] == "embeddings"), None
        )

        if embeddings_task and "models" in embeddings_task:
            for model_name, model_data in embeddings_task["models"].items():
                # Simplify model name for display
                display_model = (
                    model_name.split("/")[-1] if "/" in model_name else model_name
                )

                # Use mean ± std approach: mean(run1, run2, run3) ± std(run1, run2, run3)
                rps = model_data.get("final_mean_rps", 0)
                rps_std = model_data.get("final_std_rps", 0)

                devices.append(
                    {
                        "gpu_name": device_info["gpu_name"],
                        "host": device_info["host"],
                        "backend": "STS",  # Sentence Transformers backend
                        "model_name": display_model,
                        "rps": rps,
                        "rps_std": rps_std,
                    }
                )

    if not devices:
        print("No embeddings results found to plot")
        return

    # Sort by throughput (descending - higher is better)
    devices_sorted = sorted(devices, key=lambda x: x["rps"], reverse=True)

    # Create labels: "gpu_name\n[STS] model_name" (two lines)
    y_labels = [f"{d['gpu_name']}\n[STS] {d['model_name']}" for d in devices_sorted]
    rps_values = [d["rps"] for d in devices_sorted]
    rps_std_values = [d["rps_std"] for d in devices_sorted]

    # Get colors for each GPU based on vendor
    colors = [get_gpu_vendor_color(d["gpu_name"]) for d in devices_sorted]

    # Create figure with scientific style (horizontal bars, dynamic height)
    plt.style.use("seaborn-v0_8-paper")
    n_devices = len(y_labels)
    fig_height = max(6, n_devices * 0.6)  # Dynamic height: min 6, scales with devices
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=300)

    # Y positions
    y_pos = np.arange(len(y_labels))

    # Plot horizontal bars with error bars
    bars = ax.barh(
        y_pos,
        rps_values,
        xerr=rps_std_values,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Styling
    ax.set_ylabel("GPU Device [Backend]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Throughput (rows/second)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Text Embeddings Performance: Throughput [mean ± std]\n(Higher is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()  # Best performer at top
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, rps_values, rps_std_values)):
        width = bar.get_width()
        ax.text(
            width + std + max(rps_values) * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Highlight best performer (first one, since sorted by RPS)
    bars[0].set_edgecolor("green")
    bars[0].set_linewidth(2.5)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Embeddings performance plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_embeddings_performance()
