"""Generate scientific performance profile plots for VLM benchmark metrics."""

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


def plot_vlm_performance(
    results_dir: Path = Path("results"),
    output_path_latency: Path = Path("results/plots/vlm_latency.png"),
    output_path_tps: Path = Path("results/plots/vlm_tps.png"),
    backend_filter: str | None = None,
) -> None:
    """
    Generate two scientific performance profile plots for VLM metrics.

    Plot 1: E2E Latency P50 - lower is better (full request to response time)
    Plot 2: Tokens Per Second (TPS) - higher is better
    Both sorted by overall performance (TPS)
    """
    results = load_results(results_dir)

    if not results:
        scope = f" for backend {backend_filter}" if backend_filter else ""
        print(f"No results found to plot{scope}")
        return

    # Extract VLM metrics
    devices = []
    for result in results:
        device_info = result["device_info"]
        vlm_tasks = [t for t in result["tasks"] if t["task"] == "vlms" and "model" in t]

        for vlm_task in vlm_tasks:
            backend = vlm_task.get("backend", "UNKNOWN")
            if backend_filter and backend != backend_filter:
                continue

            model_data = vlm_task["model"]
            latency = model_data.get("final_p50_e2e_latency_s")
            tps = model_data.get("final_p50_tps")

            if latency is None or tps is None:
                continue

            devices.append(
                {
                    "gpu_name": device_info["gpu_name"],
                    "host": device_info["host"],
                    "backend": backend,
                    "latency": latency,
                    "tps": tps,
                    "latency_std": model_data.get("final_p50_e2e_latency_std_s", 0),
                    "tps_std": model_data.get("final_p50_tps_std", 0),
                }
            )

    if not devices:
        scope = (
            f" for backend {backend_filter.replace('_', ' ')}" if backend_filter else ""
        )
        print(f"No VLM results found to plot{scope}")
        return

    # Sort by TPS (descending - higher is better)
    devices_sorted = sorted(devices, key=lambda x: x["tps"], reverse=True)

    # Create labels: "gpu_name\n[BACKEND]" format
    y_labels = [f"{d['gpu_name']}\n[{d['backend']}]" for d in devices_sorted]
    latency_values = [d["latency"] for d in devices_sorted]
    latency_std_values = [d["latency_std"] for d in devices_sorted]
    tps_values = [d["tps"] for d in devices_sorted]
    tps_std_values = [d["tps_std"] for d in devices_sorted]

    # Get colors for each GPU
    colors = [get_gpu_vendor_color(d["gpu_name"]) for d in devices_sorted]

    # Create figure with scientific style
    plt.style.use("seaborn-v0_8-paper")

    # Y positions and dynamic height
    n_devices = len(y_labels)
    fig_height = max(6, n_devices * 0.6)
    y_pos = np.arange(len(y_labels))
    backend_label = f" ({backend_filter.replace('_', ' ')})" if backend_filter else ""

    # ===== PLOT 1: E2E Latency P50 (lower is better) =====
    fig1, ax1 = plt.subplots(figsize=(12, fig_height), dpi=300)

    # Plot horizontal bars with error bars
    bars1 = ax1.barh(
        y_pos,
        latency_values,
        xerr=latency_std_values,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Styling
    ax1.set_ylabel("GPU Device [Backend]", fontsize=12, fontweight="bold")
    ax1.set_xlabel("End-to-End Latency (seconds)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"VLM Inference Performance: E2E Latency P50 (sec) [mean(P50) ± std(P50)]{backend_label}\n(Lower is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=10)
    ax1.invert_yaxis()  # Best performer at top
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
    ax1.set_axisbelow(True)
    ax1.set_xlim(left=0)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, latency_values, latency_std_values)):
        width = bar.get_width()
        ax1.text(
            width + std + max(latency_values) * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f}s",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Highlight best performer
    best_idx = latency_values.index(min(latency_values))
    bars1[best_idx].set_edgecolor("green")
    bars1[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    output_path_latency.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_latency, dpi=300, bbox_inches="tight")
    scope_msg = (
        f" for backend {backend_filter.replace('_', ' ')}" if backend_filter else ""
    )
    print(f"✅ VLM E2E Latency plot saved to {output_path_latency}{scope_msg}")
    plt.close()

    # ===== PLOT 2: TPS (higher is better) =====
    fig2, ax2 = plt.subplots(figsize=(12, fig_height), dpi=300)

    # Plot horizontal bars with error bars
    bars2 = ax2.barh(
        y_pos,
        tps_values,
        xerr=tps_std_values,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Styling
    ax2.set_ylabel("GPU Device [Backend]", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tokens per second", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"VLM Inference Performance: TPS [mean(P50) ± std(P50)]{backend_label}\n(Higher is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(y_labels, fontsize=10)
    ax2.invert_yaxis()  # Best performer at top
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
    ax2.set_axisbelow(True)
    ax2.set_xlim(left=0)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars2, tps_values, tps_std_values)):
        width = bar.get_width()
        ax2.text(
            width + std + max(tps_values) * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Highlight best performer (first one, since sorted by TPS)
    bars2[0].set_edgecolor("green")
    bars2[0].set_linewidth(2.5)

    plt.tight_layout()
    output_path_tps.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_tps, dpi=300, bbox_inches="tight")
    print(f"✅ VLM TPS plot saved to {output_path_tps}{scope_msg}")
    plt.close()


if __name__ == "__main__":
    plot_vlm_performance()
