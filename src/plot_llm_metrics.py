"""Generate scientific performance profile plots for LLM benchmark metrics."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def get_gpu_vendor_color(gpu_name: str) -> str:
    """Determine GPU vendor color for consistent visualization."""
    gpu_lower = gpu_name.lower()
    if any(x in gpu_lower for x in ["apple", "m1", "m2", "m3", "m4"]):
        return "#2E86AB"  # Blue for Apple
    elif any(x in gpu_lower for x in ["nvidia", "rtx", "gtx", "tesla", "a100"]):
        return "#76B900"  # NVIDIA Green
    elif any(x in gpu_lower for x in ["amd", "radeon", "rx"]):
        return "#ED1C24"  # AMD Red
    elif any(x in gpu_lower for x in ["intel", "arc", "uhd"]):
        return "#F77F00"  # Orange for Intel
    else:
        return "#808080"  # Gray for Other


def load_results(results_dir: Path = Path("results")) -> list[dict[str, Any]]:
    """Load all result JSON files from the results directory."""
    results = []
    for json_file in results_dir.glob("report_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    return results


def plot_llm_performance(
    results_dir: Path = Path("results"),
    output_path_ttft: Path = Path("results/llm_ttft.png"),
    output_path_tps: Path = Path("results/llm_tps.png"),
) -> None:
    """
    Generate two scientific performance profile plots for LLM metrics.

    Plot 1: Time To First Token (TTFT) - lower is better
    Plot 2: Tokens Per Second (TPS) - higher is better
    Both sorted by overall performance (TPS)
    """
    results = load_results(results_dir)

    if not results:
        print("No results found to plot")
        return

    # Extract LLM metrics
    devices = []
    for result in results:
        device_info = result["device_info"]
        llm_task = next((t for t in result["tasks"] if t["task"] == "llms"), None)

        if llm_task and "model" in llm_task:
            model_data = llm_task["model"]
            devices.append(
                {
                    "gpu_name": device_info["gpu_name"],
                    "host": device_info["host"],
                    "ttft": model_data["final_50p_ttft_s"],
                    "tps": model_data["final_50p_tokens_per_sec"],
                    "ttft_std": model_data.get("final_std_ttft_s", 0),
                    "tps_std": model_data.get("final_std_tokens_per_sec", 0),
                }
            )

    if not devices:
        print("No LLM results found to plot")
        return

    # Sort by TPS (descending - higher is better)
    devices_sorted = sorted(devices, key=lambda x: x["tps"], reverse=True)

    gpu_names = [d["gpu_name"] for d in devices_sorted]
    ttft_values = [d["ttft"] for d in devices_sorted]
    ttft_std_values = [d["ttft_std"] for d in devices_sorted]
    tps_values = [d["tps"] for d in devices_sorted]
    tps_std_values = [d["tps_std"] for d in devices_sorted]

    # Get colors for each GPU
    colors = [get_gpu_vendor_color(name) for name in gpu_names]

    # Create figure with scientific style
    plt.style.use("seaborn-v0_8-paper")

    # X positions
    x_pos = np.arange(len(gpu_names))

    # ===== PLOT 1: TTFT (lower is better) =====
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot bars with error bars
    bars1 = ax1.bar(
        x_pos,
        ttft_values,
        yerr=ttft_std_values,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Styling
    ax1.set_xlabel("GPU Device (sorted by throughput)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Time To First Token (seconds)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "LLM Inference Performance: Time To First Token\n(Lower is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(gpu_names, rotation=15, ha="right")
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="y")
    ax1.set_axisbelow(True)
    ax1.set_ylim(bottom=0)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, ttft_values, ttft_std_values)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.1,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Highlight best performer
    best_idx = ttft_values.index(min(ttft_values))
    bars1[best_idx].set_edgecolor("green")
    bars1[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    output_path_ttft.parent.mkdir(exist_ok=True)
    plt.savefig(output_path_ttft, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_ttft.with_suffix(".svg"), format="svg", bbox_inches="tight")
    print(f"✅ LLM TTFT plot saved to {output_path_ttft}")
    plt.close()

    # ===== PLOT 2: TPS (higher is better) =====
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot bars with error bars
    bars2 = ax2.bar(
        x_pos,
        tps_values,
        yerr=tps_std_values,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Styling
    ax2.set_xlabel("GPU Device (sorted by throughput)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Throughput (tokens/second)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "LLM Inference Performance: Throughput\n(Higher is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(gpu_names, rotation=15, ha="right")
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="y")
    ax2.set_axisbelow(True)
    ax2.set_ylim(bottom=0)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars2, tps_values, tps_std_values)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Highlight best performer (first one, since sorted by TPS)
    bars2[0].set_edgecolor("green")
    bars2[0].set_linewidth(2.5)

    plt.tight_layout()
    output_path_tps.parent.mkdir(exist_ok=True)
    plt.savefig(output_path_tps, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_tps.with_suffix(".svg"), format="svg", bbox_inches="tight")
    print(f"✅ LLM TPS plot saved to {output_path_tps}")
    plt.close()


if __name__ == "__main__":
    plot_llm_performance()
