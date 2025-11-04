"""Generate scientific performance profile plots for LLM benchmark metrics."""

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


def plot_llm_performance(
    results_dir: Path = Path("results"),
    output_path_ttft: Path = Path("results/plots/llm_ttft.png"),
    output_path_tps: Path = Path("results/plots/llm_tps.png"),
    backend_filter: str | None = None,
) -> None:
    """
    Generate two scientific performance profile plots for LLM metrics.

    Plot 1: Time To First Token (TTFT) - lower is better
    Plot 2: Tokens Per Second (TPS) - higher is better
    Both sorted by overall performance (TPS)
    """
    results = load_results(results_dir)

    if not results:
        scope = f" for backend {backend_filter}" if backend_filter else ""
        print(f"No results found to plot{scope}")
        return

    # Extract LLM metrics
    devices = []
    for result in results:
        device_info = result["device_info"]
        llm_tasks = [t for t in result["tasks"] if t["task"] == "llms" and "model" in t]

        for llm_task in llm_tasks:
            backend = llm_task.get("backend", "UNKNOWN")
            if backend_filter and backend != backend_filter:
                continue

            model_data = llm_task["model"]
            ttft = model_data.get("final_50p_ttft_s")
            # Try new key first, fallback to old for compatibility
            tps = model_data.get("final_50p_e2e_tps") or model_data.get(
                "final_50p_tokens_per_sec"
            )

            if ttft is None or tps is None:
                continue

            devices.append(
                {
                    "gpu_name": device_info["gpu_name"],
                    "host": device_info["host"],
                    "backend": backend,
                    "ttft": ttft,
                    "tps": tps,
                    "ttft_std": model_data.get("final_std_ttft_s", 0),
                    "tps_std": model_data.get("final_std_e2e_tps")
                    or model_data.get("final_std_tokens_per_sec", 0),
                }
            )

    if not devices:
        scope = (
            f" for backend {backend_filter.replace('_', ' ')}" if backend_filter else ""
        )
        print(f"No LLM results found to plot{scope}")
        return

    # Sort by TPS (descending - higher is better)
    devices_sorted = sorted(devices, key=lambda x: x["tps"], reverse=True)

    x_labels = [
        f"{d['host']} [{d['backend']}]" if backend_filter is None else d["host"]
        for d in devices_sorted
    ]
    ttft_values = [d["ttft"] for d in devices_sorted]
    ttft_std_values = [d["ttft_std"] for d in devices_sorted]
    tps_values = [d["tps"] for d in devices_sorted]
    tps_std_values = [d["tps_std"] for d in devices_sorted]

    # Get colors for each GPU
    colors = [get_gpu_vendor_color(d["gpu_name"]) for d in devices_sorted]

    # Create figure with scientific style
    plt.style.use("seaborn-v0_8-paper")

    # X positions
    x_pos = np.arange(len(x_labels))
    backend_label = f" ({backend_filter.replace('_', ' ')})" if backend_filter else ""

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
        f"LLM Inference Performance: Time To First Token{backend_label}\n(Lower is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=15, ha="right")
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
    output_path_ttft.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_ttft, dpi=300, bbox_inches="tight")
    scope_msg = (
        f" for backend {backend_filter.replace('_', ' ')}" if backend_filter else ""
    )
    print(f"✅ LLM TTFT plot saved to {output_path_ttft}{scope_msg}")
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
        f"LLM Inference Performance: Throughput{backend_label}\n(Higher is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=15, ha="right")
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
    output_path_tps.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_tps, dpi=300, bbox_inches="tight")
    print(f"✅ LLM TPS plot saved to {output_path_tps}{scope_msg}")
    plt.close()


if __name__ == "__main__":
    plot_llm_performance()
