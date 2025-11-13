"""Generate efficiency (performance per watt) comparison plot."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from .vendor_color import get_gpu_vendor_color

sns.set_style("dark")


def load_results(results_dir: Path = Path("results")) -> list[dict[str, Any]]:
    """Load all result JSON files from the results directory."""
    results = []
    for json_file in results_dir.glob("report_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    return results


def extract_gpu_watts_p50(result: dict[str, Any]) -> float | None:
    """Extract GPU power consumption P50."""
    power = result.get("power_metrics", {})
    gpu_p50 = power.get("gpu_power_watts_p50")
    if gpu_p50 is None:
        gpu_p50 = power.get("gpu_watts_p50")
    return gpu_p50 if isinstance(gpu_p50, (int, float)) else None


def extract_embeddings_rps(result: dict[str, Any]) -> float | None:
    """Extract embeddings RPS."""
    for task in result.get("tasks", []):
        if task.get("task") == "embeddings":
            models = task.get("models", {})
            for model_data in models.values():
                return model_data.get("final_mean_rps")
    return None


def extract_llm_tps_by_backend(result: dict[str, Any]) -> dict[str, float | None]:
    """Extract LLM TPS by backend."""
    tps_by_backend: dict[str, float | None] = {"LM_STUDIO": None, "OLLAMA": None}
    for task in result.get("tasks", []):
        if task.get("task") == "llms":
            backend = task.get("backend", "").upper()
            if backend not in tps_by_backend:
                continue
            model = task.get("model", {})
            runs = model.get("runs", [])
            if runs:
                tps_values = [run.get("p50_tps") for run in runs if run.get("p50_tps")]
                if tps_values:
                    tps_by_backend[backend] = sum(tps_values) / len(tps_values)
    return tps_by_backend


def extract_vlm_tps_by_backend(result: dict[str, Any]) -> dict[str, float | None]:
    """Extract VLM TPS by backend."""
    tps_by_backend: dict[str, float | None] = {"LM_STUDIO": None, "OLLAMA": None}
    for task in result.get("tasks", []):
        if task.get("task") == "vlms":
            backend = task.get("backend", "").upper()
            if backend not in tps_by_backend:
                continue
            model = task.get("model", {})
            runs = model.get("runs", [])
            if runs:
                tps_values = [run.get("p50_tps") for run in runs if run.get("p50_tps")]
                if tps_values:
                    tps_by_backend[backend] = sum(tps_values) / len(tps_values)
    return tps_by_backend


def plot_efficiency_comparison(
    results_dir: Path = Path("results"),
    output_dir: Path = Path("results/plots"),
) -> None:
    """
    Generate efficiency comparison bar charts (performance per watt).

    Creates 3 separate plots: Embeddings, LLM, and VLM efficiency.
    """
    results = load_results(results_dir)

    if not results:
        print("No results found to plot efficiency")
        return

    # Extract efficiency data organized by device
    device_efficiencies: dict[str, dict[str, float | None]] = {}

    for result in results:
        device_info = result["device_info"]
        gpu_name = device_info["gpu_name"]
        gpu_watts = extract_gpu_watts_p50(result)

        if not gpu_watts:
            continue

        if gpu_name not in device_efficiencies:
            device_efficiencies[gpu_name] = {
                "embeddings": None,
                "llm_lms": None,
                "llm_ollama": None,
                "vlm_lms": None,
                "vlm_ollama": None,
            }

        # Embeddings efficiency
        emb_rps = extract_embeddings_rps(result)
        if emb_rps:
            device_efficiencies[gpu_name]["embeddings"] = float(emb_rps) / gpu_watts

        # LLM efficiency
        llm_tps = extract_llm_tps_by_backend(result)
        if llm_tps["LM_STUDIO"]:
            device_efficiencies[gpu_name]["llm_lms"] = (
                float(llm_tps["LM_STUDIO"]) / gpu_watts
            )
        if llm_tps["OLLAMA"]:
            device_efficiencies[gpu_name]["llm_ollama"] = (
                float(llm_tps["OLLAMA"]) / gpu_watts
            )

        # VLM efficiency
        vlm_tps = extract_vlm_tps_by_backend(result)
        if vlm_tps["LM_STUDIO"]:
            device_efficiencies[gpu_name]["vlm_lms"] = (
                float(vlm_tps["LM_STUDIO"]) / gpu_watts
            )
        if vlm_tps["OLLAMA"]:
            device_efficiencies[gpu_name]["vlm_ollama"] = (
                float(vlm_tps["OLLAMA"]) / gpu_watts
            )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Embeddings Efficiency
    emb_data: list[tuple[str, float]] = [
        (dev, data["embeddings"])
        for dev, data in device_efficiencies.items()
        if data["embeddings"] is not None
    ]
    if emb_data:
        fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
        emb_data_sorted = sorted(emb_data, key=lambda x: x[1], reverse=True)
        devices = [d[0] for d in emb_data_sorted]
        values = [d[1] for d in emb_data_sorted]
        colors = [get_gpu_vendor_color(dev) for dev in devices]

        bars = ax1.barh(
            range(len(devices)),
            values,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_yticks(range(len(devices)))
        ax1.set_yticklabels(devices, fontsize=10)
        ax1.set_xlabel("Efficiency (RPS/W)", fontsize=11, fontweight="bold")
        ax1.set_title(
            "Embeddings Efficiency: Performance per Watt (Higher is Better)",
            fontsize=13,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
        ax1.set_axisbelow(True)
        ax1.invert_yaxis()

        # Add value labels
        max_val = max(values)
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        emb_output = output_dir / "efficiency_embeddings.png"
        plt.savefig(emb_output, dpi=300, bbox_inches="tight")
        print(f"✅ Embeddings efficiency plot saved to {emb_output}")
        plt.close()

    # Plot 2: LLM Efficiency
    llm_data: list[tuple[str, str, float]] = []
    for dev, data in device_efficiencies.items():
        if data["llm_lms"] is not None:
            llm_data.append((dev, "LM Studio", data["llm_lms"]))
        if data["llm_ollama"] is not None:
            llm_data.append((dev, "Ollama", data["llm_ollama"]))

    if llm_data:
        fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=300)
        llm_data_sorted = sorted(llm_data, key=lambda x: x[2], reverse=True)
        labels = [f"{d[0]} ({d[1]})" for d in llm_data_sorted]
        values = [d[2] for d in llm_data_sorted]
        colors = [get_gpu_vendor_color(d[0]) for d in llm_data_sorted]

        bars = ax2.barh(
            range(len(labels)),
            values,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=10)
        ax2.set_xlabel("Efficiency (TPS/W)", fontsize=11, fontweight="bold")
        ax2.set_title(
            "LLM Inference Efficiency: Performance per Watt (Higher is Better)",
            fontsize=13,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
        ax2.set_axisbelow(True)
        ax2.invert_yaxis()

        # Add value labels
        max_val = max(values)
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        llm_output = output_dir / "efficiency_llm.png"
        plt.savefig(llm_output, dpi=300, bbox_inches="tight")
        print(f"✅ LLM efficiency plot saved to {llm_output}")
        plt.close()

    # Plot 3: VLM Efficiency
    vlm_data: list[tuple[str, str, float]] = []
    for dev, data in device_efficiencies.items():
        if data["vlm_lms"] is not None:
            vlm_data.append((dev, "LM Studio", data["vlm_lms"]))
        if data["vlm_ollama"] is not None:
            vlm_data.append((dev, "Ollama", data["vlm_ollama"]))

    if vlm_data:
        fig3, ax3 = plt.subplots(figsize=(12, 8), dpi=300)
        vlm_data_sorted = sorted(vlm_data, key=lambda x: x[2], reverse=True)
        labels = [f"{d[0]} ({d[1]})" for d in vlm_data_sorted]
        values = [d[2] for d in vlm_data_sorted]
        colors = [get_gpu_vendor_color(d[0]) for d in vlm_data_sorted]

        bars = ax3.barh(
            range(len(labels)),
            values,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        ax3.set_yticks(range(len(labels)))
        ax3.set_yticklabels(labels, fontsize=10)
        ax3.set_xlabel("Efficiency (TPS/W)", fontsize=11, fontweight="bold")
        ax3.set_title(
            "VLM Inference Efficiency: Performance per Watt (Higher is Better)",
            fontsize=13,
            fontweight="bold",
        )
        ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="x")
        ax3.set_axisbelow(True)
        ax3.invert_yaxis()

        # Add value labels
        max_val = max(values)
        for bar, val in zip(bars, values):
            ax3.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        vlm_output = output_dir / "efficiency_vlm.png"
        plt.savefig(vlm_output, dpi=300, bbox_inches="tight")
        print(f"✅ VLM efficiency plot saved to {vlm_output}")
        plt.close()


if __name__ == "__main__":
    plot_efficiency_comparison()
