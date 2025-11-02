"""
Generate synthetic benchmark results for demonstration purposes.

Creates fake benchmark data for multiple devices to showcase multi-device plotting.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from statistics import median, pstdev
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"
SYNTHETIC_FILENAME_PREFIX = "report_synth"
DEFAULT_LLM_DATASET = "awesome-chatgpt-prompts"
DEFAULT_VLM_DATASET = "Hallucination_COCO"
RANDOM_SEED = 20250206

random.seed(RANDOM_SEED)

DEVICES = [
    {
        "platform": "Darwin",
        "device": "mps",
        "ram_gb": 36.0,
        "processor": "Apple M4 Max (14 cores)",
        "gpu_name": "Apple M4 Max (40 cores)",
        "gpu_memory_gb": "shared with system RAM",
        "host": "Mac16,10",
        "ttft_multiplier": 0.78,
        "tg_multiplier": 0.82,
        "vlm_ttft_multiplier": 0.88,
        "vlm_tg_multiplier": 0.9,
        "embedding_multiplier": 0.82,
        "llm_backends": [
            {
                "name": "LM_STUDIO",
                "model_name": "granite-coder-34b",
                "num_prompts": 18,
                "ttft_bias": 0.88,
                "tg_bias": 0.92,
            }
        ],
        "vlm_backends": [
            {
                "name": "LM_STUDIO",
                "model_name": "minicpm-v:8b",
                "num_prompts": 12,
                "ttft_bias": 0.92,
                "tg_bias": 0.95,
            }
        ],
    },
    {
        "platform": "Linux",
        "device": "cuda",
        "ram_gb": 32.0,
        "processor": "Intel Core i7-13700K",
        "gpu_name": "NVIDIA RTX 4060 Ti",
        "gpu_memory_gb": 16,
        "host": "RTX4060Ti-PC",
        "ttft_multiplier": 1.02,
        "tg_multiplier": 1.0,
        "vlm_ttft_multiplier": 1.0,
        "vlm_tg_multiplier": 1.04,
        "embedding_multiplier": 1.0,
        "llm_backends": [
            {
                "name": "OLLAMA",
                "model_name": "llama3:13b-instruct",
                "num_prompts": 16,
                "ttft_bias": 1.0,
                "tg_bias": 1.0,
            }
        ],
        "vlm_backends": [
            {
                "name": "OLLAMA",
                "model_name": "minicpm-v:8b",
                "num_prompts": 9,
                "ttft_bias": 1.05,
                "tg_bias": 1.08,
            }
        ],
    },
    {
        "platform": "Windows",
        "device": "rocm",
        "ram_gb": 64.0,
        "processor": "AMD Ryzen 9 7950X",
        "gpu_name": "AMD Radeon RX 7900 XTX",
        "gpu_memory_gb": 24,
        "host": "Radeon-Workstation",
        "ttft_multiplier": 1.08,
        "tg_multiplier": 1.04,
        "vlm_ttft_multiplier": 1.05,
        "vlm_tg_multiplier": 1.06,
        "embedding_multiplier": 1.05,
        "llm_backends": [
            {
                "name": "LM_STUDIO",
                "model_name": "mixtral-8x7b-instruct",
                "num_prompts": 15,
                "ttft_bias": 0.98,
                "tg_bias": 1.0,
            },
            {
                "name": "OLLAMA",
                "model_name": "mistral:7b-instruct",
                "num_prompts": 12,
                "ttft_bias": 1.12,
                "tg_bias": 1.1,
            },
        ],
        "vlm_backends": [
            {
                "name": "LM_STUDIO",
                "model_name": "llava:13b",
                "num_prompts": 10,
                "ttft_bias": 1.02,
                "tg_bias": 1.08,
            }
        ],
    },
    {
        "platform": "Linux",
        "device": "openvino",
        "ram_gb": 48.0,
        "processor": "Intel Core Ultra 7 165H",
        "gpu_name": "Intel Arc A770",
        "gpu_memory_gb": 16,
        "host": "Arc-A770-Lab",
        "ttft_multiplier": 1.3,
        "tg_multiplier": 1.35,
        "vlm_ttft_multiplier": 1.28,
        "vlm_tg_multiplier": 1.32,
        "embedding_multiplier": 1.25,
        "llm_backends": [
            {
                "name": "OLLAMA",
                "model_name": "phi3:mini-4k",
                "num_prompts": 14,
                "ttft_bias": 1.22,
                "tg_bias": 1.28,
            }
        ],
        "vlm_backends": [],
    },
    {
        "platform": "Darwin",
        "device": "mps",
        "ram_gb": 24.0,
        "processor": "Apple M2 (8 cores)",
        "gpu_name": "Apple M2 (10 cores)",
        "gpu_memory_gb": "shared with system RAM",
        "host": "Mac14,7",
        "ttft_multiplier": 1.38,
        "tg_multiplier": 1.33,
        "vlm_ttft_multiplier": 1.32,
        "vlm_tg_multiplier": 1.28,
        "embedding_multiplier": 1.18,
        "llm_backends": [
            {
                "name": "LM_STUDIO",
                "model_name": "nous-hermes-llama2-13b",
                "num_prompts": 12,
                "ttft_bias": 1.25,
                "tg_bias": 1.3,
            }
        ],
        "vlm_backends": [
            {
                "name": "LM_STUDIO",
                "model_name": "llava:7b",
                "num_prompts": 8,
                "ttft_bias": 1.2,
                "tg_bias": 1.25,
            }
        ],
    },
]

EMBEDDING_MODELS = [
    {
        "name": "text-embedding-3-large",
        "base_rps": 260.0,
        "base_time": 3.4,
        "embedding_dimension": 3072,
        "batch_size": 16,
    },
    {
        "name": "text-embedding-3-small",
        "base_rps": 410.0,
        "base_time": 2.2,
        "embedding_dimension": 1536,
        "batch_size": 32,
    },
    {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "base_rps": 180.0,
        "base_time": 4.6,
        "embedding_dimension": 768,
        "batch_size": 24,
    },
]


def _weighted_random(value: float, variation: float = 0.12) -> float:
    """Apply a small random variation around a base value."""
    return value * random.uniform(1 - variation, 1 + variation)


def _safe_std(values: list[float]) -> float:
    """Return population standard deviation or 0 if not enough samples."""
    return pstdev(values) if len(values) > 1 else 0.0


def _sanitize_host_for_filename(host: str) -> str:
    """Make host names safe to use in filenames."""
    return host.replace(" ", "_").replace(",", "_")


def generate_prompt_details(
    device_config: dict[str, Any],
    num_prompts: int = 15,
    task_type: str = "llms",
    ttft_bias: float = 1.0,
    tg_bias: float = 1.0,
) -> list[dict[str, Any]]:
    """Generate synthetic prompt metrics with device/backend tuning."""
    details: list[dict[str, Any]] = []

    for _ in range(num_prompts):
        if task_type == "vlms":
            input_tokens = random.randint(90, 950)
            output_tokens = random.randint(60, 900)
            base_ttft = 0.8 + (input_tokens / 900) * 3.6
            base_tg = (output_tokens / 95) * 1.05
            ttft_multiplier = (
                device_config.get("vlm_ttft_multiplier", device_config.get("ttft_multiplier", 1.0))
                * ttft_bias
            )
            tg_multiplier = (
                device_config.get("vlm_tg_multiplier", device_config.get("tg_multiplier", 1.0))
                * tg_bias
            )
        else:
            input_tokens = random.randint(40, 900)
            output_tokens = random.randint(80, 1600)
            base_ttft = 0.45 + (input_tokens / 950) * 3.1
            base_tg = (output_tokens / 95) * 0.75
            ttft_multiplier = device_config.get("ttft_multiplier", 1.0) * ttft_bias
            tg_multiplier = device_config.get("tg_multiplier", 1.0) * tg_bias

        ttft_s = base_ttft * ttft_multiplier * random.uniform(0.9, 1.12)
        tg_s = base_tg * tg_multiplier * random.uniform(0.9, 1.15)

        total_latency_s = ttft_s + tg_s
        tokens_per_sec = output_tokens / tg_s if tg_s > 0 else 0

        details.append(
            {
                "ttft_s": round(ttft_s, 4),
                "tg_s": round(tg_s, 4),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "tokens_per_sec": round(tokens_per_sec, 4),
                "total_latency_s": round(total_latency_s, 4),
            }
        )

    return details


def summarize_prompt_details(prompt_details: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate statistics needed by reports and plots."""
    ttft_values = [d["ttft_s"] for d in prompt_details]
    tg_values = [d["tg_s"] for d in prompt_details]
    latency_values = [d["total_latency_s"] for d in prompt_details]
    tps_values = [d["tokens_per_sec"] for d in prompt_details]

    return {
        "ttft_median": round(median(ttft_values), 4),
        "ttft_std": round(_safe_std(ttft_values), 4),
        "tg_median": round(median(tg_values), 4),
        "tg_std": round(_safe_std(tg_values), 4),
        "latency_median": round(median(latency_values), 4),
        "latency_std": round(_safe_std(latency_values), 4),
        "tps_median": round(median(tps_values), 4),
        "tps_std": round(_safe_std(tps_values), 4),
        "total_input_tokens": int(sum(d["input_tokens"] for d in prompt_details)),
        "total_output_tokens": int(sum(d["output_tokens"] for d in prompt_details)),
        "total_time_seconds": round(sum(latency_values), 2),
    }


def generate_runs_from_details(
    prompt_details: list[dict[str, Any]],
    num_runs: int = 3,
) -> list[dict[str, int]]:
    """Create aggregate run summaries to mimic repeated measurements."""
    total_input = sum(d["input_tokens"] for d in prompt_details)
    total_output = sum(d["output_tokens"] for d in prompt_details)
    runs: list[dict[str, int]] = []

    for _ in range(num_runs):
        input_scale = random.uniform(0.92, 1.08)
        output_scale = random.uniform(0.9, 1.12)
        runs.append(
            {
                "total_input_tokens": int(total_input * input_scale),
                "total_output_tokens": int(total_output * output_scale),
            }
        )

    return runs


def generate_embeddings_models(device_config: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Create synthetic embedding metrics for multiple models."""
    models: dict[str, Any] = {}
    total_time = 0.0
    multiplier = device_config.get("embedding_multiplier", 1.0)

    for model in EMBEDDING_MODELS:
        median_rps = _weighted_random(model["base_rps"] / multiplier, variation=0.15)
        median_time = _weighted_random(model["base_time"] * multiplier, variation=0.18)

        std_rps = median_rps * random.uniform(0.04, 0.12)
        std_time = median_time * random.uniform(0.05, 0.15)

        models[model["name"]] = {
            "median_rows_per_second": round(median_rps, 2),
            "std_rows_per_second": round(std_rps, 2),
            "median_encoding_time_seconds": round(median_time, 2),
            "std_encoding_time_seconds": round(std_time, 2),
            "embedding_dimension": model["embedding_dimension"],
            "batch_size": model["batch_size"],
        }

        total_time += median_time

    return models, round(total_time, 2)


def create_synthetic_result(device_config: dict[str, Any]) -> dict[str, Any]:
    """Create a complete synthetic benchmark result for one device."""
    embedding_models, embedding_time = generate_embeddings_models(device_config)

    tasks: list[dict[str, Any]] = [
        {
            "task": "embeddings",
            "dataset": "imdb-100",
            "num_samples": 100,
            "models": embedding_models,
            "total_time_seconds": embedding_time,
        }
    ]

    for backend in device_config.get("llm_backends", []):
        prompt_count = backend.get("num_prompts", 15)
        details = generate_prompt_details(
            device_config,
            num_prompts=prompt_count,
            task_type="llms",
            ttft_bias=backend.get("ttft_bias", 1.0),
            tg_bias=backend.get("tg_bias", 1.0),
        )
        stats = summarize_prompt_details(details)
        runs = generate_runs_from_details(details)

        tasks.append(
            {
                "task": "llms",
                "backend": backend["name"],
                "dataset": backend.get("dataset", DEFAULT_LLM_DATASET),
                "num_prompts": prompt_count,
                "model": {
                    "model_name": backend.get("model_name", "synthetic-llm"),
                    "num_prompts": prompt_count,
                    "num_runs": len(runs),
                    "all_prompt_details": details,
                    "final_50p_latency_s": stats["latency_median"],
                    "final_std_latency_s": stats["latency_std"],
                    "final_50p_ttft_s": stats["ttft_median"],
                    "final_std_ttft_s": stats["ttft_std"],
                    "final_50p_tokens_per_sec": stats["tps_median"],
                    "final_std_tokens_per_sec": stats["tps_std"],
                    "runs": runs,
                },
                "total_time_seconds": stats["total_time_seconds"],
            }
        )

    for backend in device_config.get("vlm_backends", []):
        prompt_count = backend.get("num_prompts", 10)
        details = generate_prompt_details(
            device_config,
            num_prompts=prompt_count,
            task_type="vlms",
            ttft_bias=backend.get("ttft_bias", 1.0),
            tg_bias=backend.get("tg_bias", 1.0),
        )
        stats = summarize_prompt_details(details)
        runs = generate_runs_from_details(details)

        tasks.append(
            {
                "task": "vlms",
                "backend": backend["name"],
                "dataset": backend.get("dataset", DEFAULT_VLM_DATASET),
                "num_prompts": prompt_count,
                "model": {
                    "model_name": backend.get("model_name", "synthetic-vlm"),
                    "num_prompts": prompt_count,
                    "num_runs": len(runs),
                    "all_prompt_details": details,
                    "final_50p_latency_s": stats["latency_median"],
                    "final_std_latency_s": stats["latency_std"],
                    "final_50p_ttft_s": stats["ttft_median"],
                    "final_std_ttft_s": stats["ttft_std"],
                    "final_50p_tokens_per_sec": stats["tps_median"],
                    "final_std_tokens_per_sec": stats["tps_std"],
                    "runs": runs,
                },
                "total_time_seconds": stats["total_time_seconds"],
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "device_info": {
            "platform": device_config["platform"],
            "device": device_config["device"],
            "ram_gb": device_config["ram_gb"],
            "processor": device_config["processor"],
            "gpu_name": device_config["gpu_name"],
            "gpu_memory_gb": device_config["gpu_memory_gb"],
            "host": device_config["host"],
        },
        "tasks": tasks,
    }


def remove_existing_synthetic_reports() -> None:
    """Remove previously generated synthetic reports to avoid stale data."""
    pattern = f"{SYNTHETIC_FILENAME_PREFIX}_*.json"
    for existing in RESULTS_DIR.glob(pattern):
        try:
            existing.unlink()
            print(f"🧹 Removed old synthetic report: {existing.name}")
        except OSError as exc:
            print(f"⚠️  Failed to remove {existing.name}: {exc}")


def main() -> None:
    """Generate synthetic results for all configured devices."""
    print("Generating synthetic benchmark results...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    remove_existing_synthetic_reports()

    for device in DEVICES:
        result = create_synthetic_result(device)

        safe_host = _sanitize_host_for_filename(device["host"])
        filename = f"{SYNTHETIC_FILENAME_PREFIX}_{safe_host}.json"
        output_path = RESULTS_DIR / filename

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        embeddings_task = next(t for t in result["tasks"] if t["task"] == "embeddings")
        llm_tasks = [t for t in result["tasks"] if t["task"] == "llms"]
        vlm_tasks = [t for t in result["tasks"] if t["task"] == "vlms"]

        print(f"\n  Created: {filename}")
        print(f"    Device: {device['gpu_name']} ({device['platform']})")
        print(f"    Embedding models: {len(embeddings_task['models'])}")

    print(f"\nDone! Generated {len(DEVICES)} synthetic result files in {RESULTS_DIR}/")
    print("\nNext steps:")
    print("  1. uv run python src/generate_results_table.py  # refresh README and plots")
    print("  2. open README.md                                # review rendered sections")


if __name__ == "__main__":
    main()
