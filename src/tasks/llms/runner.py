from src.data.awesome_prompts import dataset
from .executor import run_model_with_repeats
from src.settings import VLM_MODEL_NAME


def run_llms_benchmark():
    """
    Runs full LLM benchmark using awesome prompts

    Returns:
        dict: Results with median latency as main metric
    """
    # Load 10 prompts from awesome prompts dataset
    prompts = dataset["prompt"][:3]

    print(f"Dataset loaded: {dataset}")
    print(f"Number of prompts: {len(prompts)}\n")

    # Run benchmark with 3 repeats per prompt
    model_results = run_model_with_repeats(
        model_name=VLM_MODEL_NAME, prompts=prompts, num_runs=3
    )

    # Use median latency as main metric
    total_time = model_results["final_median_latency_s"]

    results = {
        "task": "llms",
        "dataset": "awesome-chatgpt-prompts",
        "num_prompts": len(prompts),
        "model": model_results,
        "total_time_seconds": round(total_time, 2),
    }

    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {results['total_time_seconds']} seconds")
    print(f"{'='*60}")

    return results
