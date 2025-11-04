from src.data.awesome_prompts import dataset
from .executor import run_model_with_repeats


def run_llms_benchmark(model_name: str | None = None, base_url: str | None = None):
    """
    Runs full LLM benchmark using awesome prompts

    Args:
        model_name: Model name to use (defaults to LLM_MODEL_NAME from settings)
        base_url: API base URL (defaults to LLM_BASE_URL from settings)

    Returns:
        dict: Results with median latency as main metric
    """
    # Use provided model_name or fallback to settings

    # Load 10 prompts from awesome prompts dataset
    prompts = dataset["prompt"]

    print("Dataset loaded")
    print(f"Number of prompts: {len(prompts)}\n")

    # Run benchmark with 3 repeats per prompt
    model_results = run_model_with_repeats(
        model_name=model_name, prompts=prompts, num_runs=3, base_url=base_url
    )

    # Use median E2E latency as main metric
    total_time = model_results["final_50p_e2e_latency_s"]

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
