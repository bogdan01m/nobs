from src.data.awesome_prompts import dataset
from .executor import run_model_with_repeats
from src.settings import LLM_MODEL_NAME
from src.score import calculate_score


def run_llms_benchmark():
    """
    Runs full LLM benchmark using awesome prompts

    Returns:
        dict: Results with median latency and final score
    """
    # Load 10 prompts from awesome prompts dataset
    prompts = dataset["prompt"][:3]

    print(f"Dataset loaded: {dataset}")
    print(f"Number of prompts: {len(prompts)}\n")

    # Run benchmark with 3 repeats per prompt
    model_results = run_model_with_repeats(
        model_name=LLM_MODEL_NAME, prompts=prompts, num_runs=3
    )

    # Calculate task score using the same formula as embeddings
    # Uses median latency across all runs
    task_score = calculate_score(
        total_time=model_results["final_median_latency_s"], num_tasks=1
    )

    results = {
        "task": "llms",
        "dataset": "awesome-chatgpt-prompts",
        "num_prompts": len(prompts),
        "model": model_results,
        "task_score": task_score,
    }

    print(f"\n{'='*60}")
    print(f"FINAL LLM BENCHMARK SCORE: {results['task_score']}")
    print(f"{'='*60}")

    return results
