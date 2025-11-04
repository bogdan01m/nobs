from src.data.hallucination_coco import dataset
from .executor import run_model_with_repeats


def run_vlms_benchmark(model_name: str | None = None, base_url: str | None = None):
    """
    Runs full VLM benchmark using Hallucination COCO dataset

    Args:
        model_name: Model name to use (defaults to VLM_MODEL_NAME from settings)
        base_url: API base URL (defaults to VLM_BASE_URL from settings)

    Returns:
        dict: Results with median latency as main metric
    """
    # Use provided model_name or fallback to settings

    # Load first 3 questions and images from hallucination_coco dataset
    questions = dataset["question"]
    images = dataset["image"]  # PIL Image objects

    print("Dataset loaded")
    print(f"Number of questions: {len(questions)}\n")

    # Run benchmark with 3 repeats per question
    model_results = run_model_with_repeats(
        model_name=model_name,
        prompts=questions,
        images=images,
        num_runs=3,
        base_url=base_url,
    )

    # Use median latency as main metric
    total_time = model_results["final_50p_e2e_latency_s"]

    results = {
        "task": "vlms",
        "dataset": "Hallucination_COCO",
        "num_questions": len(questions),
        "model": model_results,
        "total_time_seconds": round(total_time, 2),
    }

    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {results['total_time_seconds']} seconds")
    print(f"{'='*60}")

    return results
