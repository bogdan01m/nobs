from src.data.hallucination_coco import dataset
from .executor import run_model_with_repeats
from src.settings import VLM_MODEL_NAME


def run_vlms_benchmark():
    """
    Runs full VLM benchmark using Hallucination COCO dataset

    Returns:
        dict: Results with median latency as main metric
    """
    # Load first 3 questions and images from hallucination_coco dataset
    questions = dataset["question"][:3]
    images = dataset["image"][:3]  # PIL Image objects

    print(f"Dataset loaded: {dataset}")
    print(f"Number of questions: {len(questions)}\n")

    # Run benchmark with 3 repeats per question
    model_results = run_model_with_repeats(
        model_name=VLM_MODEL_NAME, prompts=questions, images=images, num_runs=3
    )

    # Use median latency as main metric
    total_time = model_results["final_50p_latency_s"]

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
