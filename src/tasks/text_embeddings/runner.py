from src.data.imdb_data import dataset
from .executor import run_model_with_repeats
from src.score import calculate_score

def run_embeddings_benchmark():
    """
    Runs full embeddings models benchmark

    Returns:
        dict: Results for all models with metadata
    """
    texts = dataset['text'][:100]
    print(f"Dataset loaded: {dataset}")
    print(f"Number of rows: {len(texts)}\n")

    results = {
        "task": "embeddings",
        "dataset_size": len(texts),
        "models": {}
    }

    # Models for benchmark
    gte_model = "thenlper/gte-large"
    modernbert_model = "nomic-ai/modernbert-embed-base"
    gte_key = "gte-large"
    modernbert_key = "modernbert-embed-base"
    # Run benchmarks with repeats
    results["models"][gte_model] = run_model_with_repeats(
        model_name=gte_model,
        model_key=gte_key,
        texts=texts,
        batch_size=16,
        num_runs=3
    )

    results["models"][modernbert_model] = run_model_with_repeats(
        model_name=modernbert_model,
        model_key=modernbert_key,
        texts=texts,
        batch_size=16,
        num_runs=3
    )

    # Calculate task score using median times
    total_median_time = sum(
        model_result["median_encoding_time_seconds"]
        for model_result in results["models"].values()
    )
    print(total_median_time)
    results["task_score"] = calculate_score(total_median_time, num_tasks=1)

    print(f"\n{'='*60}")
    print(f"FINAL TASK SCORE: {results['task_score']}")
    print(f"{'='*60}")

    return results