from src.data.imdb_data import dataset
from .executor import run_model_with_repeats


def run_embeddings_benchmark() -> dict:
    """
    Runs full embeddings models benchmark

    Returns:
        dict: Results for all models with metadata
    """
    texts = dataset["text"]
    print("Dataset loaded")
    print(f"Number of rows: {len(texts)}\n")

    results: dict = {"task": "embeddings", "dataset_size": len(texts), "models": {}}

    # Models for benchmark
    gte_model = "thenlper/gte-large"
    modernbert_model = "nomic-ai/modernbert-embed-base"
    gte_key = "gte-large"
    modernbert_key = "modernbert-embed-base"
    # Run benchmarks with repeats
    results["models"][gte_model] = run_model_with_repeats(
        model_name=gte_model, model_key=gte_key, texts=texts, batch_size=16, num_runs=3
    )

    results["models"][modernbert_model] = run_model_with_repeats(
        model_name=modernbert_model,
        model_key=modernbert_key,
        texts=texts,
        batch_size=16,
        num_runs=3,
    )

    # Calculate total time as main metric
    total_median_time = sum(
        model_result["median_encoding_time_seconds"]
        for model_result in results["models"].values()
    )
    results["total_time_seconds"] = round(total_median_time, 2)

    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {results['total_time_seconds']} seconds")
    print(f"{'='*60}")

    return results
