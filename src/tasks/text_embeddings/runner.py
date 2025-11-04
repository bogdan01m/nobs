from src.data.imdb_data import dataset
from src.settings import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE
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

    # Get model name from settings
    model_name = EMBEDDING_MODEL_NAME
    model_key = model_name.split("/")[-1]  # Extract short name from full path

    # Run benchmark
    results["models"][model_name] = run_model_with_repeats(
        model_name=model_name,
        model_key=model_key,
        texts=texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        num_runs=3,
    )

    # Calculate total time as main metric
    total_median_time = results["models"][model_name]["median_encoding_time_seconds"]
    results["total_time_seconds"] = round(total_median_time, 2)

    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {results['total_time_seconds']} seconds")
    print(f"{'='*60}")

    return results
