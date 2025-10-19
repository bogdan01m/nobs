import time
from sentence_transformers import SentenceTransformer
from src.memory_cleaner import clear_memory
from statistics import median


def run_single_model(
    model_name: str, model_key: str, texts: list, batch_size: int = 16
):
    """
    Loads model, encodes texts, measures time and clears memory

    Args:
        model_name: HuggingFace model name
        model_key: Key for saving results
        texts: List of texts to encode
        batch_size: Batch size

    Returns:
        dict: Benchmark results
    """
    print("=" * 50)
    print(f"{model_key.upper()} MODEL")
    print("=" * 50)

    # Load model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model_dtype = str(next(model.parameters()).dtype)
    print(f"Model loaded, max_seq_length: {model.max_seq_length}")
    print(f"Model device: {model.device}, dtype: {model_dtype}")

    # Encode texts with timing
    print("Encoding texts...")
    start_time = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    encoding_time = time.perf_counter() - start_time

    print(f"Encoding completed in {encoding_time:.2f}s")
    print(f"Speed: {len(texts) / encoding_time:.2f} rows/sec")

    # Results
    results = {
        "model_name": model_name,
        "batch_size": batch_size,
        "max_seq_length": model.max_seq_length,
        "embedding_dimension": embeddings.shape[1],
        "device": str(model.device),
        "dtype": model_dtype,
        "actual_dataset_size": len(texts),
        "encoding_time_seconds": round(encoding_time, 2),
        "rows_per_second": round(len(texts) / encoding_time, 2),
    }

    # Clear memory
    print(f"\nClearing {model_key} model from memory...")
    del model, embeddings
    clear_memory()

    return results


def run_model_with_repeats(
    model_name: str,
    model_key: str,
    texts: list,
    batch_size: int = 16,
    num_runs: int = 3,
):
    """
    Runs model multiple times with warmup run and calculates median

    Args:
        model_name: HuggingFace model name
        model_key: Key for saving results
        texts: List of texts to encode
        batch_size: Batch size for encoding
        num_runs: Number of real runs (after warmup)

    Returns:
        dict: Aggregated results with median values
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*60}")

    # Warmup run (not counted in results)
    print("\n🔥 WARMUP RUN (not counted)")
    _ = run_single_model(
        model_name=model_name,
        model_key=model_key,
        texts=texts[:50],
        batch_size=batch_size,
    )

    # Real runs
    runs = []
    for i in range(num_runs):
        print(f"\n📊 RUN {i+1}/{num_runs}")
        run_result = run_single_model(
            model_name=model_name,
            model_key=model_key,
            texts=texts,
            batch_size=batch_size,
        )
        runs.append(
            {
                "run": i + 1,
                "encoding_time_seconds": run_result["encoding_time_seconds"],
                "rows_per_second": run_result["rows_per_second"],
            }
        )

    # Calculate median
    median_time = median([r["encoding_time_seconds"] for r in runs])
    median_rps = median([r["rows_per_second"] for r in runs])

    # Take metadata from last run (all runs have the same metadata)
    last_run = run_result

    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "max_seq_length": last_run["max_seq_length"],
        "embedding_dimension": last_run["embedding_dimension"],
        "device": last_run["device"],
        "dtype": last_run["dtype"],
        "actual_dataset_size": last_run["actual_dataset_size"],
        "num_runs": num_runs,
        "runs": runs,
        "median_encoding_time_seconds": round(median_time, 2),
        "median_rows_per_second": round(median_rps, 2),
    }
