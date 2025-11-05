import time
import torch
from sentence_transformers import SentenceTransformer
from src.system_info.memory_cleaner import clear_memory
from src.utils.task_logger import log_embedding_task
from statistics import median, stdev
from src.settings import EMBEDDING_MAX_LEN


def sync_device(device_type: str):
    """
    Synchronize device operations to ensure accurate timing measurements.

    Args:
        device_type: Device type string (e.g., "cuda", "mps", "cpu")
    """
    if device_type.startswith("cuda"):
        torch.cuda.synchronize()
    elif device_type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    # CPU doesn't need synchronization


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
    model = SentenceTransformer(
        model_name,
        tokenizer_kwargs={"max_seq_length": EMBEDDING_MAX_LEN},
        model_kwargs={"dtype": torch.float16},
        trust_remote_code=True,
    )
    model_dtype = str(next(model.parameters()).dtype)
    print(f"Model loaded, max_seq_length: {model.max_seq_length}")
    print(f"Model device: {model.device}, dtype: {model_dtype}")

    # Encode texts with timing
    print("Encoding texts...")
    device_type = str(model.device)

    # Synchronize device before starting timer
    sync_device(device_type)
    start_time = time.perf_counter()

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)

    # Synchronize device before stopping timer
    sync_device(device_type)
    encoding_time = time.perf_counter() - start_time

    print(f"Encoding completed in {encoding_time:.2f}s")
    print(f"Speed: {len(texts) / encoding_time:.2f} rows/sec")

    # Log first embedding sample
    if len(texts) > 0 and len(embeddings) > 0:
        log_embedding_task(
            text=texts[0],
            embedding_preview=embeddings[0].tolist(),
            shape=embeddings.shape,
        )

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

    # Calculate median and std deviation
    encoding_times = [r["encoding_time_seconds"] for r in runs]
    rows_per_sec = [r["rows_per_second"] for r in runs]

    median_time = median(encoding_times)
    std_time = stdev(encoding_times) if len(encoding_times) > 1 else 0.0

    median_rps = median(rows_per_sec)
    std_rps = stdev(rows_per_sec) if len(rows_per_sec) > 1 else 0.0

    # Take metadata from last run (all runs have the same metadata)
    last_run = run_result

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Median Encoding Time: {median_time:.2f} ± {std_time:.2f}s")
    print(f"Median Rows/sec: {median_rps:.2f} ± {std_rps:.2f}")
    print(f"{'='*60}")

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
        "std_encoding_time_seconds": round(std_time, 2),
        "median_rows_per_second": round(median_rps, 2),
        "std_rows_per_second": round(std_rps, 2),
    }
