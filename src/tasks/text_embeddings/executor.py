import time
import torch
from sentence_transformers import SentenceTransformer
from src.system_info.memory_cleaner import clear_memory
from src.utils.task_logger import log_embedding_task
from src.utils.metrics import calculate_mean_std
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
    Loads model, encodes texts, measures E2E latency and RPS

    Args:
        model_name: HuggingFace model name
        model_key: Key for saving results
        texts: List of texts to encode
        batch_size: Batch size

    Returns:
        dict: Benchmark results with E2E latency and RPS
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

    # Encode texts (model.encode handles batching internally)
    print("Encoding texts...")
    device_type = str(model.device)

    # Synchronize device before starting timer
    sync_device(device_type)
    start_time = time.perf_counter()

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)

    # Synchronize device before stopping timer
    sync_device(device_type)
    e2e_latency = time.perf_counter() - start_time

    # Calculate RPS
    rps = len(texts) / e2e_latency

    print(f"Encoding completed in {e2e_latency:.2f}s")
    print(f"Speed: {rps:.2f} rows/sec")

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
        "e2e_latency_s": round(e2e_latency, 4),
        "rps": round(rps, 4),
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
    Runs model multiple times for statistical significance

    Args:
        model_name: HuggingFace model name
        model_key: Key for saving results
        texts: List of texts to encode
        batch_size: Batch size for encoding
        num_runs: Number of real runs (after warmup)

    Returns:
        dict: Aggregated results with percentiles across runs
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"Number of texts: {len(texts)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print(f"{'='*60}")

    # Warmup run (not counted in results)
    print("\n🔥 WARMUP RUN (not counted)")
    _ = run_single_model(
        model_name=model_name,
        model_key=model_key,
        texts=texts[:50],
        batch_size=batch_size,
    )

    # Collect metrics from all runs
    all_run_results = []
    all_e2e_latencies = []
    all_rps = []

    for i in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}/{num_runs}")
        print(f"{'='*60}")

        run_result = run_single_model(
            model_name=model_name,
            model_key=model_key,
            texts=texts,
            batch_size=batch_size,
        )

        all_run_results.append(
            {
                "run": i + 1,
                "e2e_latency_s": run_result["e2e_latency_s"],
                "rps": run_result["rps"],
            }
        )

        # Collect metrics across runs
        all_e2e_latencies.append(run_result["e2e_latency_s"])
        all_rps.append(run_result["rps"])

    # Calculate mean ± std for metrics across runs (per docs/metrics.md)
    # For embeddings: mean(run1, run2, run3) ± std(run1, run2, run3)
    final_mean_latency, final_std_latency = calculate_mean_std(all_e2e_latencies)
    final_mean_rps, final_std_rps = calculate_mean_std(all_rps)

    # Take metadata from last run (all runs have the same metadata)
    last_run = run_result

    results = {
        "model_name": model_name,
        "batch_size": batch_size,
        "max_seq_length": last_run["max_seq_length"],
        "embedding_dimension": last_run["embedding_dimension"],
        "device": last_run["device"],
        "dtype": last_run["dtype"],
        "actual_dataset_size": last_run["actual_dataset_size"],
        "num_runs": num_runs,
        "runs": all_run_results,
        # Mean ± std across runs: mean(run1, run2, run3) ± std(run1, run2, run3)
        "final_mean_e2e_latency_s": round(final_mean_latency, 4)
        if final_mean_latency
        else None,
        "final_std_e2e_latency_s": round(final_std_latency, 4)
        if final_std_latency
        else None,
        "final_mean_rps": round(final_mean_rps, 4) if final_mean_rps else None,
        "final_std_rps": round(final_std_rps, 4) if final_std_rps else None,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(
        f"Final Mean E2E Latency: {results['final_mean_e2e_latency_s']:.4f} ± {results['final_std_e2e_latency_s']:.4f}s"
    )
    print(
        f"Final Mean RPS: {results['final_mean_rps']:.4f} ± {results['final_std_rps']:.4f}"
    )
    print(f"{'='*60}")

    return results
