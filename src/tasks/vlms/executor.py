from .chat_stream import stream_with_results
from src.memory_cleaner import clear_memory
from statistics import median, stdev
from tqdm import tqdm


def run_single_model(prompts: list, images: list | None = None):
    """
    Runs a list of prompts through the VLM and collects metrics

    Args:
        prompts: List of prompts to send to the VLM
        images: Optional list of PIL Images (one per prompt). If None, runs text-only.

    Returns:
        dict: Aggregated benchmark results including latency and token metrics
    """
    all_latencies = []
    all_ttft = []
    all_tokens_per_sec = []
    total_input_tokens = 0
    total_output_tokens = 0

    # If images is None or empty, run text-only
    if not images:
        images = [None] * len(prompts)

    # Ensure images matches prompts length
    if len(images) != len(prompts):
        raise ValueError(
            f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})"
        )

    for prompt, image in tqdm(
        zip(prompts, images), desc="Running prompts", unit="prompt", total=len(prompts)
    ):
        # Run the model with streaming (with or without image)
        result = stream_with_results(prompt, image=image)

        # Collect metrics
        all_latencies.append(result["total_latency_s"])
        if result.get("ttft_s"):
            all_ttft.append(result["ttft_s"])
        if result.get("tokens_per_sec"):
            all_tokens_per_sec.append(result["tokens_per_sec"])
        if result.get("input_tokens"):
            total_input_tokens += result["input_tokens"]
        if result.get("output_tokens"):
            total_output_tokens += result["output_tokens"]

    # Calculate median metrics for this run
    median_latency = median(all_latencies)
    median_ttft = median(all_ttft) if all_ttft else None
    median_tokens_per_sec = median(all_tokens_per_sec) if all_tokens_per_sec else None

    return {
        "median_latency_s": round(median_latency, 4),
        "median_ttft_s": round(median_ttft, 4) if median_ttft else None,
        "median_tokens_per_sec": round(median_tokens_per_sec, 4)
        if median_tokens_per_sec
        else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "num_prompts": len(prompts),
    }


def run_model_with_repeats(
    model_name: str, prompts: list, images: list | None = None, num_runs: int = 3
):
    """
    Runs model multiple times for statistical significance

    Args:
        model_name: VLM model name
        prompts: List of prompts to run
        images: Optional list of PIL Images (one per prompt). If None, runs text-only.
        num_runs: Number of times to repeat the full prompt set (default 3)

    Returns:
        dict: Aggregated results with median latency across all runs
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"With images: {images is not None and len(images) > 0}")
    print(f"Number of runs: {num_runs}")
    print(f"{'='*60}")

    # Warmup run (not counted in results)
    print("\n🔥 WARMUP RUN (not counted)")
    warmup_images = [images[0]] if images else None
    _ = run_single_model(prompts=[prompts[0]], images=warmup_images)
    clear_memory()

    # Collect metrics from all runs
    all_run_results = []
    all_latencies = []
    all_ttft = []
    all_tokens_per_sec = []

    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")

        result = run_single_model(prompts=prompts, images=images)

        all_run_results.append(
            {
                "run": run_idx + 1,
                "median_latency_s": result["median_latency_s"],
                "median_ttft_s": result["median_ttft_s"],
                "median_tokens_per_sec": result["median_tokens_per_sec"],
                "total_input_tokens": result["total_input_tokens"],
                "total_output_tokens": result["total_output_tokens"],
            }
        )

        # Collect for overall median calculation
        all_latencies.append(result["median_latency_s"])
        if result["median_ttft_s"]:
            all_ttft.append(result["median_ttft_s"])
        if result["median_tokens_per_sec"]:
            all_tokens_per_sec.append(result["median_tokens_per_sec"])

        clear_memory()

    # Calculate overall median metrics and std deviation across all runs
    final_median_latency = median(all_latencies)
    final_std_latency = stdev(all_latencies) if len(all_latencies) > 1 else 0.0

    final_median_ttft = median(all_ttft) if all_ttft else None
    final_std_ttft = stdev(all_ttft) if all_ttft and len(all_ttft) > 1 else 0.0

    final_median_tokens_per_sec = (
        median(all_tokens_per_sec) if all_tokens_per_sec else None
    )
    final_std_tokens_per_sec = (
        stdev(all_tokens_per_sec)
        if all_tokens_per_sec and len(all_tokens_per_sec) > 1
        else 0.0
    )

    results = {
        "model_name": model_name,
        "num_prompts": len(prompts),
        "num_runs": num_runs,
        "runs": all_run_results,
        "final_50p_latency_s": round(final_median_latency, 4),
        "final_std_latency_s": round(final_std_latency, 4),
        "final_50p_ttft_s": round(final_median_ttft, 4) if final_median_ttft else None,
        "final_std_ttft_s": round(final_std_ttft, 4) if all_ttft else None,
        "final_50p_tokens_per_sec": round(final_median_tokens_per_sec, 4)
        if final_median_tokens_per_sec
        else None,
        "final_std_tokens_per_sec": round(final_std_tokens_per_sec, 4)
        if all_tokens_per_sec
        else None,
    }

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(
        f"Final Median Latency: {results['final_50p_latency_s']:.4f} ± {results['final_std_latency_s']:.4f}s"
    )
    print(
        f"Final Median TTFT: {results['final_50p_ttft_s']:.4f} ± {results['final_std_ttft_s']:.4f}s"
        if results["final_50p_ttft_s"]
        else "Final Median TTFT: N/A"
    )
    print(
        f"Final Median Tokens/sec: {results['final_50p_tokens_per_sec']:.4f} ± {results['final_std_tokens_per_sec']:.4f}"
        if results["final_50p_tokens_per_sec"]
        else "Final Median Tokens/sec: N/A"
    )
    print(f"{'='*60}")

    return results
