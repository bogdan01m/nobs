from .chat_stream import stream_with_results
from src.system_info.memory_cleaner import clear_memory
from src.utils.metrics import calculate_percentiles, calculate_mean_std
from tqdm import tqdm


def run_single_model(
    prompts: list,
    images: list | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
):
    """
    Runs a list of prompts through the VLM and collects metrics

    Args:
        prompts: List of prompts to send to the VLM
        images: Optional list of PIL Images (one per prompt). If None, runs text-only.
        model_name: Model name to use (passed to chat_stream)
        base_url: API base URL (passed to chat_stream)

    Returns:
        dict: Aggregated benchmark results including latency and token metrics,
              plus detailed per-prompt metrics
    """
    all_latencies = []
    all_ttft = []
    all_tg = []
    all_tokens_per_sec = []
    total_input_tokens = 0
    total_output_tokens = 0
    per_prompt_details = []

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
        result = stream_with_results(
            prompt, image=image, model_name=model_name, base_url=base_url
        )

        # Collect metrics
        all_latencies.append(result["total_latency_s"])
        if result.get("ttft_s"):
            all_ttft.append(result["ttft_s"])
        if result.get("tg_s"):
            all_tg.append(result["tg_s"])
        if result.get("tokens_per_sec"):
            all_tokens_per_sec.append(result["tokens_per_sec"])
        if result.get("input_tokens"):
            total_input_tokens += result["input_tokens"]
        if result.get("output_tokens"):
            total_output_tokens += result["output_tokens"]

        # Store detailed metrics for this prompt
        per_prompt_details.append(
            {
                "ttft_s": result.get("ttft_s"),
                "tg_s": result.get("tg_s"),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "tokens_per_sec": result.get("tokens_per_sec"),
                "total_latency_s": result["total_latency_s"],
            }
        )

    # Calculate percentiles (P25, P50, P75, P95) for this run according to docs/metrics.md
    p25_lat, p50_lat, p75_lat, p95_lat = calculate_percentiles(all_latencies)
    p25_ttft, p50_ttft, p75_ttft, p95_ttft = calculate_percentiles(all_ttft)
    p25_tg, p50_tg, p75_tg, p95_tg = calculate_percentiles(all_tg)
    p25_tps, p50_tps, p75_tps, p95_tps = calculate_percentiles(all_tokens_per_sec)

    return {
        "p25_latency_s": round(p25_lat, 4) if p25_lat is not None else None,
        "p50_latency_s": round(p50_lat, 4) if p50_lat is not None else None,
        "p75_latency_s": round(p75_lat, 4) if p75_lat is not None else None,
        "p95_latency_s": round(p95_lat, 4) if p95_lat is not None else None,
        "p25_ttft_s": round(p25_ttft, 4) if p25_ttft is not None else None,
        "p50_ttft_s": round(p50_ttft, 4) if p50_ttft is not None else None,
        "p75_ttft_s": round(p75_ttft, 4) if p75_ttft is not None else None,
        "p95_ttft_s": round(p95_ttft, 4) if p95_ttft is not None else None,
        "p25_tg_s": round(p25_tg, 4) if p25_tg is not None else None,
        "p50_tg_s": round(p50_tg, 4) if p50_tg is not None else None,
        "p75_tg_s": round(p75_tg, 4) if p75_tg is not None else None,
        "p95_tg_s": round(p95_tg, 4) if p95_tg is not None else None,
        "p25_tps": round(p25_tps, 4) if p25_tps is not None else None,
        "p50_tps": round(p50_tps, 4) if p50_tps is not None else None,
        "p75_tps": round(p75_tps, 4) if p75_tps is not None else None,
        "p95_tps": round(p95_tps, 4) if p95_tps is not None else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "num_prompts": len(prompts),
        "per_prompt_details": per_prompt_details,
    }


def run_model_with_repeats(
    model_name: str | None,
    prompts: list,
    images: list | None = None,
    num_runs: int = 3,
    base_url: str | None = None,
):
    """
    Runs model multiple times for statistical significance

    Args:
        model_name: VLM model name
        prompts: List of prompts to run
        images: Optional list of PIL Images (one per prompt). If None, runs text-only.
        num_runs: Number of times to repeat the full prompt set (default 3)
        base_url: API base URL (optional, defaults to settings)

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
    _ = run_single_model(
        prompts=[prompts[0]],
        images=warmup_images,
        model_name=model_name,
        base_url=base_url,
    )
    clear_memory()

    # Collect metrics from all runs
    all_run_results = []
    all_prompt_details = []

    # Collect percentile metrics across runs for mean ± std calculation
    p25_latencies, p50_latencies, p75_latencies, p95_latencies = [], [], [], []
    p25_ttfts, p50_ttfts, p75_ttfts, p95_ttfts = [], [], [], []
    p25_tgs, p50_tgs, p75_tgs, p95_tgs = [], [], [], []
    p25_tpss, p50_tpss, p75_tpss, p95_tpss = [], [], [], []

    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")

        result = run_single_model(
            prompts=prompts, images=images, model_name=model_name, base_url=base_url
        )

        all_run_results.append(
            {
                "run": run_idx + 1,
                "p25_latency_s": result["p25_latency_s"],
                "p50_latency_s": result["p50_latency_s"],
                "p75_latency_s": result["p75_latency_s"],
                "p95_latency_s": result["p95_latency_s"],
                "p25_ttft_s": result["p25_ttft_s"],
                "p50_ttft_s": result["p50_ttft_s"],
                "p75_ttft_s": result["p75_ttft_s"],
                "p95_ttft_s": result["p95_ttft_s"],
                "p25_tg_s": result["p25_tg_s"],
                "p50_tg_s": result["p50_tg_s"],
                "p75_tg_s": result["p75_tg_s"],
                "p95_tg_s": result["p95_tg_s"],
                "p25_tps": result["p25_tps"],
                "p50_tps": result["p50_tps"],
                "p75_tps": result["p75_tps"],
                "p95_tps": result["p95_tps"],
                "total_input_tokens": result["total_input_tokens"],
                "total_output_tokens": result["total_output_tokens"],
                "per_prompt_details": result["per_prompt_details"],
            }
        )

        # Collect percentiles from each run for cross-run statistics (docs/metrics.md)
        p25_latencies.append(result["p25_latency_s"])
        p50_latencies.append(result["p50_latency_s"])
        p75_latencies.append(result["p75_latency_s"])
        p95_latencies.append(result["p95_latency_s"])

        if result["p25_ttft_s"]:
            p25_ttfts.append(result["p25_ttft_s"])
        if result["p50_ttft_s"]:
            p50_ttfts.append(result["p50_ttft_s"])
        if result["p75_ttft_s"]:
            p75_ttfts.append(result["p75_ttft_s"])
        if result["p95_ttft_s"]:
            p95_ttfts.append(result["p95_ttft_s"])

        if result["p25_tg_s"]:
            p25_tgs.append(result["p25_tg_s"])
        if result["p50_tg_s"]:
            p50_tgs.append(result["p50_tg_s"])
        if result["p75_tg_s"]:
            p75_tgs.append(result["p75_tg_s"])
        if result["p95_tg_s"]:
            p95_tgs.append(result["p95_tg_s"])

        if result["p25_tps"]:
            p25_tpss.append(result["p25_tps"])
        if result["p50_tps"]:
            p50_tpss.append(result["p50_tps"])
        if result["p75_tps"]:
            p75_tpss.append(result["p75_tps"])
        if result["p95_tps"]:
            p95_tpss.append(result["p95_tps"])

        # Collect all prompt details from all runs
        all_prompt_details.extend(result["per_prompt_details"])

        clear_memory()

    # Calculate mean ± std for each percentile metric across runs (per docs/metrics.md)
    # Latency metrics
    mean_p25_lat, std_p25_lat = calculate_mean_std(p25_latencies)
    mean_p50_lat, std_p50_lat = calculate_mean_std(p50_latencies)
    mean_p75_lat, std_p75_lat = calculate_mean_std(p75_latencies)
    mean_p95_lat, std_p95_lat = calculate_mean_std(p95_latencies)

    # TTFT metrics
    mean_p25_ttft, std_p25_ttft = calculate_mean_std(p25_ttfts)
    mean_p50_ttft, std_p50_ttft = calculate_mean_std(p50_ttfts)
    mean_p75_ttft, std_p75_ttft = calculate_mean_std(p75_ttfts)
    mean_p95_ttft, std_p95_ttft = calculate_mean_std(p95_ttfts)

    # TG metrics
    mean_p25_tg, std_p25_tg = calculate_mean_std(p25_tgs)
    mean_p50_tg, std_p50_tg = calculate_mean_std(p50_tgs)
    mean_p75_tg, std_p75_tg = calculate_mean_std(p75_tgs)
    mean_p95_tg, std_p95_tg = calculate_mean_std(p95_tgs)

    # TPS metrics
    mean_p25_tps, std_p25_tps = calculate_mean_std(p25_tpss)
    mean_p50_tps, std_p50_tps = calculate_mean_std(p50_tpss)
    mean_p75_tps, std_p75_tps = calculate_mean_std(p75_tpss)
    mean_p95_tps, std_p95_tps = calculate_mean_std(p95_tpss)

    results = {
        "model_name": model_name,
        "num_prompts": len(prompts),
        "num_runs": num_runs,
        "runs": all_run_results,
        "all_prompt_details": all_prompt_details,
        # E2E Latency
        "final_p25_e2e_latency_s": round(mean_p25_lat, 4) if mean_p25_lat else None,
        "final_p25_e2e_latency_std_s": round(std_p25_lat, 4) if std_p25_lat else None,
        "final_p50_e2e_latency_s": round(mean_p50_lat, 4) if mean_p50_lat else None,
        "final_p50_e2e_latency_std_s": round(std_p50_lat, 4) if std_p50_lat else None,
        "final_p75_e2e_latency_s": round(mean_p75_lat, 4) if mean_p75_lat else None,
        "final_p75_e2e_latency_std_s": round(std_p75_lat, 4) if std_p75_lat else None,
        "final_p95_e2e_latency_s": round(mean_p95_lat, 4) if mean_p95_lat else None,
        "final_p95_e2e_latency_std_s": round(std_p95_lat, 4) if std_p95_lat else None,
        # TTFT
        "final_p25_ttft_s": round(mean_p25_ttft, 4) if mean_p25_ttft else None,
        "final_p25_ttft_std_s": round(std_p25_ttft, 4) if std_p25_ttft else None,
        "final_p50_ttft_s": round(mean_p50_ttft, 4) if mean_p50_ttft else None,
        "final_p50_ttft_std_s": round(std_p50_ttft, 4) if std_p50_ttft else None,
        "final_p75_ttft_s": round(mean_p75_ttft, 4) if mean_p75_ttft else None,
        "final_p75_ttft_std_s": round(std_p75_ttft, 4) if std_p75_ttft else None,
        "final_p95_ttft_s": round(mean_p95_ttft, 4) if mean_p95_ttft else None,
        "final_p95_ttft_std_s": round(std_p95_ttft, 4) if std_p95_ttft else None,
        # TG
        "final_p25_tg_s": round(mean_p25_tg, 4) if mean_p25_tg else None,
        "final_p25_tg_std_s": round(std_p25_tg, 4) if std_p25_tg else None,
        "final_p50_tg_s": round(mean_p50_tg, 4) if mean_p50_tg else None,
        "final_p50_tg_std_s": round(std_p50_tg, 4) if std_p50_tg else None,
        "final_p75_tg_s": round(mean_p75_tg, 4) if mean_p75_tg else None,
        "final_p75_tg_std_s": round(std_p75_tg, 4) if std_p75_tg else None,
        "final_p95_tg_s": round(mean_p95_tg, 4) if mean_p95_tg else None,
        "final_p95_tg_std_s": round(std_p95_tg, 4) if std_p95_tg else None,
        # TPS
        "final_p25_tps": round(mean_p25_tps, 4) if mean_p25_tps else None,
        "final_p25_tps_std": round(std_p25_tps, 4) if std_p25_tps else None,
        "final_p50_tps": round(mean_p50_tps, 4) if mean_p50_tps else None,
        "final_p50_tps_std": round(std_p50_tps, 4) if std_p50_tps else None,
        "final_p75_tps": round(mean_p75_tps, 4) if mean_p75_tps else None,
        "final_p75_tps_std": round(std_p75_tps, 4) if std_p75_tps else None,
        "final_p95_tps": round(mean_p95_tps, 4) if mean_p95_tps else None,
        "final_p95_tps_std": round(std_p95_tps, 4) if std_p95_tps else None,
    }

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(
        f"Final Mean P50 E2E Latency: {results['final_p50_e2e_latency_s']:.4f} ± {results['final_p50_e2e_latency_std_s']:.4f}s"
    )
    print(
        f"Final Mean P50 TTFT: {results['final_p50_ttft_s']:.4f} ± {results['final_p50_ttft_std_s']:.4f}s"
        if results["final_p50_ttft_s"]
        else "Final Mean P50 TTFT: N/A"
    )
    print(
        f"Final Mean P50 TG: {results['final_p50_tg_s']:.4f} ± {results['final_p50_tg_std_s']:.4f}s"
        if results["final_p50_tg_s"]
        else "Final Mean P50 TG: N/A"
    )
    print(
        f"Final Mean P50 TPS: {results['final_p50_tps']:.4f} ± {results['final_p50_tps_std']:.4f}"
        if results["final_p50_tps"]
        else "Final Mean P50 TPS: N/A"
    )
    print(f"{'='*60}")

    return results
