import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.system_info.device_info import get_device_info
from src.system_info.power_metrics import PowerMonitor
from src.cli import display_device_info, display_final_summary, select_benchmarks
from src.orchestration import run_all_benchmarks
from src.reporting import save_report


def main():
    """
    Main entry point for NOBS benchmark suite.
    """
    # Get and display device info
    device_info = get_device_info()
    display_device_info(device_info)

    # Interactive menu for benchmark selection
    # All options are pre-checked by default for quick start
    benchmark_config = select_benchmarks(device_info)

    # Check if user cancelled or nothing selected
    any_selected = (
        benchmark_config.get("embeddings", False)
        or benchmark_config.get("llm", {}).get("lm_studio", {}).get("enabled", False)
        or benchmark_config.get("llm", {}).get("ollama", {}).get("enabled", False)
        or benchmark_config.get("vlm", {}).get("lm_studio", {}).get("enabled", False)
        or benchmark_config.get("vlm", {}).get("ollama", {}).get("enabled", False)
    )

    if not any_selected:
        print("\nNo benchmarks selected. Exiting.")
        return

    # Start power monitoring and run benchmarks
    print("\n" + "=" * 60)
    print("Starting Power Monitoring...")
    print("=" * 60)

    use_power_metrics = benchmark_config.get("power_metrics", False)

    # Determine if we should ask about sudo powermetrics (only on macOS)
    ask_sudo = use_power_metrics and device_info.get("platform") == "Darwin"

    with PowerMonitor(
        interval=1.0,
        ask_user=ask_sudo,  # Only ask about sudo on macOS
    ) as pm:
        all_task_results = run_all_benchmarks(benchmark_config, device_info)

    # Power monitoring stopped
    print("\n" + "=" * 60)
    print("Power Monitoring Stopped")
    print("=" * 60)

    # Save results with power metrics
    final_results = {
        "tasks": all_task_results,
        "power_metrics": pm.results if use_power_metrics else None,
    }
    report_path = save_report(final_results, device_info)

    # Display final summary
    display_final_summary(
        report_path, all_task_results, pm.results if use_power_metrics else None
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print("\n\nUnexpected error during benchmark execution:")
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        print("\nIf this issue persists, please report it on GitHub.")
    finally:
        print("\n" + "=" * 60)
        print("Benchmark session ended.")
        print("=" * 60)
