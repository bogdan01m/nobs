"""Display functions for CLI output."""

import json
from pathlib import Path


def display_device_info(device_info: dict):
    """
    Display formatted device information.

    Args:
        device_info: Device information dict from get_device_info()
    """
    print("=" * 60)
    print("NoBS Benchmark")
    print("(No Bullshit Benchmark for Real AI Performance)")
    print("=" * 60)
    print("Device Info:")
    print(json.dumps(device_info, indent=2))
    print("=" * 60)
    print()


def display_final_summary(
    report_path: Path, all_task_results: list, power_metrics: dict | None = None
):
    """
    Display final summary of benchmark results.

    Args:
        report_path: Path to saved report file
        all_task_results: List of task results
        power_metrics: Optional power metrics dict
    """
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Tasks completed: {len(all_task_results)}")

    for task_result in all_task_results:
        task_name = task_result.get("task", "Unknown")
        print(f"  ✓ {task_name}")

    if power_metrics:
        duration = power_metrics.get("duration_seconds", 0)
        num_samples = power_metrics.get("num_samples", 0)
        print("\nPower monitoring:")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Samples: {num_samples}")

    print(f"\nReport saved to: {report_path}")
    print("=" * 60)
