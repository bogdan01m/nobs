"""CLI interaction modules for NOBS benchmark suite."""

from src.cli.display import display_device_info, display_final_summary
from src.cli.menu import select_benchmarks

__all__ = ["display_device_info", "display_final_summary", "select_benchmarks"]
