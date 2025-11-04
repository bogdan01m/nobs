"""Data loading utilities for benchmark results."""

import json
from pathlib import Path
from typing import Any


def load_results(results_dir: Path = Path("results")) -> list[dict[str, Any]]:
    """Load all result JSON files from the results directory.

    Args:
        results_dir: Directory containing report_*.json files

    Returns:
        List of result dictionaries sorted by host name
    """
    results = []
    for json_file in results_dir.glob("report_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    return sorted(results, key=lambda x: x["device_info"]["host"])
