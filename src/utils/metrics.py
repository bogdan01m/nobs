"""Utilities for calculating benchmark metrics."""

import numpy as np
from statistics import mean, stdev


def calculate_percentiles(
    values: list[float],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Calculate P25, P50, P75, P95 percentiles for a list of values.

    Args:
        values: List of numeric values to calculate percentiles from.

    Returns:
        Tuple of (P25, P50, P75, P95) percentiles. Returns (None, None, None, None) if input is empty.
    """
    if not values:
        return None, None, None, None
    arr = np.array(values)
    return (
        np.percentile(arr, 25),
        np.percentile(arr, 50),
        np.percentile(arr, 75),
        np.percentile(arr, 95),
    )


def calculate_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Calculate mean and standard deviation for a list of values.

    Args:
        values: List of numeric values to calculate statistics from.

    Returns:
        Tuple of (mean, std). Returns (None, None) if input is empty.
    """
    if not values:
        return None, None
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    return m, s
