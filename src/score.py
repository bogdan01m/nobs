def calculate_score(total_time: float,
                    num_tasks: int = 1,
                    C: float = 3600) -> float:
    """
    Calculate benchmark score.

    Args:
        total_time: Total execution time in seconds
        num_tasks: Number of tasks (default 1 for single task score)
        C: Normalization constant (default 3600 seconds = 1 hour)

    Returns:
        Score as float
    """
    if total_time <= 0:
        return float('inf')
    return round(num_tasks * C / float(total_time), 4)
