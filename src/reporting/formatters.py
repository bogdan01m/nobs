"""Value formatting utilities for display."""

from typing import Any

# Platform emojis
PLATFORM_EMOJIS = {
    "Darwin": "🍏 macOS",
    "Linux": "🐧 Linux",
    "Windows": "🪟 Windows",
}

# Rank emojis for top 3
RANK_EMOJIS = {1: "🥇", 2: "🥈", 3: "🥉"}

# Vendor emojis
VENDOR_EMOJIS = {
    "Apple": "⚫",
    "NVIDIA": "🟢",
    "AMD": "🔴",
    "Intel": "🔵",
    "Other": "⚪",
}


def format_time_with_std(value: float | None, std: float) -> str:
    """Format time value with standard deviation.

    Args:
        value: Time value in seconds
        std: Standard deviation

    Returns:
        Formatted string like "2.89 ± 0.15" or "-" if None
    """
    if value is None:
        return "-"
    return f"{value:.2f} ± {std:.2f}"


def format_metric_with_std(value: float | None, std: float, unit: str = "") -> str:
    """Format metric value with standard deviation and optional unit.

    Args:
        value: Metric value
        std: Standard deviation
        unit: Optional unit suffix (e.g., "s", "ms")

    Returns:
        Formatted string or "N/A" if None
    """
    if value is None:
        return "N/A"
    formatted = f"{value:.2f} ± {std:.2f}"
    if unit:
        formatted += f" {unit}"
    return formatted


def format_percentile_pair(p50: float | str, p95: float | str, unit: str) -> str:
    """Format p50/p95 percentile pair.

    Args:
        p50: 50th percentile value
        p95: 95th percentile value
        unit: Unit string (e.g., "%", "GB", "W")

    Returns:
        Formatted string like "45.2% / 78.3%" or "N/A"
    """
    if p50 == "N/A" or p95 == "N/A":
        return "N/A"
    return f"{p50:.1f}{unit} / {p95:.1f}{unit}"


def format_vram(gpu_mem: float | str) -> str:
    """Format VRAM value.

    Args:
        gpu_mem: GPU memory in GB or string description

    Returns:
        Formatted string like "32 GB" or original string
    """
    if isinstance(gpu_mem, (int, float)):
        return f"{gpu_mem:.0f} GB"
    return str(gpu_mem)


def format_platform(platform: str) -> str:
    """Format platform name with emoji.

    Args:
        platform: Platform name (Darwin, Linux, Windows)

    Returns:
        Platform name with emoji
    """
    return PLATFORM_EMOJIS.get(platform, platform)


def format_rank(rank: int) -> str:
    """Format rank with emoji for top 3.

    Args:
        rank: Ranking position (1-indexed)

    Returns:
        Formatted rank string with emoji if applicable
    """
    emoji = RANK_EMOJIS.get(rank, "")
    return f"{emoji} {rank}" if emoji else str(rank)


def get_gpu_vendor(gpu_name: str) -> str:
    """Determine GPU vendor from GPU name.

    Args:
        gpu_name: GPU name string

    Returns:
        Vendor name: Apple, NVIDIA, AMD, Intel, or Other
    """
    gpu_lower = gpu_name.lower()
    if any(keyword in gpu_lower for keyword in ["apple", "m1", "m2", "m3", "m4"]):
        return "Apple"
    elif any(
        keyword in gpu_lower for keyword in ["nvidia", "rtx", "gtx", "tesla", "a100"]
    ):
        return "NVIDIA"
    elif any(keyword in gpu_lower for keyword in ["amd", "radeon", "rx"]):
        return "AMD"
    elif any(keyword in gpu_lower for keyword in ["intel", "arc", "uhd"]):
        return "Intel"
    else:
        return "Other"


def format_battery_info(start: Any, end: Any, delta: Any) -> str:
    """Format battery information.

    Args:
        start: Starting battery percentage
        end: Ending battery percentage
        delta: Battery drain percentage

    Returns:
        Formatted string like "95% / 82% / -13%" or "N/A"
    """
    if start == "N/A" or end == "N/A" or delta == "N/A":
        return "N/A"
    return f"{start}% / {end}% / {delta:+.1f}%"
