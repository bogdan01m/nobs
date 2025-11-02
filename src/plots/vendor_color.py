def get_gpu_vendor_color(gpu_name: str) -> str:
    """Determine GPU vendor color for consistent visualization."""
    gpu_lower = gpu_name.lower()
    if any(x in gpu_lower for x in ["apple", "m1", "m2", "m3", "m4"]):
        return "#808080"  # Gray for Apple
    elif any(x in gpu_lower for x in ["nvidia", "rtx", "gtx", "tesla", "a100"]):
        return "#76B900"  # NVIDIA Green
    elif any(x in gpu_lower for x in ["amd", "radeon", "rx"]):
        return "#ED1C24"  # AMD Red
    elif any(x in gpu_lower for x in ["intel", "arc", "uhd"]):
        return "#2E86AB"  # Blue for Intel
    else:
        return "#F77F00"  # Orange for Other
