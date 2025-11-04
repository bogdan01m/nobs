"""Report saving and management."""

import hashlib
import json
from datetime import datetime
from pathlib import Path


def generate_unique_hex(timestamp: str, device_info: dict, length: int = 8) -> str:
    """
    Generate a unique hex identifier based on timestamp and device info.

    Args:
        timestamp: ISO format timestamp string
        device_info: Device information dictionary
        length: Length of hex string (default: 8)

    Returns:
        str: Unique hex identifier (e.g., "8e9123f0")
    """
    # Combine timestamp and relevant device info for uniqueness
    data_to_hash = f"{timestamp}|{device_info.get('host', '')}|{device_info.get('processor', '')}|{device_info.get('gpu_name', '')}|{device_info.get('platform', '')}"

    # Create SHA256 hash
    hash_obj = hashlib.sha256(data_to_hash.encode("utf-8"))

    # Return first 'length' characters of hex digest
    return hash_obj.hexdigest()[:length]


def save_report(results: dict, device_info: dict):
    """
    Stores run results in dir: results/

    Args:
        results: bench results
        device_info: device results

    Returns:
        Path: Path to saved report file
    """
    # create `results`
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Generate timestamp and unique hex
    timestamp = datetime.now().isoformat()
    unique_hex = generate_unique_hex(timestamp, device_info)

    # Получаем host из device_info и добавляем hex
    host = device_info.get("host", "unknown_host")
    report_filename = f"report_{host}_{unique_hex}.json"
    report_path = results_dir / report_filename

    # Добавляем device_info и timestamp в результаты
    full_report = {
        "timestamp": timestamp,
        "device_info": device_info,
        **results,
    }

    # Сохраняем отчет
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print("=" * 50)
    print("REPORT SAVED")
    print("=" * 50)
    print(f"Location: {report_path}")
    print(f"Host: {host}")
    print(f"Unique ID: {unique_hex}")
    print("=" * 50)

    return report_path
