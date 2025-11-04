"""Report saving and management."""

import json
from datetime import datetime
from pathlib import Path


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

    # Получаем host из device_info
    host = device_info.get("host", "unknown_host")
    report_filename = f"report_{host}.json"
    report_path = results_dir / report_filename

    # Добавляем device_info и timestamp в результаты
    full_report = {
        "timestamp": datetime.now().isoformat(),
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
    print("=" * 50)

    return report_path
