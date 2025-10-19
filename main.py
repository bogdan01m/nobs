import json
from datetime import datetime
from pathlib import Path
from src.device_info import get_device_info
from src.tasks.text_embeddings.runner import run_embeddings_benchmark
from src.score import calculate_score


def save_report(results: dict, device_info: dict):
    """
    Сохраняет отчет в папку results/

    Args:
        results: Результаты бенчмарка
        device_info: Информация об устройстве
    """
    # Создаем папку results
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
        **results
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


def main():
    """
    Главная функция для запуска всех бенчмарков
    """
    # Получаем информацию об устройстве
    device_info = get_device_info()

    print("=" * 50)
    print("NOBS BENCHMARK SUITE")
    print("=" * 50)
    print("Device Info:")
    print(json.dumps(device_info, indent=2))
    print("=" * 50)
    print()

    # Список для хранения всех результатов задач
    all_task_results = []

    # Запускаем embeddings бенчмарк
    print("Starting Embeddings Benchmark...")
    print()
    embeddings_results = run_embeddings_benchmark()
    all_task_results.append(embeddings_results)

    # Рассчитываем суммарный скор
    total_score = sum(task.get("task_score", 0) for task in all_task_results)

    # Добавляем суммарный скор в результаты
    final_results = {
        "tasks": all_task_results,
        "total_score": total_score
    }

    # Сохраняем отчет
    report_path = save_report(final_results, device_info)

    print()
    print("=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {report_path}")
    print(f"Total Score: {total_score}")
    print("=" * 50)


if __name__ == "__main__":
    main()