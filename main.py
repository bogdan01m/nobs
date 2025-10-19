import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from datetime import datetime
from pathlib import Path
from src.device_info import get_device_info
from src.tasks.text_embeddings.runner import run_embeddings_benchmark
from src.tasks.llms.runner import run_llms_benchmark
from src.lm_studio_setup import setup_lm_studio, cleanup_lm_studio


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


def main():
    """
    Главная функция для запуска всех бенчмарков
    """
    # Получаем информацию об устройстве
    device_info = get_device_info()

    print("=" * 60)
    print("NoBS Benchmark")
    print("(No Bullshit Benchmark for Real AI Performance)")
    print("=" * 60)
    print("Device Info:")
    print(json.dumps(device_info, indent=2))
    print("=" * 60)
    print()

    # Список для хранения всех результатов задач
    all_task_results = []
    llm_was_setup = False  # Флаг для cleanup

    # Запускаем embeddings бенчмарк
    while True:
        run_embeddings = input("\nRun Embeddings benchmark? (y/n): ").strip().lower()
        if run_embeddings in ["y", "n"]:
            break
        print("Please enter 'y' or 'n'")

    if run_embeddings == "y":
        print("Starting Embeddings Benchmark...")
        print()
        embeddings_results = run_embeddings_benchmark()
        all_task_results.append(embeddings_results)
    else:
        print("Skipping Embeddings benchmark.")

    # Спрашиваем про LLM бенчмарк
    print("\n" + "=" * 50)
    print("LLM Benchmark")
    print("=" * 50)
    print("This benchmark requires LM Studio to be installed.")
    print("Download from: https://lmstudio.ai/")
    print()
    print("Default settings (no .env changes needed):")
    print("  LLM_MODEL_NAME=openai/gpt-oss-20b")
    print("  LLM_BASE_URL=http://127.0.0.1:1234/v1")
    print()
    print("Only modify .env if using custom API/model.")
    print("=" * 50)

    while True:
        run_llm = input("\nRun LLM benchmark? (y/n): ").strip().lower()
        if run_llm in ["y", "n"]:
            break
        print("Please enter 'y' or 'n'")

    if run_llm == "y":
        # Автоматическая настройка: скачать модель, запустить сервер, загрузить в память
        print()
        print("Auto-setup will:")
        print("  1. Download model (if not already downloaded)")
        print("  2. Start LM Studio server")
        print("  3. Load model into memory")
        print("  4. Verify model is ready")

        while True:
            auto_setup = input("\nAuto-setup model and server? (y/n): ").strip().lower()
            if auto_setup in ["y", "n"]:
                break
            print("Please enter 'y' or 'n'")

        if auto_setup == "y":
            if not setup_lm_studio():
                print("✗ LM Studio setup failed. Skipping LLM benchmark.")
            else:
                llm_was_setup = True
                print("\n" + "=" * 50)
                print("Starting LLM Benchmark...")
                print("=" * 50)
                print()
                llm_results = run_llms_benchmark()
                all_task_results.append(llm_results)
        else:
            print("\n" + "=" * 50)
            print("Starting LLM Benchmark...")
            print("=" * 50)
            print()
            llm_results = run_llms_benchmark()
            all_task_results.append(llm_results)
    else:
        print("Skipping LLM benchmark.")

    # Cleanup LM Studio если был auto-setup
    if llm_was_setup:
        print()
        cleanup_lm_studio()

    # Сохраняем результаты без total_score
    final_results = {"tasks": all_task_results}

    # Сохраняем отчет
    report_path = save_report(final_results, device_info)

    print()
    print("=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {report_path}")
    print()
    print("Task Scores:")
    for task in all_task_results:
        task_name = task.get("task", "unknown")
        task_score = task.get("task_score", "N/A")
        print(f"  {task_name}: {task_score}")
    print("=" * 50)


if __name__ == "__main__":
    main()
