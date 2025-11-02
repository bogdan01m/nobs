import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from datetime import datetime
from pathlib import Path
from src.system_info.device_info import get_device_info
from src.system_info.power_metrics import PowerMonitor
from src.tasks.text_embeddings.runner import run_embeddings_benchmark
from src.tasks.llms.runner import run_llms_benchmark
from src.tasks.vlms.runner import run_vlms_benchmark
from src.lm_studio_setup import setup_lm_studio, cleanup_lm_studio, check_lms_cli
from src.ollama_setup import setup_ollama, cleanup_ollama, check_ollama_cli
from src.settings import (
    LMS_LLM_MODEL_NAME,
    LMS_LLM_BASE_URL,
    LMS_VLM_MODEL_NAME,
    LMS_VLM_BASE_URL,
    LLM_BACKEND,
    VLM_BACKEND,
    OLLAMA_LLM_MODEL_NAME,
    OLLAMA_LLM_BASE_URL,
    OLLAMA_VLM_MODEL_NAME,
    OLLAMA_VLM_BASE_URL,
)


def select_backend(task_type: str):
    """
    Выбирает бэкенд(ы) для запуска модели

    Args:
        task_type: "llm" или "vlm"

    Returns:
        list[dict]: Список бэкендов с ключами: backend, model_name, base_url, setup_func, cleanup_func
        Empty list если бэкенды не доступны
    """
    backend_preference = LLM_BACKEND if task_type == "llm" else VLM_BACKEND

    print()
    print("=" * 50)
    print(f"Backend Selection for {task_type.upper()}")
    print("=" * 50)

    # BOTH: запускаем на обоих бэкендах
    if backend_preference == "BOTH":
        print("Configuration: BOTH (running on LM Studio AND Ollama)")
        backends = []

        # Проверяем LM Studio
        if check_lms_cli():
            print("✓ LM Studio CLI found - will use LM Studio")
            backends.append(
                {
                    "backend": "LM_STUDIO",
                    "model_name": LMS_LLM_MODEL_NAME
                    if task_type == "llm"
                    else LMS_VLM_MODEL_NAME,
                    "base_url": LMS_LLM_BASE_URL
                    if task_type == "llm"
                    else LMS_VLM_BASE_URL,
                    "setup_func": setup_lm_studio,
                    "cleanup_func": cleanup_lm_studio,
                }
            )
        else:
            print("✗ LM Studio CLI not found")

        # Проверяем Ollama
        if check_ollama_cli():
            print("✓ Ollama CLI found - will use Ollama")
            backends.append(
                {
                    "backend": "OLLAMA",
                    "model_name": OLLAMA_LLM_MODEL_NAME
                    if task_type == "llm"
                    else OLLAMA_VLM_MODEL_NAME,
                    "base_url": OLLAMA_LLM_BASE_URL
                    if task_type == "llm"
                    else OLLAMA_VLM_BASE_URL,
                    "setup_func": setup_ollama,
                    "cleanup_func": cleanup_ollama,
                }
            )
        else:
            print("✗ Ollama CLI not found")

        if not backends:
            print("✗ No backends available")
            print("Please install:")
            print("  - LM Studio: https://lmstudio.ai/")
            print("  - Ollama: https://ollama.ai/")

        return backends

    # Если указан конкретный бэкенд
    elif backend_preference == "LM_STUDIO":
        print("Configuration: LM Studio (forced)")
        if check_lms_cli():
            return [
                {
                    "backend": "LM_STUDIO",
                    "model_name": LMS_LLM_MODEL_NAME
                    if task_type == "llm"
                    else LMS_VLM_MODEL_NAME,
                    "base_url": LMS_LLM_BASE_URL
                    if task_type == "llm"
                    else LMS_VLM_BASE_URL,
                    "setup_func": setup_lm_studio,
                    "cleanup_func": cleanup_lm_studio,
                }
            ]
        else:
            print("✗ LM Studio CLI not found")
            return []

    elif backend_preference == "OLLAMA":
        print("Configuration: Ollama (forced)")
        if check_ollama_cli():
            return [
                {
                    "backend": "OLLAMA",
                    "model_name": OLLAMA_LLM_MODEL_NAME
                    if task_type == "llm"
                    else OLLAMA_VLM_MODEL_NAME,
                    "base_url": OLLAMA_LLM_BASE_URL
                    if task_type == "llm"
                    else OLLAMA_VLM_BASE_URL,
                    "setup_func": setup_ollama,
                    "cleanup_func": cleanup_ollama,
                }
            ]
        else:
            print("✗ Ollama CLI not found")
            return []

    # AUTO: пробуем LM Studio, потом Ollama (выбираем первый доступный)
    else:
        print("Configuration: AUTO (trying LM Studio first, then Ollama)")

        # Пробуем LM Studio
        if check_lms_cli():
            print("✓ LM Studio CLI found - using LM Studio")
            return [
                {
                    "backend": "LM_STUDIO",
                    "model_name": LMS_LLM_MODEL_NAME
                    if task_type == "llm"
                    else LMS_VLM_MODEL_NAME,
                    "base_url": LMS_LLM_BASE_URL
                    if task_type == "llm"
                    else LMS_VLM_BASE_URL,
                    "setup_func": setup_lm_studio,
                    "cleanup_func": cleanup_lm_studio,
                }
            ]

        # Fallback на Ollama
        elif check_ollama_cli():
            print("✓ Ollama CLI found - using Ollama")
            return [
                {
                    "backend": "OLLAMA",
                    "model_name": OLLAMA_LLM_MODEL_NAME
                    if task_type == "llm"
                    else OLLAMA_VLM_MODEL_NAME,
                    "base_url": OLLAMA_LLM_BASE_URL
                    if task_type == "llm"
                    else OLLAMA_VLM_BASE_URL,
                    "setup_func": setup_ollama,
                    "cleanup_func": cleanup_ollama,
                }
            ]

        else:
            print("✗ No backend found (neither LM Studio nor Ollama)")
            print("Please install:")
            print("  - LM Studio: https://lmstudio.ai/")
            print("  - Ollama: https://ollama.ai/")
            return []


def save_report(results: dict, device_info: dict):
    """
    Stores run results in dir: results/

    Args:
        results: bench results
        device_info: device results
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


def main():
    """
    Func to start bench run
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

    # ========== STEP 1: Ask user which benchmarks to run ==========

    # Ask about Embeddings
    while True:
        run_embeddings = input("\nRun Embeddings benchmark? (y/n): ").strip().lower()
        if run_embeddings in ["y", "n"]:
            break
        print("Please enter 'y' or 'n'")

    # Ask about LLM benchmark
    print("\n" + "=" * 50)
    print("LLM Benchmark")
    print("=" * 50)
    print("This benchmark requires LM Studio to be installed.")
    print("Download from: https://lmstudio.ai/")
    print()
    print("Default settings (configure in .env):")
    print(f"  LMS_LLM_MODEL_NAME={LMS_LLM_MODEL_NAME}")
    print(f"  LMS_LLM_BASE_URL={LMS_LLM_BASE_URL}")
    print()
    print("Model selection is automatic based on your device:")
    print("  - Apple Silicon (M-series): Uses MLX models")
    print("  - Other devices: Uses GGUF models")
    print("=" * 50)

    while True:
        run_llm = input("\nRun LLM benchmark? (y/n): ").strip().lower()
        if run_llm in ["y", "n"]:
            break
        print("Please enter 'y' or 'n'")

    # Ask about LLM auto-setup
    auto_setup_llm = False
    if run_llm == "y":
        print()
        print("Auto-setup will:")
        print("  1. Download model (if not already downloaded)")
        print("  2. Start LM Studio server")
        print("  3. Load model into memory")
        print("  4. Verify model is ready")

        while True:
            response = input("\nAuto-setup model and server? (y/n): ").strip().lower()
            if response in ["y", "n"]:
                auto_setup_llm = response == "y"
                break
            print("Please enter 'y' or 'n'")

    # Ask about VLM benchmark
    print("\n" + "=" * 50)
    print("VLM (Vision-Language Model) Benchmark")
    print("=" * 50)
    print("This benchmark requires LM Studio with a VLM model.")
    print("Dataset: Hallucination_COCO (3 questions with images)")
    print()
    print("Default settings (configure in .env):")
    print(f"  LMS_VLM_MODEL_NAME={LMS_VLM_MODEL_NAME}")
    print(f"  LMS_VLM_BASE_URL={LMS_VLM_BASE_URL}")
    print("=" * 50)

    while True:
        run_vlm = input("\nRun VLM benchmark? (y/n): ").strip().lower()
        if run_vlm in ["y", "n"]:
            break
        print("Please enter 'y' or 'n'")

    # Ask about VLM auto-setup
    auto_setup_vlm = False
    if run_vlm == "y":
        print()
        print("Auto-setup will:")
        print("  1. Download VLM model (if not already downloaded)")
        print("  2. Start LM Studio server")
        print("  3. Load VLM model into memory")
        print("  4. Verify model is ready")

        while True:
            response = (
                input("\nAuto-setup VLM model and server? (y/n): ").strip().lower()
            )
            if response in ["y", "n"]:
                auto_setup_vlm = response == "y"
                break
            print("Please enter 'y' or 'n'")

    # ========== STEP 2: Start Power Monitoring and Run Benchmarks ==========

    # Initialize power monitoring (will ask about sudo powermetrics on macOS)
    print("\n" + "=" * 60)
    print("Starting Power Monitoring...")
    print("=" * 60)

    all_task_results = []

    # Start power monitoring for entire benchmark session
    with PowerMonitor(interval=1.0) as pm:
        print("\n" + "=" * 60)
        print("BENCHMARKS STARTING")
        print("=" * 60)

        # Run Embeddings benchmark
        if run_embeddings == "y":
            print("\nStarting Embeddings Benchmark...")
            print()
            embeddings_results = run_embeddings_benchmark()
            all_task_results.append(embeddings_results)
        else:
            print("\nSkipping Embeddings benchmark.")

        # Run LLM benchmark
        if run_llm == "y":
            if auto_setup_llm:
                # Select backend(s) - can be multiple when BOTH is selected
                llm_backends = select_backend("llm")

                if not llm_backends:
                    print("✗ No backend available. Skipping LLM benchmark.")
                else:
                    # Run benchmark on each selected backend
                    for backend in llm_backends:
                        print()
                        print(
                            f"\nUsing {backend['backend']} with model: {backend['model_name']}"
                        )

                        # Setup the backend
                        if backend["setup_func"](
                            backend["model_name"], backend["base_url"]
                        ):
                            print("\n" + "=" * 50)
                            print(f"Starting LLM Benchmark on {backend['backend']}...")
                            print("=" * 50)
                            print()
                            llm_results = run_llms_benchmark(
                                model_name=backend["model_name"],
                                base_url=backend["base_url"],
                            )

                            # Add backend info to results
                            llm_results["backend"] = backend["backend"]
                            llm_results["backend_model"] = backend["model_name"]

                            all_task_results.append(llm_results)

                            # Cleanup after this backend
                            print()
                            backend["cleanup_func"]()
                        else:
                            print(
                                f"✗ {backend['backend']} setup failed. Skipping this backend."
                            )
            else:
                # Manual setup - just run the benchmark once
                print("\n" + "=" * 50)
                print("Starting LLM Benchmark...")
                print("=" * 50)
                print()
                llm_results = run_llms_benchmark()
                all_task_results.append(llm_results)
        else:
            print("\nSkipping LLM benchmark.")

        # Run VLM benchmark
        if run_vlm == "y":
            if auto_setup_vlm:
                # Select backend(s) - can be multiple when BOTH is selected
                vlm_backends = select_backend("vlm")

                if not vlm_backends:
                    print("✗ No backend available. Skipping VLM benchmark.")
                else:
                    # Run benchmark on each selected backend
                    for backend in vlm_backends:
                        print()
                        print(
                            f"\nUsing {backend['backend']} with model: {backend['model_name']}"
                        )

                        # Setup the backend
                        if backend["setup_func"](
                            backend["model_name"], backend["base_url"]
                        ):
                            print("\n" + "=" * 50)
                            print(f"Starting VLM Benchmark on {backend['backend']}...")
                            print("=" * 50)
                            print()
                            vlm_results = run_vlms_benchmark(
                                model_name=backend["model_name"],
                                base_url=backend["base_url"],
                            )

                            # Add backend info to results
                            vlm_results["backend"] = backend["backend"]
                            vlm_results["backend_model"] = backend["model_name"]

                            all_task_results.append(vlm_results)

                            # Cleanup after this backend
                            print()
                            backend["cleanup_func"]()
                        else:
                            print(
                                f"✗ {backend['backend']} setup failed. Skipping this backend."
                            )
            else:
                # Manual setup - just run the benchmark once
                print("\n" + "=" * 50)
                print("Starting VLM Benchmark...")
                print("=" * 50)
                print()
                vlm_results = run_vlms_benchmark()
                all_task_results.append(vlm_results)
        else:
            print("\nSkipping VLM benchmark.")

    # Power monitoring stopped, results in pm.results
    print("\n" + "=" * 60)
    print("Power Monitoring Stopped")
    print("=" * 60)

    # ========== STEP 3: Save results with power metrics ==========

    final_results = {
        "tasks": all_task_results,
        "power_metrics": pm.results,
    }
    report_path = save_report(final_results, device_info)

    print()
    print("=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {report_path}")
    print()
    print("Task Times:")
    for task in all_task_results:
        task_name = task.get("task", "unknown")
        total_time = task.get("total_time_seconds", "N/A")
        print(f"  {task_name}: {total_time}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
