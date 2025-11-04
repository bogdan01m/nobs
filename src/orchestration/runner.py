"""Benchmark orchestration and execution."""

from src.tasks.text_embeddings.runner import run_embeddings_benchmark
from src.tasks.llms.runner import run_llms_benchmark
from src.tasks.vlms.runner import run_vlms_benchmark
from src.lm_studio_setup import setup_lm_studio, cleanup_lm_studio
from src.ollama_setup import setup_ollama, cleanup_ollama
from src.settings import (
    LMS_LLM_MODEL_NAME,
    LMS_LLM_BASE_URL,
    LMS_VLM_MODEL_NAME,
    LMS_VLM_BASE_URL,
    OLLAMA_LLM_MODEL_NAME,
    OLLAMA_LLM_BASE_URL,
    OLLAMA_VLM_MODEL_NAME,
    OLLAMA_VLM_BASE_URL,
)


def run_all_benchmarks(config: dict, device_info: dict) -> list:
    """
    Run all selected benchmarks based on configuration.

    Args:
        config: Configuration dict from select_benchmarks()
        device_info: Device information dict

    Returns:
        list: List of task results
    """
    all_task_results = []

    print("\n" + "=" * 60)
    print("BENCHMARKS STARTING")
    print("=" * 60)

    # Run Embeddings benchmark
    if config.get("embeddings", False):
        print("\nStarting Embeddings Benchmark...")
        print()
        embeddings_results = run_embeddings_benchmark()
        all_task_results.append(embeddings_results)
    else:
        print("\nSkipping Embeddings benchmark.")

    # Run LLM benchmarks
    llm_config = config.get("llm", {})

    # LM Studio LLM
    if llm_config.get("lm_studio", {}).get("enabled", False):
        auto_setup = llm_config["lm_studio"].get("auto_setup", False)
        backend_info = {
            "backend": "LM_STUDIO",
            "model_name": LMS_LLM_MODEL_NAME,
            "base_url": LMS_LLM_BASE_URL,
            "setup_func": setup_lm_studio,
            "cleanup_func": cleanup_lm_studio,
        }

        result = _run_llm_with_backend(backend_info, auto_setup)
        if result:
            all_task_results.append(result)

    # Ollama LLM
    if llm_config.get("ollama", {}).get("enabled", False):
        auto_setup = llm_config["ollama"].get("auto_setup", False)
        backend_info = {
            "backend": "OLLAMA",
            "model_name": OLLAMA_LLM_MODEL_NAME,
            "base_url": OLLAMA_LLM_BASE_URL,
            "setup_func": setup_ollama,
            "cleanup_func": cleanup_ollama,
        }

        result = _run_llm_with_backend(backend_info, auto_setup)
        if result:
            all_task_results.append(result)

    # Run VLM benchmarks
    vlm_config = config.get("vlm", {})

    # LM Studio VLM
    if vlm_config.get("lm_studio", {}).get("enabled", False):
        auto_setup = vlm_config["lm_studio"].get("auto_setup", False)
        backend_info = {
            "backend": "LM_STUDIO",
            "model_name": LMS_VLM_MODEL_NAME,
            "base_url": LMS_VLM_BASE_URL,
            "setup_func": setup_lm_studio,
            "cleanup_func": cleanup_lm_studio,
        }

        result = _run_vlm_with_backend(backend_info, auto_setup)
        if result:
            all_task_results.append(result)

    # Ollama VLM
    if vlm_config.get("ollama", {}).get("enabled", False):
        auto_setup = vlm_config["ollama"].get("auto_setup", False)
        backend_info = {
            "backend": "OLLAMA",
            "model_name": OLLAMA_VLM_MODEL_NAME,
            "base_url": OLLAMA_VLM_BASE_URL,
            "setup_func": setup_ollama,
            "cleanup_func": cleanup_ollama,
        }

        result = _run_vlm_with_backend(backend_info, auto_setup)
        if result:
            all_task_results.append(result)

    return all_task_results


def _run_llm_with_backend(backend_info: dict, auto_setup: bool) -> dict | None:
    """
    Run LLM benchmark with given backend.

    Args:
        backend_info: Backend configuration dict
        auto_setup: Whether to auto-setup the backend

    Returns:
        dict: Task results or None if failed
    """
    print()
    print(f"\nUsing {backend_info['backend']} with model: {backend_info['model_name']}")

    if auto_setup:
        # Setup the backend
        if backend_info["setup_func"](
            backend_info["model_name"], backend_info["base_url"]
        ):
            print("\n" + "=" * 50)
            print(f"Starting LLM Benchmark on {backend_info['backend']}...")
            print("=" * 50)
            print()
            llm_results = run_llms_benchmark(
                model_name=backend_info["model_name"],
                base_url=backend_info["base_url"],
            )

            # Add backend info to results
            llm_results["backend"] = backend_info["backend"]
            llm_results["backend_model"] = backend_info["model_name"]

            # Cleanup after this backend
            print()
            backend_info["cleanup_func"]()

            return llm_results
        else:
            print(f"✗ {backend_info['backend']} setup failed. Skipping this backend.")
            return None
    else:
        # Manual setup - just run the benchmark
        print("\n" + "=" * 50)
        print(f"Starting LLM Benchmark on {backend_info['backend']}...")
        print("=" * 50)
        print(
            "Note: Auto-setup is disabled. Make sure the backend is running manually."
        )
        print()
        llm_results = run_llms_benchmark(
            model_name=backend_info["model_name"],
            base_url=backend_info["base_url"],
        )

        # Add backend info to results
        llm_results["backend"] = backend_info["backend"]
        llm_results["backend_model"] = backend_info["model_name"]

        return llm_results


def _run_vlm_with_backend(backend_info: dict, auto_setup: bool) -> dict | None:
    """
    Run VLM benchmark with given backend.

    Args:
        backend_info: Backend configuration dict
        auto_setup: Whether to auto-setup the backend

    Returns:
        dict: Task results or None if failed
    """
    print()
    print(f"\nUsing {backend_info['backend']} with model: {backend_info['model_name']}")

    if auto_setup:
        # Setup the backend
        if backend_info["setup_func"](
            backend_info["model_name"], backend_info["base_url"]
        ):
            print("\n" + "=" * 50)
            print(f"Starting VLM Benchmark on {backend_info['backend']}...")
            print("=" * 50)
            print()
            vlm_results = run_vlms_benchmark(
                model_name=backend_info["model_name"],
                base_url=backend_info["base_url"],
            )

            # Add backend info to results
            vlm_results["backend"] = backend_info["backend"]
            vlm_results["backend_model"] = backend_info["model_name"]

            # Cleanup after this backend
            print()
            backend_info["cleanup_func"]()

            return vlm_results
        else:
            print(f"✗ {backend_info['backend']} setup failed. Skipping this backend.")
            return None
    else:
        # Manual setup - just run the benchmark
        print("\n" + "=" * 50)
        print(f"Starting VLM Benchmark on {backend_info['backend']}...")
        print("=" * 50)
        print(
            "Note: Auto-setup is disabled. Make sure the backend is running manually."
        )
        print()
        vlm_results = run_vlms_benchmark(
            model_name=backend_info["model_name"],
            base_url=backend_info["base_url"],
        )

        # Add backend info to results
        vlm_results["backend"] = backend_info["backend"]
        vlm_results["backend_model"] = backend_info["model_name"]

        return vlm_results
