"""Interactive menu for benchmark selection."""

import platform
from simple_term_menu import TerminalMenu

from src.lm_studio_setup import check_lms_cli
from src.ollama_setup import check_ollama_cli


def select_benchmarks(device_info: dict) -> dict:
    """
    Shows interactive checkbox menu for selecting benchmarks.

    Args:
        device_info: Device information dict from get_device_info()

    Returns:
        dict: Configuration with selected benchmarks
        {
            'embeddings': bool,
            'llm': {
                'lm_studio': {'enabled': bool, 'auto_setup': bool},
                'ollama': {'enabled': bool, 'auto_setup': bool}
            },
            'vlm': {
                'lm_studio': {'enabled': bool, 'auto_setup': bool},
                'ollama': {'enabled': bool, 'auto_setup': bool}
            },
            'power_metrics': bool
        }
    """
    # Check backend availability
    lms_available = check_lms_cli()
    ollama_available = check_ollama_cli()
    is_macos = platform.system() == "Darwin"

    # Build menu items and track their values
    menu_items = []
    item_values = []
    preselected_indices = []

    # Text Embeddings - always available
    menu_items.append("Text Embeddings")
    item_values.append("embeddings")
    preselected_indices.append(0)

    # LLM backends
    if lms_available:
        menu_items.append("LLM - LM Studio (auto-setup)")
        item_values.append("llm_lm_studio")
        preselected_indices.append(len(menu_items) - 1)
    else:
        menu_items.append("LLM - LM Studio (unavailable - CLI not found)")
        item_values.append("llm_lm_studio_disabled")

    if ollama_available:
        menu_items.append("LLM - Ollama (auto-setup)")
        item_values.append("llm_ollama")
        preselected_indices.append(len(menu_items) - 1)
    else:
        menu_items.append("LLM - Ollama (unavailable - CLI not found)")
        item_values.append("llm_ollama_disabled")

    # VLM backends
    if lms_available:
        menu_items.append("VLM - LM Studio (auto-setup)")
        item_values.append("vlm_lm_studio")
        preselected_indices.append(len(menu_items) - 1)
    else:
        menu_items.append("VLM - LM Studio (unavailable - CLI not found)")
        item_values.append("vlm_lm_studio_disabled")

    if ollama_available:
        menu_items.append("VLM - Ollama (auto-setup)")
        item_values.append("vlm_ollama")
        preselected_indices.append(len(menu_items) - 1)
    else:
        menu_items.append("VLM - Ollama (unavailable - CLI not found)")
        item_values.append("vlm_ollama_disabled")

    # Power Metrics - available for macOS and Linux with NVIDIA GPUs
    if is_macos:
        menu_items.append("Power Metrics (macOS)")
        item_values.append("power_metrics")
        preselected_indices.append(len(menu_items) - 1)
    elif device_info.get("device") == "cuda":
        menu_items.append("Power Metrics (Linux + NVIDIA)")
        item_values.append("power_metrics")
        preselected_indices.append(len(menu_items) - 1)
    else:
        menu_items.append("Power Metrics (macOS/Linux+NVIDIA only)")
        item_values.append("power_metrics_disabled")

    # Show interactive menu
    print("\n" + "=" * 60)
    print("Select benchmarks to run:")
    print("=" * 60)

    terminal_menu = TerminalMenu(
        menu_items,
        multi_select=True,
        show_multi_select_hint=True,
        preselected_entries=preselected_indices,
    )

    selected_indices = terminal_menu.show()

    # If user cancels (ESC/Ctrl+C), return empty config
    if selected_indices is None:
        print("\nBenchmark selection cancelled.")
        return {
            "embeddings": False,
            "llm": {
                "lm_studio": {"enabled": False, "auto_setup": False},
                "ollama": {"enabled": False, "auto_setup": False},
            },
            "vlm": {
                "lm_studio": {"enabled": False, "auto_setup": False},
                "ollama": {"enabled": False, "auto_setup": False},
            },
            "power_metrics": False,
        }

    # Convert tuple of indices to set of values
    selected_values = {item_values[i] for i in selected_indices}

    # Parse selections into config
    config: dict = {
        "embeddings": "embeddings" in selected_values,
        "llm": {
            "lm_studio": {
                "enabled": "llm_lm_studio" in selected_values,
                "auto_setup": "llm_lm_studio" in selected_values,
            },
            "ollama": {
                "enabled": "llm_ollama" in selected_values,
                "auto_setup": "llm_ollama" in selected_values,
            },
        },
        "vlm": {
            "lm_studio": {
                "enabled": "vlm_lm_studio" in selected_values,
                "auto_setup": "vlm_lm_studio" in selected_values,
            },
            "ollama": {
                "enabled": "vlm_ollama" in selected_values,
                "auto_setup": "vlm_ollama" in selected_values,
            },
        },
        "power_metrics": "power_metrics" in selected_values,
    }

    # Display selected configuration
    print("\n" + "=" * 60)
    print("Selected benchmarks:")
    print("=" * 60)
    if config["embeddings"]:
        print("  ✓ Text Embeddings")
    if config["llm"]["lm_studio"]["enabled"]:
        print("  ✓ LLM - LM Studio (auto-setup)")
    if config["llm"]["ollama"]["enabled"]:
        print("  ✓ LLM - Ollama (auto-setup)")
    if config["vlm"]["lm_studio"]["enabled"]:
        print("  ✓ VLM - LM Studio (auto-setup)")
    if config["vlm"]["ollama"]["enabled"]:
        print("  ✓ VLM - Ollama (auto-setup)")
    if config["power_metrics"]:
        print("  ✓ Power Metrics")
    print("=" * 60)
    print()

    return config
