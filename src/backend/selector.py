"""Backend selection and configuration."""

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
