import subprocess
import time
import requests
from src.settings import LLM_MODEL_NAME, LLM_BASE_URL


def check_lms_cli():
    """Проверяет, установлен ли LM Studio CLI"""
    try:
        result = subprocess.run(["lms", "status"], capture_output=True, text=True)
        # Если команда выполнилась (даже если сервер OFF), значит CLI установлен
        return result.returncode == 0
    except FileNotFoundError:
        return False


def is_model_downloaded(model_name: str):
    """Проверяет, загружена ли модель"""
    try:
        result = subprocess.run(
            ["lms", "ls"], capture_output=True, text=True, check=True
        )
        return model_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def download_model(model_name: str):
    """Загружает модель"""
    print(f"Downloading model {model_name}...")
    try:
        subprocess.run(["lms", "get", model_name], check=True)
        print(f"✓ Model {model_name} downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to download model {model_name}")
        return False


def is_server_running():
    """Проверяет, запущен ли сервер"""
    try:
        result = subprocess.run(
            ["lms", "status"], capture_output=True, text=True, check=True
        )
        # Проверяем что сервер включен (Server: ON)
        return "Server: ON" in result.stdout or "Server running" in result.stdout
    except subprocess.CalledProcessError:
        return False


def start_server():
    """Запускает сервер"""
    print("Starting LM Studio server...")
    try:
        subprocess.Popen(
            ["lms", "server", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)  # Даем серверу время на запуск

        if is_server_running():
            print("✓ Server started successfully")
            return True
        else:
            print("✗ Failed to start server")
            return False
    except Exception as e:
        print(f"✗ Error starting server: {e}")
        return False


def is_model_loaded(model_name: str):
    """Проверяет, загружена ли модель в память"""
    try:
        result = subprocess.run(
            ["lms", "ps"], capture_output=True, text=True, check=True
        )
        return model_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def load_model(model_name: str):
    """Загружает модель в память"""
    print(f"Loading model {model_name}...")
    try:
        subprocess.run(["lms", "load", model_name], check=True)
        print(f"✓ Model {model_name} loaded successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to load model {model_name}")
        return False


def ping_model(base_url: str, max_retries: int = 10):
    """Проверяет, что модель готова к работе"""
    print("Verifying model is ready...")

    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                print("✓ Model is ready and responding")

                # Тестовый запрос
                print("Testing with a simple prompt...")
                test_response = requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": LLM_MODEL_NAME,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10,
                    },
                    timeout=30,
                )

                if (
                    test_response.status_code == 200
                    and "choices" in test_response.json()
                ):
                    print("✓ Model test successful")
                    return True
                else:
                    print(f"✗ Model test failed: {test_response.text}")
                    return False

        except requests.RequestException:
            pass

        print(f"Waiting for model to be ready... ({i+1}/{max_retries})")
        time.sleep(3)

    print("✗ Model did not become ready in time")
    return False


def unload_all_models():
    """Выгружает все модели из памяти"""
    print("Unloading all models...")
    try:
        subprocess.run(["lms", "unload", "--all"], check=True)
        print("✓ All models unloaded")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to unload models")
        return False


def stop_server():
    """Останавливает сервер LM Studio"""
    print("Stopping LM Studio server...")
    try:
        subprocess.run(["lms", "server", "stop"], check=True)
        print("✓ Server stopped")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to stop server")
        return False


def cleanup_lm_studio():
    """
    Очистка после бенчмарка: выгрузка моделей и остановка сервера

    Returns:
        bool: True если все успешно, False если есть ошибки
    """
    print("=" * 50)
    print("LM Studio Cleanup")
    print("=" * 50)
    print()

    success = True

    # Выгружаем все модели
    if not unload_all_models():
        success = False

    # Останавливаем сервер
    if not stop_server():
        success = False

    print()
    if success:
        print("=" * 50)
        print("✓ LM Studio cleanup complete!")
        print("=" * 50)
    else:
        print("=" * 50)
        print("⚠ LM Studio cleanup completed with errors")
        print("=" * 50)
    print()

    return success


def setup_lm_studio():
    """
    Полная настройка LM Studio: проверка, загрузка модели, запуск сервера

    Returns:
        bool: True если все успешно, False если есть ошибки
    """
    print("=" * 50)
    print("LM Studio Setup")
    print("=" * 50)
    print()

    # Шаг 1: Проверка CLI
    print("Step 1: Checking LM Studio CLI...")
    if not check_lms_cli():
        print("✗ LM Studio CLI not found")
        print("Please install LM Studio from https://lmstudio.ai/")
        return False
    print("✓ LM Studio CLI found")
    print()

    # Шаг 2: Проверка/загрузка модели
    print("Step 2: Checking if model is downloaded...")
    print(f"Model: {LLM_MODEL_NAME}")
    if not is_model_downloaded(LLM_MODEL_NAME):
        print("Model not found, downloading...")
        if not download_model(LLM_MODEL_NAME):
            return False
    else:
        print("✓ Model already downloaded")
    print()

    # Шаг 3: Проверка/запуск сервера
    print("Step 3: Checking if server is running...")
    if not is_server_running():
        if not start_server():
            return False
    else:
        print("✓ Server already running")
    print()

    # Шаг 4: Загрузка модели в память
    print("Step 4: Loading model...")
    if not is_model_loaded(LLM_MODEL_NAME):
        if not load_model(LLM_MODEL_NAME):
            return False
    else:
        print("✓ Model already loaded")
    print()

    # Шаг 5: Проверка готовности
    print("Step 5: Verifying model is ready...")
    if not ping_model(LLM_BASE_URL):
        return False
    print()

    print("=" * 50)
    print("✓ LM Studio is ready for benchmarking!")
    print("=" * 50)
    print()

    return True


if __name__ == "__main__":
    setup_lm_studio()
