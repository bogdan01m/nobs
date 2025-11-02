import subprocess
import time
import requests


def check_ollama_cli():
    """Проверяет, установлен ли Ollama CLI"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def is_model_downloaded(model_name: str):
    """Проверяет, загружена ли модель"""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        # Ollama list показывает модели в формате "name:tag"
        return model_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def pull_model(model_name: str):
    """Загружает модель через ollama pull"""
    print(f"Downloading model {model_name}...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✓ Model {model_name} downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to download model {model_name}")
        return False


def is_server_running():
    """Проверяет, запущен ли Ollama сервер"""
    try:
        # Если ollama list работает, значит сервер запущен
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def start_server():
    """Запускает Ollama сервер (ollama serve)"""
    print("Starting Ollama server...")
    try:
        # Запускаем сервер в фоне
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # Даем серверу время на запуск

        if is_server_running():
            print("✓ Ollama server started successfully")
            return True
        else:
            print("✗ Failed to start Ollama server")
            return False
    except Exception as e:
        print(f"✗ Error starting Ollama server: {e}")
        return False


def is_model_running(model_name: str):
    """Проверяет, запущена ли модель"""
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, check=True
        )
        return model_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def run_model(model_name: str):
    """Запускает модель через ollama run (в фоне)"""
    print(f"Loading model {model_name}...")
    try:
        # Запускаем модель в фоне с пустым промптом чтобы она просто загрузилась
        subprocess.Popen(
            ["ollama", "run", model_name, ""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)  # Даем модели время на загрузку

        if is_model_running(model_name):
            print(f"✓ Model {model_name} loaded successfully")
            return True
        else:
            print(f"✗ Model {model_name} failed to load")
            return False
    except Exception as e:
        print(f"✗ Error loading model {model_name}: {e}")
        return False


def ping_model(base_url: str, model_name: str, max_retries: int = 10):
    """Проверяет, что модель готова к работе через API"""
    print("Verifying model is ready...")

    for i in range(max_retries):
        try:
            # Проверяем доступность API
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                print("✓ API is responding")

                # Тестовый запрос
                print("Testing with a simple prompt...")
                test_response = requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": model_name,
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


def stop_all_models():
    """Останавливает все запущенные модели"""
    print("Stopping all Ollama models...")
    try:
        # Получаем список запущенных моделей
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, check=True
        )

        # Парсим вывод ollama ps и останавливаем каждую модель
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:  # Есть модели (первая строка - заголовок)
            # Пропускаем заголовок и обрабатываем каждую модель
            for line in lines[1:]:
                if line.strip():
                    # Первая колонка - это имя модели
                    model_name = line.split()[0]
                    print(f"Stopping model: {model_name}")
                    subprocess.run(["ollama", "stop", model_name], check=True)
            print("✓ All models stopped")
        else:
            print("✓ No models were running")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to stop models: {e}")
        return False


def cleanup_ollama():
    """
    Очистка после бенчмарка: остановка всех моделей

    Returns:
        bool: True если все успешно, False если есть ошибки
    """
    print("=" * 50)
    print("Ollama Cleanup")
    print("=" * 50)
    print()

    success = stop_all_models()

    print()
    if success:
        print("=" * 50)
        print("✓ Ollama cleanup complete!")
        print("=" * 50)
    else:
        print("=" * 50)
        print("⚠ Ollama cleanup completed with errors")
        print("=" * 50)
    print()

    return success


def setup_ollama(model_name: str, base_url: str):
    """
    Полная настройка Ollama: проверка, загрузка модели, запуск

    Args:
        model_name: Название модели для загрузки (например, "qwen2.5:32b")
        base_url: URL для API (например, "http://127.0.0.1:11434/v1")

    Returns:
        bool: True если все успешно, False если есть ошибки
    """
    print("=" * 50)
    print("Ollama Setup")
    print("=" * 50)
    print()

    # Шаг 1: Проверка CLI
    print("Step 1: Checking Ollama CLI...")
    if not check_ollama_cli():
        print("✗ Ollama CLI not found")
        print("Please install Ollama from https://ollama.ai/")
        return False
    print("✓ Ollama CLI found")
    print()

    # Шаг 2: Проверка/загрузка модели
    print("Step 2: Checking if model is downloaded...")
    print(f"Model: {model_name}")
    if not is_model_downloaded(model_name):
        print("Model not found, downloading...")
        if not pull_model(model_name):
            return False
    else:
        print("✓ Model already downloaded")
    print()

    # Шаг 3: Проверка/запуск сервера
    print("Step 3: Checking if Ollama server is running...")
    if not is_server_running():
        if not start_server():
            return False
    else:
        print("✓ Ollama server already running")
    print()

    # Шаг 4: Загрузка модели
    print("Step 4: Loading model...")
    if not is_model_running(model_name):
        if not run_model(model_name):
            return False
    else:
        print("✓ Model already loaded")
    print()

    # Шаг 5: Проверка готовности
    print("Step 5: Verifying model is ready...")
    if not ping_model(base_url, model_name):
        return False
    print()

    print("=" * 50)
    print("✓ Ollama is ready for benchmarking!")
    print("=" * 50)
    print()

    return True


if __name__ == "__main__":
    from src.settings import OLLAMA_LLM_MODEL_NAME, OLLAMA_LLM_BASE_URL

    setup_ollama(OLLAMA_LLM_MODEL_NAME, OLLAMA_LLM_BASE_URL)
