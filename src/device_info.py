import torch
import platform
import psutil
import subprocess
import json

def get_fastfetch_info():
    """Получает информацию о CPU/GPU через fastfetch"""
    try:
        result = subprocess.run(
            ['fastfetch', '--format', 'json'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            cpu_info = None
            gpu_info = None

            for item in data:
                item_type = item.get('type', '')

                if item_type == 'CPU':
                    cpu_result = item.get('result', {})
                    cpu_name = cpu_result.get('cpu')
                    cores = cpu_result.get('cores', {})
                    physical_cores = cores.get('physical', '')
                    if cpu_name:
                        cpu_info = f"{cpu_name} ({physical_cores})" if physical_cores else cpu_name

                elif item_type == 'GPU':
                    gpu_result = item.get('result', [])
                    if isinstance(gpu_result, list) and len(gpu_result) > 0:
                        gpu_data = gpu_result[0]
                        vendor = gpu_data.get('vendor', '')
                        name = gpu_data.get('name', '')
                        core_count = gpu_data.get('coreCount', '')

                        # Избегаем дублирования vendor если он уже в name
                        if vendor and name and not name.startswith(vendor):
                            gpu_info = f"{vendor} {name}"
                        elif name:
                            gpu_info = name
                        else:
                            gpu_info = vendor

                        # Добавляем количество ядер если есть
                        if core_count and gpu_info:
                            gpu_info = f"{gpu_info} ({core_count} cores)"

            return cpu_info, gpu_info
    except Exception as e:
        print(f"Warning: Could not get fastfetch info: {e}")

    return None, None

def get_device_info():
    """Собирает информацию об устройстве, GPU/CPU и памяти"""
    # Пытаемся получить информацию из fastfetch
    cpu_name, gpu_name_fastfetch = get_fastfetch_info()

    info = {
        "platform": platform.system(),
        "processor": cpu_name or platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "device": "cpu"
    }

    # Проверка CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        info["cuda_version"] = torch.version.cuda

    # Проверка MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["gpu_name"] = gpu_name_fastfetch or "Apple Silicon GPU"
        info["gpu_memory_gb"] = "shared with system RAM"

    return info