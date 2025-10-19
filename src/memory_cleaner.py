import gc
import torch


def clear_memory():
    """Очистка памяти независимо от устройства"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    print("Memory cleared\n")
