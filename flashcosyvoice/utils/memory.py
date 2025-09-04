import os

import torch

from device_utils import get_device


try:
    from pynvml import *  # noqa
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def get_gpu_memory():
    device = get_device()
    
    if device.type == "mps":
        # MPS doesn't provide detailed memory stats like CUDA
        # Return placeholder values for compatibility
        return 8 * 1024 * 1024 * 1024, 0, 8 * 1024 * 1024 * 1024  # 8GB total, 0 used, 8GB free
    
    elif device.type == "cuda" and NVML_AVAILABLE:
        torch.cuda.synchronize()
        nvmlInit()
        visible_device = list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(',')))
        cuda_device_idx = torch.cuda.current_device()
        cuda_device_idx = visible_device[cuda_device_idx]
        handle = nvmlDeviceGetHandleByIndex(cuda_device_idx)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        total_memory = mem_info.total
        used_memory = mem_info.used
        free_memory = mem_info.free
        nvmlShutdown()
        return total_memory, used_memory, free_memory
    
    else:
        # Fallback for CPU or when NVML is not available
        return 8 * 1024 * 1024 * 1024, 0, 8 * 1024 * 1024 * 1024  # 8GB placeholder
