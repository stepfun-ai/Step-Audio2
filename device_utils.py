"""Device utility functions for MPS and CUDA compatibility."""

import torch


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def is_mps_available():
    """Check if MPS backend is available."""
    return torch.backends.mps.is_available()


def is_cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def set_manual_seed(seed):
    """Set manual seed for all available backends."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def empty_cache():
    """Empty cache for the current device."""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize():
    """Synchronize the current device."""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.synchronize()
    # MPS doesn't have a synchronize equivalent


def to_device_efficient(tensor, device=None):
    """Move tensor to device efficiently with pinned memory for CUDA."""
    if device is None:
        device = get_device()
    
    if device.type == "cuda":
        return tensor.cuda(non_blocking=True) if tensor.is_pinned() else tensor.to(device)
    else:
        return tensor.to(device)