"""
SENSE-v2 Memory-Aware Kernels

Cross-platform fused kernel implementations with runtime backend detection.
Supports Triton (CUDA/ROCm), PyTorch CUDA, MPS, and CPU fallbacks.
"""

import torch
from typing import Literal
import os

# Runtime feature detection
TRITON_AVAILABLE = False
_TRITON_VERSION = None

try:
    import triton
    import triton.language as tl
    _TRITON_VERSION = getattr(triton, "__version__", "unknown")
    # Triton requires CUDA or ROCm
    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

# Environment override for testing
_FORCE_BACKEND = os.environ.get("SENSE_FORCE_BACKEND", None)

BackendType = Literal["triton", "pytorch_cuda", "pytorch_rocm", "pytorch_mps", "pytorch_cpu"]


def get_backend() -> BackendType:
    """
    Return the best available backend for fused kernels.

    Priority order:
    1. Triton (CUDA/ROCm) - ~84% memory savings
    2. PyTorch CUDA - chunked fallback
    3. PyTorch ROCm - chunked fallback
    4. PyTorch MPS (Apple Silicon) - ~50-60% memory savings
    5. PyTorch CPU - universal fallback

    Returns:
        Backend identifier string
    """
    # Allow environment override for testing
    if _FORCE_BACKEND:
        return _FORCE_BACKEND

    if TRITON_AVAILABLE:
        return "triton"
    elif torch.cuda.is_available():
        # Check if ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return "pytorch_rocm"
        return "pytorch_cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "pytorch_mps"
    return "pytorch_cpu"


def get_backend_info() -> dict:
    """
    Get detailed information about the current backend configuration.

    Returns:
        Dictionary with backend details
    """
    backend = get_backend()

    info = {
        "backend": backend,
        "triton_available": TRITON_AVAILABLE,
        "triton_version": _TRITON_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
        "force_override": _FORCE_BACKEND,
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        # Check for ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            info["rocm_version"] = torch.version.hip

    return info


def supports_fused_kernels() -> bool:
    """Check if the current backend supports memory-efficient fused kernels."""
    return TRITON_AVAILABLE


def get_memory_savings_estimate() -> float:
    """
    Estimate memory savings percentage for the current backend.

    Returns:
        Estimated memory savings as a fraction (0.0 to 1.0)
    """
    backend = get_backend()

    # Empirical estimates based on benchmarks
    savings = {
        "triton": 0.84,        # ~84% savings with full fusion
        "pytorch_cuda": 0.55,  # ~55% with chunked approach
        "pytorch_rocm": 0.55,  # ~55% with chunked approach
        "pytorch_mps": 0.50,   # ~50% with chunked approach
        "pytorch_cpu": 0.50,   # ~50% with chunked approach
    }

    return savings.get(backend, 0.0)


# Import functional API for convenience
from sense.llm.kernels.functional import (
    fused_linear_cross_entropy,
    FusedLinearCrossEntropy,
)

__all__ = [
    "get_backend",
    "get_backend_info",
    "supports_fused_kernels",
    "get_memory_savings_estimate",
    "TRITON_AVAILABLE",
    "fused_linear_cross_entropy",
    "FusedLinearCrossEntropy",
]
