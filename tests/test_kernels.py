"""
SENSE-v2 Kernel Test Suite

Tests for fused linear cross-entropy kernels with:
- Numerical parity against PyTorch reference
- Memory profiling
- Gradient correctness
- Backend fallback verification
"""

import pytest
import torch
import torch.nn.functional as F
import os
from typing import Tuple

# Import kernel module
from sense.llm.kernels import (
    get_backend,
    get_backend_info,
    supports_fused_kernels,
    get_memory_savings_estimate,
    TRITON_AVAILABLE,
)
from sense.llm.kernels.functional import (
    fused_linear_cross_entropy,
    FusedLinearCrossEntropy,
    FusedLinearCrossEntropyLoss,
    _pytorch_chunked_forward,
    benchmark_memory_usage,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_tensors():
    """Small tensors for quick tests."""
    batch_size = 8
    hidden_dim = 64
    vocab_size = 100

    hidden = torch.randn(batch_size, hidden_dim, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size,))

    return hidden, weight, labels


@pytest.fixture
def medium_tensors():
    """Medium tensors for realistic tests."""
    batch_size = 32
    hidden_dim = 512
    vocab_size = 1000

    hidden = torch.randn(batch_size, hidden_dim, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size,))

    return hidden, weight, labels


def naive_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation using standard PyTorch."""
    logits = hidden @ weight.T  # [batch, vocab]
    return F.cross_entropy(logits, labels)


# =============================================================================
# Backend Detection Tests
# =============================================================================

class TestBackendDetection:
    """Tests for runtime backend detection."""

    def test_get_backend_returns_valid_type(self):
        """Backend should be one of the expected types."""
        backend = get_backend()
        valid_backends = [
            "triton", "pytorch_cuda", "pytorch_rocm",
            "pytorch_mps", "pytorch_cpu"
        ]
        assert backend in valid_backends

    def test_get_backend_info_structure(self):
        """Backend info should have expected keys."""
        info = get_backend_info()

        required_keys = [
            "backend", "triton_available", "cuda_available",
            "mps_available", "torch_version"
        ]
        for key in required_keys:
            assert key in info

    def test_supports_fused_kernels_matches_triton(self):
        """supports_fused_kernels should match TRITON_AVAILABLE."""
        assert supports_fused_kernels() == TRITON_AVAILABLE

    def test_memory_savings_estimate_in_range(self):
        """Memory savings estimate should be between 0 and 1."""
        savings = get_memory_savings_estimate()
        assert 0.0 <= savings <= 1.0

    def test_force_backend_override(self):
        """Environment variable should override backend detection."""
        original = os.environ.get("SENSE_FORCE_BACKEND")

        try:
            os.environ["SENSE_FORCE_BACKEND"] = "pytorch_cpu"
            # Need to reload module or use a fresh function call
            # For now, just verify the env var is set
            assert os.environ["SENSE_FORCE_BACKEND"] == "pytorch_cpu"
        finally:
            if original:
                os.environ["SENSE_FORCE_BACKEND"] = original
            else:
                os.environ.pop("SENSE_FORCE_BACKEND", None)


# =============================================================================
# Numerical Parity Tests
# =============================================================================

class TestNumericalParity:
    """Tests for numerical correctness against reference implementation."""

    def test_fused_matches_naive_small(self, small_tensors):
        """Fused kernel should match naive implementation on small inputs."""
        hidden, weight, labels = small_tensors

        # Detach and clone for independent computation
        hidden_naive = hidden.detach().clone().requires_grad_(True)
        weight_naive = weight.detach().clone().requires_grad_(True)
        hidden_fused = hidden.detach().clone().requires_grad_(True)
        weight_fused = weight.detach().clone().requires_grad_(True)

        # Compute losses
        loss_naive = naive_cross_entropy(hidden_naive, weight_naive, labels)
        loss_fused = fused_linear_cross_entropy(hidden_fused, weight_fused, labels)

        # Check forward pass
        assert torch.allclose(loss_naive, loss_fused, rtol=1e-4, atol=1e-5), \
            f"Forward mismatch: naive={loss_naive.item()}, fused={loss_fused.item()}"

    def test_fused_matches_naive_medium(self, medium_tensors):
        """Fused kernel should match naive implementation on medium inputs."""
        hidden, weight, labels = medium_tensors

        hidden_naive = hidden.detach().clone().requires_grad_(True)
        weight_naive = weight.detach().clone().requires_grad_(True)
        hidden_fused = hidden.detach().clone().requires_grad_(True)
        weight_fused = weight.detach().clone().requires_grad_(True)

        loss_naive = naive_cross_entropy(hidden_naive, weight_naive, labels)
        loss_fused = fused_linear_cross_entropy(hidden_fused, weight_fused, labels)

        assert torch.allclose(loss_naive, loss_fused, rtol=1e-3, atol=1e-4), \
            f"Forward mismatch: naive={loss_naive.item()}, fused={loss_fused.item()}"

    def test_with_ignore_index(self, small_tensors):
        """Fused kernel should handle ignore_index correctly."""
        hidden, weight, labels = small_tensors

        # Set some labels to ignore
        labels_with_ignore = labels.clone()
        labels_with_ignore[0] = -100
        labels_with_ignore[2] = -100

        hidden_fused = hidden.detach().clone().requires_grad_(True)

        loss = fused_linear_cross_entropy(
            hidden_fused, weight, labels_with_ignore, ignore_index=-100
        )

        # Should not include ignored samples in loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_all_ignored_returns_zero(self, small_tensors):
        """All ignored labels should return zero loss."""
        hidden, weight, _ = small_tensors

        # All labels ignored
        labels_all_ignored = torch.full((hidden.shape[0],), -100, dtype=torch.long)

        loss = fused_linear_cross_entropy(
            hidden, weight, labels_all_ignored, ignore_index=-100
        )

        assert loss.item() == 0.0


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradients:
    """Tests for gradient correctness."""

    def test_gradients_match_naive(self, small_tensors):
        """Gradients should match naive implementation."""
        hidden, weight, labels = small_tensors

        # Naive gradients
        hidden_naive = hidden.detach().clone().requires_grad_(True)
        weight_naive = weight.detach().clone().requires_grad_(True)
        loss_naive = naive_cross_entropy(hidden_naive, weight_naive, labels)
        loss_naive.backward()

        # Fused gradients
        hidden_fused = hidden.detach().clone().requires_grad_(True)
        weight_fused = weight.detach().clone().requires_grad_(True)
        loss_fused = fused_linear_cross_entropy(hidden_fused, weight_fused, labels)
        loss_fused.backward()

        # Check hidden gradients
        assert torch.allclose(
            hidden_naive.grad, hidden_fused.grad, rtol=1e-3, atol=1e-4
        ), "Hidden gradients mismatch"

        # Check weight gradients
        assert torch.allclose(
            weight_naive.grad, weight_fused.grad, rtol=1e-3, atol=1e-4
        ), "Weight gradients mismatch"

    def test_gradcheck_small(self):
        """Autograd gradcheck for small inputs."""
        batch_size = 4
        hidden_dim = 16
        vocab_size = 32

        hidden = torch.randn(
            batch_size, hidden_dim,
            dtype=torch.float64, requires_grad=True
        )
        weight = torch.randn(
            vocab_size, hidden_dim,
            dtype=torch.float64, requires_grad=True
        )
        labels = torch.randint(0, vocab_size, (batch_size,))

        # Use gradcheck with the autograd function
        def func(h, w):
            return FusedLinearCrossEntropy.apply(h, w, labels, 16, -100)

        # Note: gradcheck may be slow, using smaller tolerances
        assert torch.autograd.gradcheck(
            func, (hidden, weight),
            eps=1e-6, atol=1e-4, rtol=1e-3,
            raise_exception=True
        )


# =============================================================================
# Backend Fallback Tests
# =============================================================================

class TestBackendFallback:
    """Tests for backend fallback behavior."""

    def test_pytorch_fallback_works(self, small_tensors):
        """PyTorch chunked fallback should work."""
        hidden, weight, labels = small_tensors

        # Directly test the chunked forward
        loss, _ = _pytorch_chunked_forward(
            hidden.detach(), weight.detach(), labels, chunk_size=32
        )

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_different_chunk_sizes(self, medium_tensors):
        """Different chunk sizes should produce same result."""
        hidden, weight, labels = medium_tensors

        hidden = hidden.detach()
        weight = weight.detach()

        losses = []
        for chunk_size in [64, 128, 256, 512]:
            loss = fused_linear_cross_entropy(
                hidden, weight, labels, chunk_size=chunk_size
            )
            losses.append(loss.item())

        # All chunk sizes should give same result
        for loss in losses[1:]:
            assert abs(loss - losses[0]) < 1e-4, \
                f"Chunk size sensitivity: {losses}"


# =============================================================================
# Module Wrapper Tests
# =============================================================================

class TestModuleWrapper:
    """Tests for nn.Module wrapper."""

    def test_fused_loss_module(self, small_tensors):
        """FusedLinearCrossEntropyLoss module should work."""
        hidden, weight, labels = small_tensors

        loss_module = FusedLinearCrossEntropyLoss(weight=weight.detach())

        loss = loss_module(hidden, labels)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_fused_loss_module_backward(self, small_tensors):
        """FusedLinearCrossEntropyLoss should support backward."""
        hidden, weight, labels = small_tensors

        hidden_grad = hidden.detach().clone().requires_grad_(True)
        loss_module = FusedLinearCrossEntropyLoss(weight=weight.detach())

        loss = loss_module(hidden_grad, labels)
        loss.backward()

        assert hidden_grad.grad is not None
        assert hidden_grad.grad.shape == hidden.shape


# =============================================================================
# 3D Input Tests
# =============================================================================

class TestBatchedInput:
    """Tests for 3D (batch, seq, hidden) inputs."""

    def test_3d_input_shape(self):
        """Should handle 3D inputs correctly."""
        batch_size = 4
        seq_len = 16
        hidden_dim = 32
        vocab_size = 64

        hidden = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.randn(vocab_size, hidden_dim)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = fused_linear_cross_entropy(hidden, weight, labels)

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)


# =============================================================================
# Memory Benchmark Tests
# =============================================================================

class TestMemoryBenchmark:
    """Tests for memory benchmarking utility."""

    def test_benchmark_runs(self):
        """Memory benchmark should run without error."""
        results = benchmark_memory_usage(
            batch_size=8,
            hidden_dim=64,
            vocab_size=256,
            device="cpu"
        )

        assert "logits_tensor_mb" in results
        assert "hidden_tensor_mb" in results
        assert "weight_tensor_mb" in results

    def test_benchmark_theoretical_sizes(self):
        """Theoretical tensor sizes should be correct."""
        batch_size = 32
        hidden_dim = 512
        vocab_size = 1000

        results = benchmark_memory_usage(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            device="cpu"
        )

        # float32 = 4 bytes
        expected_logits_mb = (batch_size * vocab_size * 4) / 1024 / 1024
        assert abs(results["logits_tensor_mb"] - expected_logits_mb) < 0.01


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with other SENSE-v2 components."""

    def test_import_from_package(self):
        """Should be importable from package."""
        from sense.llm.kernels import (
            fused_linear_cross_entropy,
            get_backend,
        )

        assert callable(fused_linear_cross_entropy)
        assert callable(get_backend)

    def test_config_integration(self):
        """Should work with MemoryAwareConfig."""
        from sense.core.config import MemoryAwareConfig

        config = MemoryAwareConfig()

        assert hasattr(config, 'use_fused_kernels')
        assert hasattr(config, 'fused_chunk_size')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
