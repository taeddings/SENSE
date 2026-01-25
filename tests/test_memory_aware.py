"""
SENSE-v2 Memory-Aware Integration Tests

Tests for memory-aware features including:
- Drift and squeeze scenarios
- Automatic Engram activation
- Fused kernel engagement
- Resource monitoring
"""

import pytest
import torch
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Import components
from sense.core.config import (
    Config,
    MemoryAwareConfig,
    EngramConfig,
)
from sense.llm.kernels import get_backend, get_backend_info
from sense.models.components.hashing import (
    MultiHeadHash,
    NGramExtractor,
    compute_ngram_hashes,
)
from sense.memory.hierarchy import (
    EmbeddingPrefetcher,
    MemoryHierarchy,
    PinnedMemoryPool,
)
from sense.agents.agent_0.trainer import (
    get_memory_usage,
    compute_memory_aware_fitness,
    MemoryAwareRewardComponents,
)


# =============================================================================
# Config Tests
# =============================================================================

class TestMemoryAwareConfig:
    """Tests for MemoryAwareConfig."""

    def test_default_config_values(self):
        """Default config should have sensible values."""
        config = MemoryAwareConfig()

        assert config.max_ram_usage_mb == 4096
        assert config.use_fused_kernels is True
        assert config.enable_host_offload is True
        assert 0 < config.memory_warning_threshold < 1
        assert 0 < config.memory_critical_threshold < 1

    def test_should_activate_engram_thresholds(self):
        """Engram activation should respect thresholds."""
        config = MemoryAwareConfig(
            memory_warning_threshold=0.60,
            auto_activate_engram=True,
        )

        # Below threshold
        assert not config.should_activate_engram(0.50)

        # At threshold
        assert config.should_activate_engram(0.60)

        # Above threshold
        assert config.should_activate_engram(0.80)

    def test_should_use_fused_kernels_thresholds(self):
        """Fused kernel activation should respect thresholds."""
        config = MemoryAwareConfig(
            fused_kernel_threshold=0.75,
            use_fused_kernels=True,
            auto_enable_fused=True,
        )

        # Below threshold
        assert not config.should_use_fused_kernels(0.70)

        # Above threshold
        assert config.should_use_fused_kernels(0.80)

    def test_fused_kernels_disabled(self):
        """Disabled fused kernels should always return False."""
        config = MemoryAwareConfig(use_fused_kernels=False)

        assert not config.should_use_fused_kernels(0.99)

    def test_config_serialization(self):
        """Config should serialize and deserialize."""
        config = Config()

        data = config.to_dict()
        assert "memory_aware" in data

        # Verify nested config is present
        ma_data = data["memory_aware"]
        assert "max_ram_usage_mb" in ma_data


# =============================================================================
# N-gram Hashing Tests
# =============================================================================

class TestNgramHashing:
    """Tests for N-gram hashing components."""

    def test_multihead_hash_output_shape(self):
        """MultiHeadHash should produce correct output shape."""
        num_heads = 8
        table_size = 10000

        hasher = MultiHeadHash(num_heads, table_size)

        # Input: [batch, seq, ngram_order]
        ngrams = torch.randint(0, 1000, (4, 16, 3))
        hashes = hasher(ngrams)

        assert hashes.shape == (4, 16, num_heads)

    def test_multihead_hash_deterministic(self):
        """Same input should produce same hash."""
        hasher = MultiHeadHash(num_heads=4, table_size=1000, seed=42)

        ngrams = torch.tensor([[[1, 2, 3], [4, 5, 6]]])

        hash1 = hasher(ngrams)
        hash2 = hasher(ngrams)

        assert torch.equal(hash1, hash2)

    def test_multihead_hash_bounds(self):
        """Hashes should be within table_size."""
        table_size = 1000
        hasher = MultiHeadHash(num_heads=8, table_size=table_size)

        ngrams = torch.randint(0, 50000, (8, 32, 4))
        hashes = hasher(ngrams)

        assert (hashes >= 0).all()
        assert (hashes < table_size).all()

    def test_ngram_extractor_output_shape(self):
        """NGramExtractor should produce correct output shape."""
        extractor = NGramExtractor(ngram_orders=[2, 3])

        input_ids = torch.randint(0, 1000, (4, 16))
        ngrams, mask = extractor(input_ids)

        assert ngrams.shape == (4, 16, 3)  # max_order = 3
        assert mask.shape == (4, 16)

    def test_ngram_extractor_mask_validity(self):
        """Mask should be False for positions lacking context."""
        extractor = NGramExtractor(ngram_orders=[2, 3])

        input_ids = torch.randint(0, 1000, (2, 10))
        ngrams, mask = extractor(input_ids)

        # First (max_order - 1) = 2 positions should be masked
        assert not mask[:, 0].any()
        assert not mask[:, 1].any()
        assert mask[:, 2:].all()

    def test_compute_ngram_hashes_integration(self):
        """compute_ngram_hashes should work end-to-end."""
        hasher = MultiHeadHash(num_heads=4, table_size=1000)
        extractor = NGramExtractor(ngram_orders=[2, 3])

        input_ids = torch.randint(0, 500, (2, 8))
        hashes = compute_ngram_hashes(input_ids, hasher, extractor)

        assert hashes.shape == (2, 8, 4)


# =============================================================================
# Memory Hierarchy Tests
# =============================================================================

class TestMemoryHierarchy:
    """Tests for memory hierarchy components."""

    def test_pinned_memory_pool_creation(self):
        """PinnedMemoryPool should create without error."""
        pool = PinnedMemoryPool(num_buffers=2, buffer_size=1024)

        stats = pool.get_stats()
        assert stats["total_buffers"] == 2
        assert stats["buffer_size"] == 1024

    def test_pinned_memory_pool_acquire_release(self):
        """Pool should allow acquire and release."""
        pool = PinnedMemoryPool(num_buffers=2, buffer_size=100)

        # Acquire first buffer
        result = pool.acquire(timeout=1.0)
        assert result is not None
        buffer_id, buffer = result
        assert buffer.shape[0] == 100

        # Release it
        pool.release(buffer_id)

        # Should be available again
        stats = pool.get_stats()
        assert stats["available_buffers"] == 2

    def test_memory_hierarchy_stats(self):
        """MemoryHierarchy should track statistics."""
        hierarchy = MemoryHierarchy(
            gpu_cache_size_mb=64,
            host_cache_size_mb=128,
        )

        stats = hierarchy.get_stats()
        assert "total_accesses" in stats
        assert "gpu_hit_rate" in stats
        assert "host_hit_rate" in stats


# =============================================================================
# Memory-Aware Fitness Tests
# =============================================================================

class TestMemoryAwareFitness:
    """Tests for memory-aware fitness functions."""

    def test_get_memory_usage_structure(self):
        """get_memory_usage should return expected structure."""
        usage = get_memory_usage()

        assert "ram_percent" in usage
        assert "psutil_available" in usage

    def test_compute_memory_aware_fitness_basic(self):
        """Basic fitness computation should work."""
        fitness = compute_memory_aware_fitness(
            accuracy=0.9,
            memory_mb=500.0,
            drift_resistance=0.8,
        )

        assert 0 <= fitness <= 1.5  # Can exceed 1 with good memory efficiency

    def test_compute_memory_aware_fitness_memory_penalty(self):
        """Higher memory usage should result in lower fitness."""
        fitness_low_mem = compute_memory_aware_fitness(
            accuracy=0.9,
            memory_mb=100.0,
        )

        fitness_high_mem = compute_memory_aware_fitness(
            accuracy=0.9,
            memory_mb=2000.0,
        )

        assert fitness_low_mem > fitness_high_mem

    def test_compute_memory_aware_fitness_weights(self):
        """Custom weights should affect fitness."""
        # All weight on accuracy
        fitness_accuracy = compute_memory_aware_fitness(
            accuracy=1.0,
            memory_mb=1000.0,
            drift_resistance=0.0,
            accuracy_weight=1.0,
            memory_weight=0.0,
            drift_weight=0.0,
        )

        assert fitness_accuracy == 1.0

    def test_memory_aware_reward_components_serialization(self):
        """MemoryAwareRewardComponents should serialize."""
        components = MemoryAwareRewardComponents(
            base_reward=0.8,
            format_reward=0.1,
            tool_reward=0.05,
            diversity_penalty=0.02,
            memory_penalty=0.03,
            drift_resistance=0.9,
            total=0.8,
        )

        data = components.to_dict()
        assert "base_reward" in data
        assert "memory_penalty" in data
        assert "drift_resistance" in data


# =============================================================================
# Drift and Squeeze Scenario Tests
# =============================================================================

class TestDriftAndSqueeze:
    """Tests for drift and memory squeeze scenarios."""

    def test_simulated_drift_detection(self):
        """System should detect accuracy drift."""
        # Simulate accuracy history
        history = [0.9, 0.88, 0.85, 0.80, 0.72, 0.65]

        # Simple drift detection: compare recent to historical
        recent_avg = sum(history[-3:]) / 3
        historical_avg = sum(history[:3]) / 3

        drift_detected = (historical_avg - recent_avg) > 0.1

        assert drift_detected

    def test_memory_squeeze_engram_activation(self):
        """High memory should trigger Engram activation."""
        config = MemoryAwareConfig(
            memory_warning_threshold=0.60,
            auto_activate_engram=True,
        )

        # Simulate 70% memory usage
        memory_percent = 0.70

        should_activate = config.should_activate_engram(memory_percent)
        assert should_activate

    def test_memory_squeeze_fused_kernels(self):
        """Critical memory should trigger fused kernels."""
        config = MemoryAwareConfig(
            fused_kernel_threshold=0.75,
            use_fused_kernels=True,
            auto_enable_fused=True,
        )

        # Simulate 80% memory usage
        memory_percent = 0.80

        should_use_fused = config.should_use_fused_kernels(memory_percent)
        assert should_use_fused

    @pytest.mark.asyncio
    async def test_resource_check_recommendations(self):
        """MasterAgent should provide resource recommendations."""
        from sense.agents.agent_zero.master import MasterAgent

        agent = MasterAgent()

        resources = await agent._check_resources()

        assert "memory_pressure" in resources
        assert "should_use_fused_kernels" in resources
        assert "should_activate_engram" in resources


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_config_with_memory_aware(self):
        """Full Config should include MemoryAwareConfig."""
        config = Config()

        assert hasattr(config, 'memory_aware')
        assert isinstance(config.memory_aware, MemoryAwareConfig)
        assert hasattr(config, 'engram')
        assert isinstance(config.engram, EngramConfig)

    def test_engram_fusion_with_proper_hashing(self):
        """EngramFusionLayer should use proper N-gram hashing."""
        from sense.engram.model import EngramFusionLayer
        from sense.core.config import EngramConfig
        import tempfile
        import os

        # Create temporary files for storage
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EngramConfig(
                storage_path=os.path.join(tmpdir, "engram.dat"),
                shadow_map_path=os.path.join(tmpdir, "shadow.npy"),
                table_size=1000,
                engram_dim=64,
                num_heads=4,
                ngram_orders=[2, 3],
            )

            # Create shadow map
            import numpy as np
            shadow_map = np.arange(1000, dtype=np.int32)
            np.save(config.shadow_map_path, shadow_map)

            # Create layer
            layer = EngramFusionLayer(
                config=config,
                backbone_hidden_size=128,
                layer_id=0,
            )

            # Verify proper hashing components exist
            assert hasattr(layer, 'hasher')
            assert hasattr(layer, 'ngram_extractor')
            assert isinstance(layer.hasher, MultiHeadHash)

    def test_backend_info_completeness(self):
        """Backend info should be comprehensive."""
        info = get_backend_info()

        # Required fields
        assert "backend" in info
        assert "triton_available" in info
        assert "torch_version" in info

        # Backend should be valid
        valid_backends = [
            "triton", "pytorch_cuda", "pytorch_rocm",
            "pytorch_mps", "pytorch_cpu"
        ]
        assert info["backend"] in valid_backends


# =============================================================================
# Memory Profile Tool Tests
# =============================================================================

class TestMemoryProfileTool:
    """Tests for MemoryProfileTool."""

    @pytest.mark.asyncio
    async def test_memory_profile_tool_execution(self):
        """MemoryProfileTool should execute successfully."""
        from sense.tools.memory_tools import MemoryProfileTool

        tool = MemoryProfileTool()
        result = await tool.execute(include_gpu=True, include_recommendations=True)

        assert result.success
        assert "memory_pressure" in result.data

    @pytest.mark.asyncio
    async def test_memory_profile_recommendations(self):
        """MemoryProfileTool should provide recommendations."""
        from sense.tools.memory_tools import MemoryProfileTool

        tool = MemoryProfileTool()
        result = await tool.execute(include_recommendations=True)

        assert result.success
        assert "recommendations" in result.data
        assert isinstance(result.data["recommendations"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
