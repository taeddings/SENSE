#!/usr/bin/env python3
"""
SENSE-v2 Memory-Aware Implementation Verification Script

Run this script to verify the memory-aware implementation:
    python verify_implementation.py

Or run the full test suite:
    pytest tests/test_kernels.py tests/test_memory_aware.py -v
"""

import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def test_pass(msg):
    print(f"  [PASS] {msg}")

def test_fail(msg, err=None):
    print(f"  [FAIL] {msg}")
    if err:
        print(f"         Error: {err}")

def main():
    print("SENSE-v2 Memory-Aware Implementation Verification")
    print("=" * 60)

    all_passed = True

    # =========================================================
    section("1. Testing Module Imports")
    # =========================================================

    # Test kernel imports
    try:
        from sense_v2.llm.kernels import (
            get_backend, get_backend_info,
            supports_fused_kernels, TRITON_AVAILABLE
        )
        info = get_backend_info()
        test_pass(f"kernels/__init__.py - backend={info['backend']}")
    except Exception as e:
        test_fail("kernels/__init__.py", e)
        all_passed = False

    # Test functional
    try:
        from sense_v2.llm.kernels.functional import (
            fused_linear_cross_entropy,
            FusedLinearCrossEntropy,
            FusedLinearCrossEntropyLoss
        )
        test_pass("kernels/functional.py")
    except Exception as e:
        test_fail("kernels/functional.py", e)
        all_passed = False

    # Test triton_src (should import even without Triton)
    try:
        from sense_v2.llm.kernels import triton_src
        test_pass("kernels/triton_src.py")
    except Exception as e:
        test_fail("kernels/triton_src.py", e)
        all_passed = False

    # Test hashing
    try:
        from sense_v2.models.components.hashing import (
            MultiHeadHash, NGramExtractor, compute_ngram_hashes
        )
        test_pass("models/components/hashing.py")
    except Exception as e:
        test_fail("models/components/hashing.py", e)
        all_passed = False

    # Test hierarchy
    try:
        from sense_v2.memory.hierarchy import (
            EmbeddingPrefetcher, MemoryHierarchy, PinnedMemoryPool
        )
        test_pass("memory/hierarchy.py")
    except Exception as e:
        test_fail("memory/hierarchy.py", e)
        all_passed = False

    # Test MemoryAwareConfig
    try:
        from sense_v2.core.config import MemoryAwareConfig, Config
        cfg = Config()
        assert hasattr(cfg, 'memory_aware')
        test_pass("core/config.py - MemoryAwareConfig")
    except Exception as e:
        test_fail("core/config.py", e)
        all_passed = False

    # Test trainer updates
    try:
        from sense_v2.agents.agent_0.trainer import (
            get_memory_usage, compute_memory_aware_fitness,
            MemoryAwareRewardComponents
        )
        test_pass("agents/agent_0/trainer.py - memory functions")
    except Exception as e:
        test_fail("agents/agent_0/trainer.py", e)
        all_passed = False

    # Test master agent updates
    try:
        from sense_v2.agents.agent_zero.master import MasterAgent
        agent = MasterAgent()
        assert hasattr(agent, '_check_resources')
        assert hasattr(agent, 'get_resource_recommendations')
        test_pass("agents/agent_zero/master.py - resource checks")
    except Exception as e:
        test_fail("agents/agent_zero/master.py", e)
        all_passed = False

    # Test MemoryProfileTool
    try:
        from sense_v2.tools.memory_tools import MemoryProfileTool
        tool = MemoryProfileTool()
        test_pass("tools/memory_tools.py - MemoryProfileTool")
    except Exception as e:
        test_fail("tools/memory_tools.py", e)
        all_passed = False

    # =========================================================
    section("2. Testing Kernel Functionality")
    # =========================================================

    try:
        import torch
        from sense_v2.llm.kernels.functional import fused_linear_cross_entropy
        import torch.nn.functional as F

        batch_size = 8
        hidden_dim = 64
        vocab_size = 100

        hidden = torch.randn(batch_size, hidden_dim, requires_grad=True)
        weight = torch.randn(vocab_size, hidden_dim, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size,))

        # Test fused forward
        loss_fused = fused_linear_cross_entropy(hidden, weight, labels)
        test_pass(f"Fused forward - loss={loss_fused.item():.4f}")

        # Test backward
        loss_fused.backward()
        assert hidden.grad is not None
        test_pass(f"Fused backward - grad shape={hidden.grad.shape}")

        # Compare to naive
        hidden2 = hidden.detach().clone()
        logits = hidden2 @ weight.detach().T
        loss_naive = F.cross_entropy(logits, labels)
        diff = abs(loss_fused.item() - loss_naive.item())
        if diff < 0.01:
            test_pass(f"Numerical parity - diff={diff:.6f}")
        else:
            test_fail(f"Numerical parity - diff={diff:.6f} (expected < 0.01)")
            all_passed = False

    except Exception as e:
        test_fail("Kernel functionality", e)
        all_passed = False

    # =========================================================
    section("3. Testing N-gram Hashing")
    # =========================================================

    try:
        import torch
        from sense_v2.models.components.hashing import (
            MultiHeadHash, NGramExtractor, compute_ngram_hashes
        )

        # Test MultiHeadHash
        hasher = MultiHeadHash(num_heads=8, table_size=10000, seed=42)
        ngrams = torch.randint(0, 1000, (4, 16, 3))
        hashes = hasher(ngrams)
        assert hashes.shape == (4, 16, 8)
        assert (hashes >= 0).all() and (hashes < 10000).all()
        test_pass(f"MultiHeadHash - shape={hashes.shape}")

        # Test determinism
        hashes2 = hasher(ngrams)
        assert torch.equal(hashes, hashes2)
        test_pass("MultiHeadHash - deterministic")

        # Test NGramExtractor
        extractor = NGramExtractor(ngram_orders=[2, 3])
        input_ids = torch.randint(0, 1000, (4, 16))
        ngrams, mask = extractor(input_ids)
        assert ngrams.shape == (4, 16, 3)
        test_pass(f"NGramExtractor - shape={ngrams.shape}")

        # Test end-to-end
        hashes = compute_ngram_hashes(input_ids, hasher, extractor)
        test_pass(f"compute_ngram_hashes - shape={hashes.shape}")

    except Exception as e:
        test_fail("N-gram hashing", e)
        all_passed = False

    # =========================================================
    section("4. Testing Configuration")
    # =========================================================

    try:
        from sense_v2.core.config import MemoryAwareConfig

        cfg = MemoryAwareConfig()

        # Test thresholds
        assert not cfg.should_activate_engram(0.50)
        assert cfg.should_activate_engram(0.70)
        test_pass("MemoryAwareConfig.should_activate_engram()")

        assert not cfg.should_use_fused_kernels(0.70)
        assert cfg.should_use_fused_kernels(0.80)
        test_pass("MemoryAwareConfig.should_use_fused_kernels()")

    except Exception as e:
        test_fail("Configuration", e)
        all_passed = False

    # =========================================================
    section("5. Testing Memory-Aware Fitness")
    # =========================================================

    try:
        from sense_v2.agents.agent_0.trainer import (
            compute_memory_aware_fitness, get_memory_usage
        )

        # Test fitness function
        fitness_low = compute_memory_aware_fitness(accuracy=0.9, memory_mb=100.0)
        fitness_high = compute_memory_aware_fitness(accuracy=0.9, memory_mb=2000.0)
        assert fitness_low > fitness_high
        test_pass(f"Memory penalty - low={fitness_low:.4f} > high={fitness_high:.4f}")

        # Test memory usage
        usage = get_memory_usage()
        test_pass(f"get_memory_usage() - psutil={usage.get('psutil_available', False)}")

    except Exception as e:
        test_fail("Memory-aware fitness", e)
        all_passed = False

    # =========================================================
    section("6. Testing Engram Updates")
    # =========================================================

    try:
        from sense_v2.engram.model import EngramFusionLayer
        from sense_v2.models.components.hashing import MultiHeadHash

        # Check that EngramFusionLayer uses proper hashing
        # We can't fully instantiate without files, but we can check the class
        import inspect
        source = inspect.getsource(EngramFusionLayer.__init__)
        assert 'MultiHeadHash' in source
        assert 'NGramExtractor' in source
        test_pass("EngramFusionLayer uses MultiHeadHash")

        # Check _compute_ngram_hashes method exists
        assert hasattr(EngramFusionLayer, '_compute_ngram_hashes')
        test_pass("EngramFusionLayer._compute_ngram_hashes() exists")

    except Exception as e:
        test_fail("Engram updates", e)
        all_passed = False

    # =========================================================
    section("Summary")
    # =========================================================

    print()
    if all_passed:
        print("All verifications PASSED!")
        print("\nTo run the full test suite:")
        print("  pytest tests/test_kernels.py tests/test_memory_aware.py -v")
        return 0
    else:
        print("Some verifications FAILED!")
        print("Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
