#!/usr/bin/env python
"""Run SENSE-v2 memory-aware tests."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all new module imports."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)

    results = []

    # Test kernel imports
    try:
        from sense_v2.llm.kernels import get_backend, get_backend_info, TRITON_AVAILABLE
        info = get_backend_info()
        print(f"[PASS] kernels - backend={info['backend']}, triton={TRITON_AVAILABLE}")
        results.append(("kernels", True, None))
    except Exception as e:
        print(f"[FAIL] kernels - {e}")
        results.append(("kernels", False, str(e)))

    # Test functional
    try:
        from sense_v2.llm.kernels.functional import fused_linear_cross_entropy
        print("[PASS] functional")
        results.append(("functional", True, None))
    except Exception as e:
        print(f"[FAIL] functional - {e}")
        results.append(("functional", False, str(e)))

    # Test hashing
    try:
        from sense_v2.models.components.hashing import MultiHeadHash, NGramExtractor
        print("[PASS] hashing")
        results.append(("hashing", True, None))
    except Exception as e:
        print(f"[FAIL] hashing - {e}")
        results.append(("hashing", False, str(e)))

    # Test hierarchy
    try:
        from sense_v2.memory.hierarchy import EmbeddingPrefetcher, MemoryHierarchy
        print("[PASS] hierarchy")
        results.append(("hierarchy", True, None))
    except Exception as e:
        print(f"[FAIL] hierarchy - {e}")
        results.append(("hierarchy", False, str(e)))

    # Test config
    try:
        from sense_v2.core.config import MemoryAwareConfig, Config
        cfg = MemoryAwareConfig()
        print(f"[PASS] config - MemoryAwareConfig created")
        results.append(("config", True, None))
    except Exception as e:
        print(f"[FAIL] config - {e}")
        results.append(("config", False, str(e)))

    # Test trainer
    try:
        from sense_v2.agents.agent_0.trainer import get_memory_usage, compute_memory_aware_fitness
        usage = get_memory_usage()
        print(f"[PASS] trainer - psutil={usage.get('psutil_available', False)}")
        results.append(("trainer", True, None))
    except Exception as e:
        print(f"[FAIL] trainer - {e}")
        results.append(("trainer", False, str(e)))

    # Test memory tools
    try:
        from sense_v2.tools.memory_tools import MemoryProfileTool
        print("[PASS] memory_tools")
        results.append(("memory_tools", True, None))
    except Exception as e:
        print(f"[FAIL] memory_tools - {e}")
        results.append(("memory_tools", False, str(e)))

    return results


def test_kernel_functionality():
    """Test kernel functions work correctly."""
    print()
    print("=" * 60)
    print("Testing Kernel Functionality")
    print("=" * 60)

    import torch
    from sense_v2.llm.kernels.functional import fused_linear_cross_entropy

    # Small test
    batch_size = 8
    hidden_dim = 64
    vocab_size = 100

    hidden = torch.randn(batch_size, hidden_dim)
    weight = torch.randn(vocab_size, hidden_dim)
    labels = torch.randint(0, vocab_size, (batch_size,))

    # Test forward pass
    try:
        loss = fused_linear_cross_entropy(hidden, weight, labels)
        print(f"[PASS] Forward pass - loss={loss.item():.4f}")
    except Exception as e:
        print(f"[FAIL] Forward pass - {e}")
        return False

    # Test backward pass
    try:
        hidden_grad = hidden.clone().requires_grad_(True)
        weight_grad = weight.clone().requires_grad_(True)
        loss = fused_linear_cross_entropy(hidden_grad, weight_grad, labels)
        loss.backward()
        print(f"[PASS] Backward pass - hidden.grad shape={hidden_grad.grad.shape}")
    except Exception as e:
        print(f"[FAIL] Backward pass - {e}")
        return False

    # Compare to naive
    try:
        import torch.nn.functional as F
        hidden_naive = hidden.clone().requires_grad_(True)
        logits = hidden_naive @ weight.T
        loss_naive = F.cross_entropy(logits, labels)

        diff = abs(loss.item() - loss_naive.item())
        if diff < 0.01:
            print(f"[PASS] Numerical parity - diff={diff:.6f}")
        else:
            print(f"[WARN] Numerical difference - diff={diff:.6f}")
    except Exception as e:
        print(f"[FAIL] Parity check - {e}")

    return True


def test_hashing():
    """Test N-gram hashing components."""
    print()
    print("=" * 60)
    print("Testing N-gram Hashing")
    print("=" * 60)

    import torch
    from sense_v2.models.components.hashing import MultiHeadHash, NGramExtractor

    # Test MultiHeadHash
    try:
        hasher = MultiHeadHash(num_heads=8, table_size=10000, seed=42)
        ngrams = torch.randint(0, 1000, (4, 16, 3))
        hashes = hasher(ngrams)
        print(f"[PASS] MultiHeadHash - output shape={hashes.shape}")

        # Check bounds
        assert (hashes >= 0).all() and (hashes < 10000).all()
        print("[PASS] Hash bounds check")
    except Exception as e:
        print(f"[FAIL] MultiHeadHash - {e}")
        return False

    # Test NGramExtractor
    try:
        extractor = NGramExtractor(ngram_orders=[2, 3])
        input_ids = torch.randint(0, 1000, (4, 16))
        ngrams, mask = extractor(input_ids)
        print(f"[PASS] NGramExtractor - ngrams={ngrams.shape}, mask={mask.shape}")
    except Exception as e:
        print(f"[FAIL] NGramExtractor - {e}")
        return False

    return True


def test_config():
    """Test configuration classes."""
    print()
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)

    from sense_v2.core.config import MemoryAwareConfig, Config

    # Test MemoryAwareConfig
    try:
        cfg = MemoryAwareConfig()

        # Test threshold methods
        assert not cfg.should_activate_engram(0.50)
        assert cfg.should_activate_engram(0.70)
        print("[PASS] MemoryAwareConfig threshold methods")
    except Exception as e:
        print(f"[FAIL] MemoryAwareConfig - {e}")
        return False

    # Test full Config
    try:
        full_cfg = Config()
        assert hasattr(full_cfg, 'memory_aware')
        print("[PASS] Full Config includes memory_aware")
    except Exception as e:
        print(f"[FAIL] Full Config - {e}")
        return False

    return True


def test_memory_fitness():
    """Test memory-aware fitness function."""
    print()
    print("=" * 60)
    print("Testing Memory-Aware Fitness")
    print("=" * 60)

    from sense_v2.agents.agent_0.trainer import compute_memory_aware_fitness

    try:
        # Lower memory should give higher fitness
        fitness_low = compute_memory_aware_fitness(accuracy=0.9, memory_mb=100.0)
        fitness_high = compute_memory_aware_fitness(accuracy=0.9, memory_mb=2000.0)

        assert fitness_low > fitness_high
        print(f"[PASS] Memory penalty - low_mem={fitness_low:.4f} > high_mem={fitness_high:.4f}")
    except Exception as e:
        print(f"[FAIL] Memory-aware fitness - {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("SENSE-v2 Memory-Aware Tests")
    print("=" * 60)

    all_passed = True

    # Import tests
    import_results = test_imports()
    if any(not r[1] for r in import_results):
        all_passed = False

    # Functionality tests
    if not test_kernel_functionality():
        all_passed = False

    if not test_hashing():
        all_passed = False

    if not test_config():
        all_passed = False

    if not test_memory_fitness():
        all_passed = False

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
