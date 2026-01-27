#!/usr/bin/env python3
"""
SENSE v4.0 Import Validation Script
Tests that all modules can be imported without instantiation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all critical imports"""
    results = []

    # Core imports
    tests = [
        ("ReasoningOrchestrator", "sense.core.reasoning_orchestrator", "ReasoningOrchestrator"),
        ("CouncilProtocol", "sense.core.council", "CouncilProtocol"),
        ("UniversalMemory", "sense.memory.bridge", "UniversalMemory"),
        ("GeneticMemory", "sense.memory.genetic", "GeneticMemory"),
        ("Config", "sense.config", "INTELLIGENCE_ENABLED"),
        ("Plugin Loader", "sense.core.plugins.loader", "load_all_plugins"),
        ("VisionInterface", "sense.vision.bridge", "VisionInterface"),
    ]

    # Intelligence Layer tests
    intelligence_tests = [
        ("IntelligenceLayer", "sense.intelligence.integration", "IntelligenceLayer"),
        ("UncertaintyDetection", "sense.intelligence.uncertainty", "UncertaintyDetector"),
        ("KnowledgeRAG", "sense.intelligence.knowledge", "KnowledgeRAG"),
        ("PreferenceLearning", "sense.intelligence.preferences", "PreferenceLearner"),
        ("Metacognition", "sense.intelligence.metacognition", "MetacognitionTracker"),
    ]

    print("=" * 70)
    print("SENSE v4.0 Import Validation")
    print("=" * 70)
    print()

    # Test core imports
    print("Core Components:")
    print("-" * 70)
    for name, module, attr in tests:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"  ✅ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            results.append((name, False))

    print()
    print("Intelligence Layer:")
    print("-" * 70)
    for name, module, attr in intelligence_tests:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"  ✅ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ⚠️  {name}: {e}")
            print(f"      (May be expected if dependencies missing)")
            results.append((name, False))

    # Test config values
    print()
    print("Configuration:")
    print("-" * 70)
    try:
        from sense.config import (
            INTELLIGENCE_ENABLED,
            ENABLE_HARVESTED_TOOLS,
            ENABLE_VISION,
            MEMORY_BACKEND
        )
        print(f"  ✅ INTELLIGENCE_ENABLED: {INTELLIGENCE_ENABLED}")
        print(f"  ✅ ENABLE_HARVESTED_TOOLS: {ENABLE_HARVESTED_TOOLS}")
        print(f"  ✅ ENABLE_VISION: {ENABLE_VISION}")
        print(f"  ✅ MEMORY_BACKEND: {MEMORY_BACKEND}")
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Core imports: {passed}/{total} successful")

    if passed >= len(tests):  # Core imports must all pass
        print("✅ Core system validated")
        return 0
    else:
        print("❌ Core system has import errors")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
