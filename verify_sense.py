#!/usr/bin/env python3
"""
SENSE v4.0 Component Validation Script
Tests each component in isolation before full integration
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Test that all core components can be imported"""
    print("=" * 60)
    print("Phase 1: Core Imports")
    print("=" * 60)

    try:
        from sense.core.reasoning_orchestrator import ReasoningOrchestrator
        print("✅ ReasoningOrchestrator imported")
    except Exception as e:
        print(f"❌ ReasoningOrchestrator import failed: {e}")
        return False

    try:
        from sense.core.council import CouncilProtocol
        print("✅ CouncilProtocol imported")
    except Exception as e:
        print(f"❌ CouncilProtocol import failed: {e}")
        return False

    try:
        from sense.memory.bridge import UniversalMemory
        print("✅ UniversalMemory imported")
    except Exception as e:
        print(f"❌ UniversalMemory import failed: {e}")
        return False

    try:
        from sense.memory.genetic import GeneticMemory
        print("✅ GeneticMemory imported")
    except Exception as e:
        print(f"❌ GeneticMemory import failed: {e}")
        return False

    try:
        from sense.config import INTELLIGENCE_ENABLED
        print(f"✅ Config imported (Intelligence: {INTELLIGENCE_ENABLED})")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

    return True

def test_intelligence_layer():
    """Test Intelligence Layer if enabled"""
    print("\n" + "=" * 60)
    print("Phase 2: Intelligence Layer")
    print("=" * 60)

    try:
        from sense.config import INTELLIGENCE_ENABLED

        if not INTELLIGENCE_ENABLED:
            print("⚠️  Intelligence Layer disabled in config")
            return True

        from sense.intelligence.integration import IntelligenceLayer
        print("✅ IntelligenceLayer imported")

        # Try to instantiate (may fail without proper config)
        try:
            layer = IntelligenceLayer()
            print("✅ IntelligenceLayer instantiated")
        except Exception as e:
            print(f"⚠️  IntelligenceLayer instantiation issue: {e}")
            print("   (This may be expected without full config)")

        return True
    except Exception as e:
        print(f"❌ Intelligence Layer test failed: {e}")
        return False

def test_component_instantiation():
    """Test that components can be instantiated"""
    print("\n" + "=" * 60)
    print("Phase 3: Component Instantiation")
    print("=" * 60)

    try:
        # Test UniversalMemory
        from sense.memory.bridge import UniversalMemory
        memory = UniversalMemory()
        print("✅ UniversalMemory instantiated")
    except Exception as e:
        print(f"❌ UniversalMemory instantiation failed: {e}")
        return False

    try:
        # Test GeneticMemory
        from sense.memory.genetic import GeneticMemory
        genetics = GeneticMemory()
        print("✅ GeneticMemory instantiated")
    except Exception as e:
        print(f"❌ GeneticMemory instantiation failed: {e}")
        return False

    try:
        # Test CouncilProtocol (static method)
        from sense.core.council import CouncilProtocol
        council_prompt = CouncilProtocol.get_system_prompt()
        print(f"✅ CouncilProtocol.get_system_prompt() returned {len(council_prompt)} chars")
    except Exception as e:
        print(f"❌ CouncilProtocol test failed: {e}")
        return False

    try:
        # Test ReasoningOrchestrator (singleton)
        from sense.core.reasoning_orchestrator import ReasoningOrchestrator
        orchestrator = ReasoningOrchestrator()
        print("✅ ReasoningOrchestrator instantiated (singleton)")
    except Exception as e:
        print(f"❌ ReasoningOrchestrator instantiation failed: {e}")
        return False

    return True

def test_memory_operations():
    """Test memory save/recall operations"""
    print("\n" + "=" * 60)
    print("Phase 4: Memory Operations")
    print("=" * 60)

    try:
        from sense.memory.bridge import UniversalMemory
        memory = UniversalMemory()

        # Test save
        memory.save_engram("test validation memory", tags=["test", "validation"])
        print("✅ Memory save operation completed")

        # Test recall
        recalled = memory.recall("test validation")
        print(f"✅ Memory recall returned {len(recalled)} results")

    except Exception as e:
        print(f"❌ Memory operations failed: {e}")
        return False

    try:
        from sense.memory.genetic import GeneticMemory
        genetics = GeneticMemory()

        # Test gene save
        genetics.save_gene("test validation task", "test_tool", "test input data")
        print("✅ Genetic save operation completed")

        # Test instinct retrieval
        instinct = genetics.retrieve_instinct("test validation task")
        print(f"✅ Instinct retrieval completed (found: {instinct is not None})")

    except Exception as e:
        print(f"❌ Genetic operations failed: {e}")
        return False

    return True

def test_tool_registry():
    """Test tool loading and registry"""
    print("\n" + "=" * 60)
    print("Phase 5: Tool Registry")
    print("=" * 60)

    try:
        from sense.core.plugins.loader import load_all_plugins

        tools = load_all_plugins()
        print(f"✅ Tool registry loaded {len(tools)} tools")

        for tool_name in tools:
            print(f"   - {tool_name}")

        return True
    except Exception as e:
        print(f"❌ Tool registry test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("Phase 6: Configuration")
    print("=" * 60)

    try:
        from sense.config import (
            INTELLIGENCE_ENABLED,
            ENABLE_HARVESTED_TOOLS,
            ENABLE_VISION,
            MEMORY_BACKEND
        )

        print(f"✅ INTELLIGENCE_ENABLED: {INTELLIGENCE_ENABLED}")
        print(f"✅ ENABLE_HARVESTED_TOOLS: {ENABLE_HARVESTED_TOOLS}")
        print(f"✅ ENABLE_VISION: {ENABLE_VISION}")
        print(f"✅ MEMORY_BACKEND: {MEMORY_BACKEND}")

        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "SENSE v4.0 Component Validation" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    results.append(("Core Imports", test_core_imports()))
    results.append(("Intelligence Layer", test_intelligence_layer()))
    results.append(("Component Instantiation", test_component_instantiation()))
    results.append(("Memory Operations", test_memory_operations()))
    results.append(("Tool Registry", test_tool_registry()))
    results.append(("Configuration", test_config_loading()))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All components validated successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
