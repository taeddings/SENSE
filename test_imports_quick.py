#!/usr/bin/env python3
"""Quick import verification script."""
import sys
sys.path.insert(0, 'src')

print('Testing imports...')

try:
    from sense.core.reasoning_orchestrator import orchestrator, ReasoningOrchestrator, Phase, UnifiedGrounding
    print('1. ReasoningOrchestrator: OK')
except Exception as e:
    print(f'1. FAILED: {e}')
    sys.exit(1)

try:
    assert orchestrator is not None
    assert orchestrator is ReasoningOrchestrator()
    print('2. Singleton pattern: OK')
except Exception as e:
    print(f'2. FAILED: {e}')
    sys.exit(1)

try:
    assert 'architect' in orchestrator._personas
    assert 'worker' in orchestrator._personas
    assert 'critic' in orchestrator._personas
    print('3. Personas loaded: OK')
except Exception as e:
    print(f'3. FAILED: {e}')
    sys.exit(1)

try:
    grounding = UnifiedGrounding()
    assert 'synthetic' in grounding.weights
    print('4. UnifiedGrounding: OK')
except Exception as e:
    print(f'4. FAILED: {e}')
    sys.exit(1)

try:
    assert Phase.ARCHITECT.value == 'architect'
    print('5. Phase enum: OK')
except Exception as e:
    print(f'5. FAILED: {e}')
    sys.exit(1)

try:
    from sense_v2 import Config, AgeMem
    print('6. sense_v2 imports: OK')
except Exception as e:
    print(f'6. FAILED: {e}')
    sys.exit(1)

print()
print('ALL TESTS PASSED!')
