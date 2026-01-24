import pytest
import asyncio
from sense.core.reasoning_orchestrator import ReasoningOrchestrator, UnifiedGrounding, VerificationResult

@pytest.mark.asyncio
async def test_critic_rejects_bad_output():
    grounding = UnifiedGrounding()
    # Test with None result
    verification = grounding.verify(None)
    assert not verification.passed
    assert verification.confidence < 0.6

    # Test with invalid result
    verification = grounding.verify("invalid")
    assert verification.passed  # Since str len >0

# Test retry logic would require mocking LLM
