import pytest
import asyncio
from sense.core.reasoning_orchestrator import ReasoningOrchestrator, TaskResult

@pytest.mark.asyncio
async def test_orchestrator_basic_flow():
    orch = ReasoningOrchestrator()
    result = await orch.solve_task("Test task: calculate 2+2")
    assert isinstance(result, TaskResult)
    assert result.task == "Test task: calculate 2+2"
    assert len(result.phases_completed) >= 3  # At least architect, worker, critic

def test_orchestrator_stats():
    orch = ReasoningOrchestrator()
    stats = orch.get_execution_stats()
    assert stats["total_tasks"] == 0

# Full flow with ToolForge would require mock memory and patterns
