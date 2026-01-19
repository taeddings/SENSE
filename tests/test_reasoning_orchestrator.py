import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from sense_v2.core.reasoning_orchestrator import (
    ReasoningOrchestrator, ReasoningContext, ReasoningResult
)


@pytest.fixture
def reasoning_orchestrator():
    """Create a ReasoningOrchestrator instance."""
    return ReasoningOrchestrator(enable_sandbox=False)  # Disable sandbox for testing


@pytest.fixture
def sample_context():
    """Create a sample ReasoningContext."""
    return ReasoningContext(
        task_id="test_task_123",
        problem="Solve: 2 + 2",
        available_tools=["calculator", "search"],
        max_steps=5,
        timeout_seconds=10,
    )


class TestReasoningOrchestrator:
    """Test suite for ReasoningOrchestrator."""

    @pytest.mark.asyncio
    async def test_initialization(self, reasoning_orchestrator):
        """Test orchestrator initialization."""
        assert reasoning_orchestrator.enable_sandbox is False  # Sandbox disabled
        assert reasoning_orchestrator.reasoning_strategy is not None
        assert len(reasoning_orchestrator.execution_history) == 0
        assert len(reasoning_orchestrator._active_executions) == 0

    @pytest.mark.asyncio
    async def test_execute_reasoning_unsandboxed(self, reasoning_orchestrator, sample_context):
        """Test reasoning execution without sandbox."""
        result = await reasoning_orchestrator.execute_reasoning(sample_context)

        assert isinstance(result, ReasoningResult)
        assert result.task_id == sample_context.task_id
        assert result.success is True
        assert len(result.steps) > 0
        assert result.execution_time >= 0
        assert "Completed reasoning" in result.final_answer

    @pytest.mark.asyncio
    async def test_execute_reasoning_timeout(self, reasoning_orchestrator):
        """Test reasoning execution with timeout."""
        context = ReasoningContext(
            task_id="timeout_test",
            problem="Very complex problem that takes forever",
            max_steps=100,
            timeout_seconds=0.001,  # Very short timeout
        )

        result = await reasoning_orchestrator.execute_reasoning(context)

        assert result.success is False
        assert "timeout" in result.error_message.lower()
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_concurrent_executions_prevented(self, reasoning_orchestrator, sample_context):
        """Test that concurrent executions with same task_id are prevented."""
        # Start first execution
        task1 = asyncio.create_task(reasoning_orchestrator.execute_reasoning(sample_context))

        # Try to start second execution with same task_id
        with pytest.raises(ValueError, match="already executing"):
            await reasoning_orchestrator.execute_reasoning(sample_context)

        # Wait for first to complete
        await task1

    def test_update_reasoning_strategy(self, reasoning_orchestrator):
        """Test updating reasoning strategy."""
        new_strategy = '''
def reason(problem, tools, max_steps, memory):
    return "Updated strategy result", []
'''

        success = reasoning_orchestrator.update_reasoning_strategy(new_strategy)
        assert success is True
        assert reasoning_orchestrator.reasoning_strategy == new_strategy

    def test_get_execution_stats_empty(self, reasoning_orchestrator):
        """Test getting execution stats when no executions."""
        stats = reasoning_orchestrator.get_execution_stats()
        assert stats["total_executions"] == 0

    @pytest.mark.asyncio
    async def test_get_execution_stats_with_data(self, reasoning_orchestrator, sample_context):
        """Test getting execution stats after executions."""
        # Execute a few reasoning tasks
        await reasoning_orchestrator.execute_reasoning(sample_context)
        await reasoning_orchestrator.execute_reasoning(sample_context)

        stats = reasoning_orchestrator.get_execution_stats()
        assert stats["total_executions"] == 2
        assert "success_rate" in stats
        assert "average_execution_time" in stats
        assert "sandbox_enabled" in stats

    def test_sandbox_availability(self):
        """Test sandbox availability detection."""
        # Test with sandbox disabled
        orch = ReasoningOrchestrator(enable_sandbox=False)
        assert orch.enable_sandbox is False

        # Test with sandbox enabled (but may not be available)
        orch = ReasoningOrchestrator(enable_sandbox=True)
        # Should be True if RestrictedPython is available, False otherwise
        assert isinstance(orch.enable_sandbox, bool)