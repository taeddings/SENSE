"""
Tests for ReasoningOrchestrator singleton pattern and initialization.

Per DIRECTIVE_ORCHESTRATOR.md requirements:
- Singleton pattern verification
- Module-level orchestrator instance
- Persona loading
- Async flow testing
"""

import pytest
from sense.core.reasoning_orchestrator import (
    orchestrator,  # Test singleton instance
    ReasoningOrchestrator,
    create_orchestrator,
    UnifiedGrounding,
    Phase,
)


class TestSingletonPattern:
    """Tests for singleton pattern implementation."""

    def test_singleton_instance_exists(self):
        """Verify orchestrator module-level instance exists."""
        assert orchestrator is not None
        assert isinstance(orchestrator, ReasoningOrchestrator)

    def test_singleton_returns_same_instance(self):
        """Creating new instance should return same object."""
        new_instance = ReasoningOrchestrator()
        assert new_instance is orchestrator

    def test_singleton_multiple_calls(self):
        """Multiple instantiations all return same singleton."""
        instance1 = ReasoningOrchestrator()
        instance2 = ReasoningOrchestrator()
        instance3 = ReasoningOrchestrator()

        assert instance1 is instance2
        assert instance2 is instance3
        assert instance1 is orchestrator

    def test_create_orchestrator_returns_singleton(self):
        """Factory function should return the singleton."""
        created = create_orchestrator()
        assert created is orchestrator

    def test_singleton_initialized_flag(self):
        """Verify _initialized flag is set."""
        assert hasattr(orchestrator, '_initialized')
        assert orchestrator._initialized is True


class TestPersonasLoading:
    """Tests for persona loading from disk."""

    def test_personas_dictionary_exists(self):
        """Verify personas dictionary is created."""
        assert hasattr(orchestrator, '_personas')
        assert isinstance(orchestrator._personas, dict)

    def test_architect_persona_loaded(self):
        """Verify architect persona is loaded."""
        assert "architect" in orchestrator._personas

    def test_worker_persona_loaded(self):
        """Verify worker persona is loaded."""
        assert "worker" in orchestrator._personas

    def test_critic_persona_loaded(self):
        """Verify critic persona is loaded."""
        assert "critic" in orchestrator._personas

    def test_personas_not_empty(self):
        """Verify personas have content."""
        for name, content in orchestrator._personas.items():
            assert len(content) > 10, f"Persona {name} is too short"


class TestGroundingInitialization:
    """Tests for UnifiedGrounding component."""

    def test_grounding_initialization(self):
        """Verify UnifiedGrounding initializes correctly."""
        grounding = UnifiedGrounding()
        assert "synthetic" in grounding.weights
        assert "realworld" in grounding.weights
        assert "experiential" in grounding.weights

    def test_grounding_weights_sum(self):
        """Verify grounding weights are properly distributed."""
        grounding = UnifiedGrounding()
        total = sum(grounding.weights.values())
        assert abs(total - 1.0) < 0.01, "Weights should sum to ~1.0"

    def test_orchestrator_has_grounding(self):
        """Verify orchestrator has grounding component."""
        assert hasattr(orchestrator, 'grounding')
        assert isinstance(orchestrator.grounding, UnifiedGrounding)


class TestPhaseEnum:
    """Tests for Phase enumeration."""

    def test_phase_architect_value(self):
        """Verify Phase.ARCHITECT value."""
        assert Phase.ARCHITECT.value == "architect"

    def test_phase_worker_value(self):
        """Verify Phase.WORKER value."""
        assert Phase.WORKER.value == "worker"

    def test_phase_critic_value(self):
        """Verify Phase.CRITIC value."""
        assert Phase.CRITIC.value == "critic"

    def test_phase_integration_value(self):
        """Verify Phase.INTEGRATION value."""
        assert Phase.INTEGRATION.value == "integration"


class TestAsyncFlow:
    """Tests for async task solving flow."""

    @pytest.mark.asyncio
    async def test_solve_task_returns_result(self):
        """Verify solve_task returns a TaskResult."""
        result = await orchestrator.solve_task("Test Task: Calculate 2 + 2")
        assert result is not None
        assert hasattr(result, 'task_id')
        assert hasattr(result, 'task')
        assert hasattr(result, 'success')

    @pytest.mark.asyncio
    async def test_solve_task_contains_task_text(self):
        """Verify result contains the original task."""
        task_text = "Test arithmetic operation"
        result = await orchestrator.solve_task(task_text)
        assert task_text in result.task

    @pytest.mark.asyncio
    async def test_solve_task_has_phases(self):
        """Verify phases are tracked in result."""
        result = await orchestrator.solve_task("Simple test task")
        assert hasattr(result, 'phases_completed')
        assert len(result.phases_completed) > 0

    @pytest.mark.asyncio
    async def test_solve_task_timing(self):
        """Verify execution time is tracked."""
        result = await orchestrator.solve_task("Timing test")
        assert hasattr(result, 'execution_time')
        assert result.execution_time >= 0


class TestOrchestratorComponents:
    """Tests for orchestrator component initialization."""

    def test_has_tool_forge(self):
        """Verify tool forge component exists."""
        assert hasattr(orchestrator, 'tool_forge')

    def test_has_age_mem(self):
        """Verify AgeMem memory component exists."""
        assert hasattr(orchestrator, 'age_mem')

    def test_has_bridge(self):
        """Verify Bridge component exists."""
        assert hasattr(orchestrator, 'bridge')

    def test_has_execution_history(self):
        """Verify execution history tracking exists."""
        assert hasattr(orchestrator, '_execution_history')
        assert isinstance(orchestrator._execution_history, list)

    def test_has_max_retries(self):
        """Verify MAX_RETRIES constant exists."""
        assert hasattr(ReasoningOrchestrator, 'MAX_RETRIES')
        assert ReasoningOrchestrator.MAX_RETRIES > 0
