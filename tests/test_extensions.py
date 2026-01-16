"""
Tests for SENSE-v2 extension system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

from sense_v2.core.extensions import (
    Extension,
    ExtensionPoint,
    ExtensionContext,
    ExtensionRegistry,
    ExtensionManager,
    ExtensionMixin,
    call_extensions,
    register_extension,
)


class TestExtensionPoint:
    """Tests for ExtensionPoint enum."""

    def test_all_points_defined(self):
        """All expected extension points exist."""
        expected = [
            "agent_init",
            "agent_shutdown",
            "monologue_start",
            "monologue_end",
            "message_loop_start",
            "message_loop_end",
            "before_tool_call",
            "after_tool_call",
            "before_llm_call",
            "after_llm_call",
        ]
        actual = [p.value for p in ExtensionPoint]
        for exp in expected:
            assert exp in actual, f"Missing extension point: {exp}"


class TestExtensionContext:
    """Tests for ExtensionContext."""

    def test_default_values(self):
        """Context has sensible defaults."""
        ctx = ExtensionContext()
        assert ctx.agent is None
        assert ctx.message is None
        assert ctx.tool_name is None
        assert ctx.metadata == {}

    def test_with_message(self):
        """with_message creates new context with message."""
        ctx = ExtensionContext(metadata={"key": "value"})
        mock_message = Mock()

        new_ctx = ctx.with_message(mock_message)

        assert new_ctx.message is mock_message
        assert new_ctx.metadata == {"key": "value"}
        assert ctx.message is None  # Original unchanged

    def test_with_tool(self):
        """with_tool creates new context with tool info."""
        ctx = ExtensionContext()
        mock_result = Mock()

        new_ctx = ctx.with_tool(
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
            result=mock_result,
        )

        assert new_ctx.tool_name == "test_tool"
        assert new_ctx.tool_args == {"arg1": "value1"}
        assert new_ctx.tool_result is mock_result


class TestExtension:
    """Tests for Extension base class."""

    def test_subclass_requires_execute(self):
        """Extension subclass must implement execute."""
        class IncompleteExtension(Extension):
            pass

        with pytest.raises(TypeError):
            IncompleteExtension()

    def test_supports_point_default_true(self):
        """By default, extensions support all points."""
        class TestExtension(Extension):
            async def execute(self, context, **kwargs):
                return None

        ext = TestExtension()
        for point in ExtensionPoint:
            assert ext.supports_point(point)

    def test_custom_priority(self):
        """Custom priority can be set."""
        class HighPriorityExtension(Extension):
            priority = 1

            async def execute(self, context, **kwargs):
                return None

        ext = HighPriorityExtension()
        assert ext.priority == 1


class TestExtensionRegistry:
    """Tests for ExtensionRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ExtensionRegistry.clear()

    def test_register_decorator(self):
        """Register decorator adds extension."""
        @ExtensionRegistry.register("test_point")
        class TestExt(Extension):
            async def execute(self, context, **kwargs):
                return "test"

        extensions = ExtensionRegistry.get_extensions("test_point")
        assert TestExt in extensions

    def test_get_extensions_empty(self):
        """Getting extensions for unknown point returns empty."""
        extensions = ExtensionRegistry.get_extensions("unknown_point")
        assert extensions == []

    def test_clear_removes_all(self):
        """Clear removes all registered extensions."""
        @ExtensionRegistry.register("point1")
        class Ext1(Extension):
            async def execute(self, context, **kwargs):
                pass

        ExtensionRegistry.clear()
        assert ExtensionRegistry.get_extensions("point1") == []


class TestExtensionManager:
    """Tests for ExtensionManager."""

    def setup_method(self):
        """Clear registry before each test."""
        ExtensionRegistry.clear()

    @pytest.mark.asyncio
    async def test_call_extensions_empty(self):
        """Calling extensions with none registered returns empty."""
        manager = ExtensionManager()
        results = await manager.call_extensions(ExtensionPoint.AGENT_INIT)
        assert results == []

    @pytest.mark.asyncio
    async def test_call_extensions_executes(self):
        """Registered extensions are executed."""
        results_list = []

        @ExtensionRegistry.register("agent_init")
        class TestExt(Extension):
            async def execute(self, context, **kwargs):
                results_list.append("executed")
                return "done"

        manager = ExtensionManager()
        results = await manager.call_extensions(ExtensionPoint.AGENT_INIT)

        assert len(results_list) == 1
        assert results_list[0] == "executed"
        assert results == ["done"]

    @pytest.mark.asyncio
    async def test_extensions_sorted_by_priority(self):
        """Extensions are called in priority order."""
        call_order = []

        @ExtensionRegistry.register("agent_init")
        class HighPriority(Extension):
            priority = 10

            async def execute(self, context, **kwargs):
                call_order.append("high")

        @ExtensionRegistry.register("agent_init")
        class LowPriority(Extension):
            priority = 90

            async def execute(self, context, **kwargs):
                call_order.append("low")

        manager = ExtensionManager()
        await manager.call_extensions(ExtensionPoint.AGENT_INIT)

        assert call_order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_extension_failure_doesnt_stop_others(self):
        """One extension failing doesn't stop others."""
        results = []

        @ExtensionRegistry.register("agent_init")
        class FailingExt(Extension):
            priority = 10

            async def execute(self, context, **kwargs):
                raise ValueError("Intentional failure")

        @ExtensionRegistry.register("agent_init")
        class SuccessExt(Extension):
            priority = 20

            async def execute(self, context, **kwargs):
                results.append("success")
                return "ok"

        manager = ExtensionManager()
        await manager.call_extensions(ExtensionPoint.AGENT_INIT)

        assert "success" in results

    @pytest.mark.asyncio
    async def test_context_passed_to_extensions(self):
        """Context is properly passed to extensions."""
        received_context = []

        @ExtensionRegistry.register("agent_init")
        class ContextExt(Extension):
            async def execute(self, context, **kwargs):
                received_context.append(context)

        mock_agent = Mock()
        manager = ExtensionManager(agent=mock_agent)
        ctx = ExtensionContext(agent=mock_agent)

        await manager.call_extensions(ExtensionPoint.AGENT_INIT, ctx)

        assert len(received_context) == 1
        assert received_context[0].agent is mock_agent

    def test_clear_cache(self):
        """Cache is cleared properly."""
        manager = ExtensionManager()
        manager._loaded_extensions["test"] = [Mock()]

        manager.clear_cache()

        assert manager._loaded_extensions == {}


class TestExtensionMixin:
    """Tests for ExtensionMixin."""

    @pytest.mark.asyncio
    async def test_init_extensions(self):
        """init_extensions creates manager."""
        class TestAgent(ExtensionMixin):
            name = "test"

        agent = TestAgent()
        agent.init_extensions()

        assert agent._extension_manager is not None

    @pytest.mark.asyncio
    async def test_call_extension_auto_inits(self):
        """call_extension auto-initializes if needed."""
        ExtensionRegistry.clear()

        class TestAgent(ExtensionMixin):
            name = "test"

        agent = TestAgent()
        results = await agent.call_extension(ExtensionPoint.AGENT_INIT)

        assert agent._extension_manager is not None

    @pytest.mark.asyncio
    async def test_lifecycle_hooks(self):
        """Lifecycle hooks call correct extension points."""
        ExtensionRegistry.clear()
        called_points = []

        class RecordingExt(Extension):
            def supports_point(self, point):
                return True

            async def execute(self, context, **kwargs):
                called_points.append("called")

        # Register for all points
        for point in ExtensionPoint:
            ExtensionRegistry._extensions[point.value] = [RecordingExt]

        class TestAgent(ExtensionMixin):
            name = "test"

        agent = TestAgent()

        await agent.on_agent_init()
        await agent.on_monologue_start()
        await agent.on_monologue_end()

        assert len(called_points) == 3


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Clear registry before each test."""
        ExtensionRegistry.clear()

    @pytest.mark.asyncio
    async def test_call_extensions_function(self):
        """call_extensions function works correctly."""
        @ExtensionRegistry.register("agent_init")
        class TestExt(Extension):
            async def execute(self, context, **kwargs):
                return "result"

        await call_extensions("agent_init")

    @pytest.mark.asyncio
    async def test_call_extensions_unknown_point(self):
        """Unknown extension point logs warning."""
        # Should not raise, just log warning
        await call_extensions("unknown_point_xyz")

    def test_register_extension_function(self):
        """register_extension function works."""
        class NewExt(Extension):
            async def execute(self, context, **kwargs):
                pass

        register_extension("custom_point", NewExt)

        extensions = ExtensionRegistry.get_extensions("custom_point")
        assert NewExt in extensions


class TestBuiltInExtensions:
    """Tests for built-in extensions."""

    def setup_method(self):
        """Import built-ins to ensure registration."""
        # Built-ins are auto-registered on import
        from sense_v2.core.extensions import (
            LoggingInitExtension,
            ToolCallLoggingExtension,
            ToolResultLoggingExtension,
            MonologueTimingExtension,
            MonologueEndTimingExtension,
        )

    @pytest.mark.asyncio
    async def test_logging_init_extension(self):
        """LoggingInitExtension logs agent initialization."""
        from sense_v2.core.extensions import LoggingInitExtension

        ext = LoggingInitExtension()
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        ctx = ExtensionContext(agent=mock_agent)

        # Should not raise
        await ext.execute(ctx)

    @pytest.mark.asyncio
    async def test_monologue_timing(self):
        """MonologueTimingExtension tracks timing."""
        from sense_v2.core.extensions import (
            MonologueTimingExtension,
            MonologueEndTimingExtension,
        )

        ctx = ExtensionContext()

        # Start timing
        start_ext = MonologueTimingExtension()
        result = await start_ext.execute(ctx)
        assert result == {"started": True}
        assert "monologue_start_time" in ctx.metadata

        # End timing
        await asyncio.sleep(0.1)  # Small delay
        end_ext = MonologueEndTimingExtension()
        result = await end_ext.execute(ctx)
        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0.1

    @pytest.mark.asyncio
    async def test_tool_logging_extensions(self):
        """Tool logging extensions work correctly."""
        from sense_v2.core.extensions import (
            ToolCallLoggingExtension,
            ToolResultLoggingExtension,
        )

        # Before call
        before_ext = ToolCallLoggingExtension()
        ctx = ExtensionContext(
            tool_name="test_tool",
            tool_args={"key": "value"},
        )
        await before_ext.execute(ctx)  # Should not raise

        # After call
        after_ext = ToolResultLoggingExtension()
        mock_result = Mock()
        mock_result.status = "success"
        ctx = ExtensionContext(
            tool_name="test_tool",
            tool_result=mock_result,
        )
        await after_ext.execute(ctx)  # Should not raise
