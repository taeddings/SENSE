"""
Tests for SENSE-v2 Memory Tools
Per SYSTEM_PROMPT: Every tool must include test_[toolname].py
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from sense.tools.memory_tools import (
    MemoryStoreTool,
    MemorySearchTool,
    MemoryRetrieveTool,
    MemoryStatsTool,
)
from sense.core.schemas import ToolResultStatus


class MockMemorySystem:
    """Mock memory system for testing."""

    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self._search_results: List[Dict] = []

    async def store(
        self,
        key: str,
        value: Any,
        memory_type=None,
        priority: float = 0.5,
        persist: bool = False
    ) -> bool:
        self._storage[key] = {
            "value": value,
            "priority": priority,
            "persist": persist,
        }
        return True

    async def retrieve(self, key: str):
        entry = self._storage.get(key)
        return entry["value"] if entry else None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type=None,
        threshold: float = 0.5
    ) -> List[Dict]:
        # Return mock search results
        return self._search_results[:top_k]

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self._storage),
            "memory_used": sum(len(str(v)) for v in self._storage.values()),
            "utilization": 0.5,
        }

    def set_search_results(self, results: List[Dict]):
        """Set mock search results for testing."""
        self._search_results = results


class TestMemoryStoreSuccess:
    """Test successful memory store operations."""

    @pytest.mark.asyncio
    async def test_memory_store_success(self):
        """Store with mock memory."""
        mock_memory = MockMemorySystem()
        tool = MemoryStoreTool(memory_system=mock_memory)

        result = await tool.execute(
            key="test_key",
            content="test content",
            memory_type="stm",
            priority=0.7
        )

        assert result.is_success is True
        assert result.output["key"] == "test_key"
        assert result.output["stored"] is True
        assert result.output["memory_type"] == "stm"
        assert result.output["priority"] == 0.7

    @pytest.mark.asyncio
    async def test_memory_store_with_persist(self):
        """Store with persistence flag."""
        mock_memory = MockMemorySystem()
        tool = MemoryStoreTool(memory_system=mock_memory)

        result = await tool.execute(
            key="persist_key",
            content="persistent data",
            persist=True
        )

        assert result.is_success is True
        assert mock_memory._storage["persist_key"]["persist"] is True

    @pytest.mark.asyncio
    async def test_memory_store_auto_type(self):
        """Store with auto memory type."""
        mock_memory = MockMemorySystem()
        tool = MemoryStoreTool(memory_system=mock_memory)

        result = await tool.execute(
            key="auto_key",
            content="auto content",
            memory_type="auto"
        )

        assert result.is_success is True
        assert result.output["memory_type"] == "auto"


class TestMemoryStoreNoSystem:
    """Test handling uninitialized memory."""

    @pytest.mark.asyncio
    async def test_memory_store_no_system(self):
        """Handle uninitialized memory."""
        tool = MemoryStoreTool()  # No memory_system provided

        result = await tool.execute(
            key="test_key",
            content="test content"
        )

        assert result.is_success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_memory_store_set_system_later(self):
        """Set memory system after construction."""
        tool = MemoryStoreTool()
        mock_memory = MockMemorySystem()

        # Initially no system
        result = await tool.execute(key="key", content="content")
        assert result.is_success is False

        # Set system
        tool.set_memory_system(mock_memory)

        # Now should work
        result = await tool.execute(key="key", content="content")
        assert result.is_success is True


class TestMemorySearchSuccess:
    """Test successful memory search operations."""

    @pytest.mark.asyncio
    async def test_memory_search_success(self):
        """Search with results."""
        mock_memory = MockMemorySystem()
        mock_memory.set_search_results([
            {"key": "result1", "content": "content1", "similarity": 0.9},
            {"key": "result2", "content": "content2", "similarity": 0.8},
        ])

        tool = MemorySearchTool(memory_system=mock_memory)

        result = await tool.execute(query="test query", top_k=5)

        assert result.is_success is True
        assert len(result.output) == 2
        assert result.output[0]["key"] == "result1"
        assert result.metadata["query"] == "test query"
        assert result.metadata["results_count"] == 2

    @pytest.mark.asyncio
    async def test_memory_search_with_threshold(self):
        """Search with similarity threshold."""
        mock_memory = MockMemorySystem()
        mock_memory.set_search_results([
            {"key": "high", "content": "high score", "similarity": 0.9},
        ])

        tool = MemorySearchTool(memory_system=mock_memory)

        result = await tool.execute(
            query="test",
            threshold=0.7,
            memory_type="ltm"
        )

        assert result.is_success is True
        assert result.metadata["memory_type"] == "ltm"

    @pytest.mark.asyncio
    async def test_memory_search_empty_results(self):
        """Search with no results."""
        mock_memory = MockMemorySystem()
        mock_memory.set_search_results([])

        tool = MemorySearchTool(memory_system=mock_memory)

        result = await tool.execute(query="no matches")

        assert result.is_success is True
        assert len(result.output) == 0
        assert result.metadata["results_count"] == 0


class TestMemoryRetrieveFound:
    """Test retrieving existing entries."""

    @pytest.mark.asyncio
    async def test_memory_retrieve_found(self):
        """Retrieve existing key."""
        mock_memory = MockMemorySystem()
        await mock_memory.store("existing_key", "stored value")

        tool = MemoryRetrieveTool(memory_system=mock_memory)

        result = await tool.execute(key="existing_key")

        assert result.is_success is True
        assert result.output["key"] == "existing_key"
        assert result.output["content"] == "stored value"
        assert result.output["found"] is True


class TestMemoryRetrieveNotFound:
    """Test retrieving nonexistent entries."""

    @pytest.mark.asyncio
    async def test_memory_retrieve_not_found(self):
        """Handle missing key."""
        mock_memory = MockMemorySystem()

        tool = MemoryRetrieveTool(memory_system=mock_memory)

        result = await tool.execute(key="nonexistent_key")

        assert result.is_success is True  # Tool succeeds, just reports not found
        assert result.output["key"] == "nonexistent_key"
        assert result.output["content"] is None
        assert result.output["found"] is False


class TestMemoryStats:
    """Test memory usage statistics."""

    @pytest.mark.asyncio
    async def test_memory_stats(self):
        """Get usage statistics."""
        mock_memory = MockMemorySystem()
        await mock_memory.store("key1", "value1")
        await mock_memory.store("key2", "value2")

        tool = MemoryStatsTool(memory_system=mock_memory)

        result = await tool.execute()

        assert result.is_success is True
        assert "total_entries" in result.output
        assert result.output["total_entries"] == 2
        assert "utilization" in result.output

    @pytest.mark.asyncio
    async def test_memory_stats_no_system(self):
        """Handle uninitialized memory for stats."""
        tool = MemoryStatsTool()

        result = await tool.execute()

        assert result.is_success is False
        assert "not initialized" in result.error.lower()


class TestMemoryToolSchemas:
    """Test tool schema definitions."""

    def test_memory_store_schema(self):
        """Verify MemoryStoreTool schema."""
        tool = MemoryStoreTool()
        schema = tool.schema

        assert schema.name == "memory_store"
        assert schema.category == "memory"

        # Check required parameters
        param_names = [p.name for p in schema.parameters]
        assert "key" in param_names
        assert "content" in param_names

        # Check memory_type enum
        mem_type_param = next(p for p in schema.parameters if p.name == "memory_type")
        assert mem_type_param.enum == ["stm", "ltm", "auto"]

        # Check priority range
        priority_param = next(p for p in schema.parameters if p.name == "priority")
        assert priority_param.min_value == 0.0
        assert priority_param.max_value == 1.0

    def test_memory_search_schema(self):
        """Verify MemorySearchTool schema."""
        tool = MemorySearchTool()
        schema = tool.schema

        assert schema.name == "memory_search"
        assert schema.category == "memory"

        # Check query parameter
        query_param = next(p for p in schema.parameters if p.name == "query")
        assert query_param.required is True

        # Check top_k range
        top_k_param = next(p for p in schema.parameters if p.name == "top_k")
        assert top_k_param.min_value == 1
        assert top_k_param.max_value == 20

    def test_memory_retrieve_schema(self):
        """Verify MemoryRetrieveTool schema."""
        tool = MemoryRetrieveTool()
        schema = tool.schema

        assert schema.name == "memory_retrieve"
        assert schema.category == "memory"
        assert len(schema.parameters) == 1
        assert schema.parameters[0].name == "key"
        assert schema.parameters[0].required is True

    def test_memory_stats_schema(self):
        """Verify MemoryStatsTool schema."""
        tool = MemoryStatsTool()
        schema = tool.schema

        assert schema.name == "memory_stats"
        assert schema.category == "memory"
        assert len(schema.parameters) == 0  # No parameters needed


class TestMemoryToolValidation:
    """Test input validation."""

    def test_memory_store_validates_key(self):
        """MemoryStoreTool validates required parameters."""
        tool = MemoryStoreTool()

        # Valid input
        errors = tool.validate_input(key="test", content="data")
        assert len(errors) == 0

        # Missing key
        errors = tool.validate_input(content="data")
        assert len(errors) > 0

        # Missing content
        errors = tool.validate_input(key="test")
        assert len(errors) > 0

    def test_memory_search_validates_query(self):
        """MemorySearchTool validates query parameter."""
        tool = MemorySearchTool()

        errors = tool.validate_input(query="search term")
        assert len(errors) == 0

        errors = tool.validate_input()  # Missing query
        assert len(errors) > 0

    def test_memory_retrieve_validates_key(self):
        """MemoryRetrieveTool validates key parameter."""
        tool = MemoryRetrieveTool()

        errors = tool.validate_input(key="some_key")
        assert len(errors) == 0

        errors = tool.validate_input()  # Missing key
        assert len(errors) > 0


class TestMemoryToolIntegration:
    """Integration tests for memory tools working together."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        """Store then retrieve same key."""
        mock_memory = MockMemorySystem()

        store_tool = MemoryStoreTool(memory_system=mock_memory)
        retrieve_tool = MemoryRetrieveTool(memory_system=mock_memory)

        # Store
        store_result = await store_tool.execute(
            key="integration_key",
            content="integration value"
        )
        assert store_result.is_success is True

        # Retrieve
        retrieve_result = await retrieve_tool.execute(key="integration_key")
        assert retrieve_result.is_success is True
        assert retrieve_result.output["content"] == "integration value"

    @pytest.mark.asyncio
    async def test_set_memory_system_on_all_tools(self):
        """Set memory system on all tools."""
        mock_memory = MockMemorySystem()

        tools = [
            MemoryStoreTool(),
            MemorySearchTool(),
            MemoryRetrieveTool(),
            MemoryStatsTool(),
        ]

        for tool in tools:
            tool.set_memory_system(mock_memory)

        # Verify all tools can now use the memory system
        store_result = await tools[0].execute(key="test", content="data")
        assert store_result.is_success is True

        stats_result = await tools[3].execute()
        assert stats_result.is_success is True
