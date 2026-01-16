"""
SENSE-v2 Memory Tools
Schema-based tools for AgeMem memory operations.
"""

from typing import Any, Dict, List, Optional
import logging

from sense_v2.core.base import BaseTool, ToolRegistry
from sense_v2.core.schemas import ToolSchema, ToolParameter, ToolResult


@ToolRegistry.register
class MemoryStoreTool(BaseTool):
    """Tool for storing information in memory."""

    def __init__(self, config: Optional[Any] = None, memory_system: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._memory = memory_system

    def set_memory_system(self, memory_system: Any) -> None:
        """Set the memory system to use."""
        self._memory = memory_system

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_store",
            description="Store information in the agent's memory system",
            parameters=[
                ToolParameter(
                    name="key",
                    param_type="string",
                    description="Unique key for the memory entry",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    param_type="string",
                    description="Content to store",
                    required=True,
                ),
                ToolParameter(
                    name="memory_type",
                    param_type="string",
                    description="Type of memory: stm, ltm, or auto",
                    required=False,
                    default="auto",
                    enum=["stm", "ltm", "auto"],
                ),
                ToolParameter(
                    name="priority",
                    param_type="float",
                    description="Priority for retention (0.0-1.0)",
                    required=False,
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ToolParameter(
                    name="persist",
                    param_type="boolean",
                    description="Force persistence to long-term memory",
                    required=False,
                    default=False,
                ),
            ],
            returns="object",
            returns_description="Store result with key and status",
            category="memory",
            max_retries=1,
        )

    async def execute(
        self,
        key: str,
        content: str,
        memory_type: str = "auto",
        priority: float = 0.5,
        persist: bool = False,
        **kwargs
    ) -> ToolResult:
        """Store content in memory."""
        if self._memory is None:
            return ToolResult.error("Memory system not initialized")

        try:
            from sense_v2.memory.agemem import MemoryType

            mem_type_map = {
                "stm": MemoryType.STM,
                "ltm": MemoryType.LTM,
                "auto": MemoryType.AUTO,
            }

            success = await self._memory.store(
                key=key,
                value=content,
                memory_type=mem_type_map.get(memory_type, MemoryType.AUTO),
                priority=priority,
                persist=persist,
            )

            if success:
                return ToolResult.success({
                    "key": key,
                    "stored": True,
                    "memory_type": memory_type,
                    "priority": priority,
                })
            else:
                return ToolResult.error("Failed to store in memory")

        except Exception as e:
            self.logger.error(f"Memory store failed: {e}")
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemorySearchTool(BaseTool):
    """Tool for searching memory."""

    def __init__(self, config: Optional[Any] = None, memory_system: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._memory = memory_system

    def set_memory_system(self, memory_system: Any) -> None:
        """Set the memory system to use."""
        self._memory = memory_system

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_search",
            description="Search the agent's memory using semantic similarity",
            parameters=[
                ToolParameter(
                    name="query",
                    param_type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    param_type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=20,
                ),
                ToolParameter(
                    name="memory_type",
                    param_type="string",
                    description="Type of memory to search: stm, ltm, or auto",
                    required=False,
                    default="auto",
                    enum=["stm", "ltm", "auto"],
                ),
                ToolParameter(
                    name="threshold",
                    param_type="float",
                    description="Minimum similarity threshold",
                    required=False,
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            returns="array",
            returns_description="List of matching memory entries with similarity scores",
            category="memory",
            max_retries=1,
        )

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str = "auto",
        threshold: float = 0.5,
        **kwargs
    ) -> ToolResult:
        """Search memory."""
        if self._memory is None:
            return ToolResult.error("Memory system not initialized")

        try:
            from sense_v2.memory.agemem import MemoryType

            mem_type_map = {
                "stm": MemoryType.STM,
                "ltm": MemoryType.LTM,
                "auto": MemoryType.AUTO,
            }

            results = await self._memory.search(
                query=query,
                top_k=top_k,
                memory_type=mem_type_map.get(memory_type, MemoryType.AUTO),
                threshold=threshold,
            )

            return ToolResult.success(
                results,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "memory_type": memory_type,
                },
            )

        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemoryRetrieveTool(BaseTool):
    """Tool for retrieving specific memory entry by key."""

    def __init__(self, config: Optional[Any] = None, memory_system: Optional[Any] = None):
        super().__init__(config)
        self._memory = memory_system

    def set_memory_system(self, memory_system: Any) -> None:
        self._memory = memory_system

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_retrieve",
            description="Retrieve a specific memory entry by key",
            parameters=[
                ToolParameter(
                    name="key",
                    param_type="string",
                    description="Key of the memory entry to retrieve",
                    required=True,
                ),
            ],
            returns="object",
            returns_description="Memory entry content or null",
            category="memory",
            max_retries=0,
        )

    async def execute(self, key: str, **kwargs) -> ToolResult:
        """Retrieve memory by key."""
        if self._memory is None:
            return ToolResult.error("Memory system not initialized")

        try:
            content = await self._memory.retrieve(key)

            if content is not None:
                return ToolResult.success({
                    "key": key,
                    "content": content,
                    "found": True,
                })
            else:
                return ToolResult.success({
                    "key": key,
                    "content": None,
                    "found": False,
                })

        except Exception as e:
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemoryStatsTool(BaseTool):
    """Tool for getting memory usage statistics."""

    def __init__(self, config: Optional[Any] = None, memory_system: Optional[Any] = None):
        super().__init__(config)
        self._memory = memory_system

    def set_memory_system(self, memory_system: Any) -> None:
        self._memory = memory_system

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_stats",
            description="Get memory system usage statistics",
            parameters=[],
            returns="object",
            returns_description="Memory usage statistics",
            category="memory",
            max_retries=0,
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Get memory stats."""
        if self._memory is None:
            return ToolResult.error("Memory system not initialized")

        try:
            stats = self._memory.get_usage_stats()
            return ToolResult.success(stats)
        except Exception as e:
            return ToolResult.error(str(e))
