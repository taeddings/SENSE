"""
SENSE-v2 Memory Operations Tools
Direct memory operations for AgeMem integration.

Part of Sprint 1: The Core

Provides schema-based tools for:
- Direct memory storage
- Semantic search/query
- STM to LTM consolidation
- Memory statistics and debugging
"""

from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime

from sense_v2.core.base import BaseTool, ToolRegistry
from sense_v2.core.schemas import (
    ToolSchema,
    ToolParameter,
    ToolResult,
    ToolResultStatus,
)
from sense_v2.memory.agemem import AgeMem, MemoryType


# Global AgeMem instance (initialized lazily)
_agemem_instance: Optional[AgeMem] = None


def get_agemem() -> AgeMem:
    """
    Get the global AgeMem instance.

    Returns:
        AgeMem instance (creates one if not exists)
    """
    global _agemem_instance
    if _agemem_instance is None:
        from sense_v2.memory.agemem import create_agemem
        _agemem_instance = create_agemem()
    return _agemem_instance


def set_agemem(agemem: AgeMem) -> None:
    """
    Set the global AgeMem instance.

    Args:
        agemem: AgeMem instance to use
    """
    global _agemem_instance
    _agemem_instance = agemem


@ToolRegistry.register
class MemoryStoreTool(BaseTool):
    """
    Direct AgeMem store operation.

    Stores content in memory with optional metadata and persistence.
    Supports both STM (short-term) and LTM (long-term) storage.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_store",
            description="Store content in agent memory (STM or LTM)",
            parameters=[
                ToolParameter(
                    name="key",
                    param_type="string",
                    description="Unique identifier for the memory entry",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    param_type="string",
                    description="Content to store in memory",
                    required=True,
                ),
                ToolParameter(
                    name="memory_type",
                    param_type="string",
                    description="Memory type: 'stm' (short-term), 'ltm' (long-term), or 'auto'",
                    required=False,
                    default="auto",
                    enum=["stm", "ltm", "auto"],
                ),
                ToolParameter(
                    name="priority",
                    param_type="float",
                    description="Retention priority (0.0 to 1.0, higher = more important)",
                    required=False,
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ToolParameter(
                    name="persist",
                    param_type="boolean",
                    description="If true, always store in LTM for persistence",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="metadata",
                    param_type="object",
                    description="Optional metadata to attach to the memory entry",
                    required=False,
                ),
            ],
            returns="object",
            returns_description="Result with storage confirmation and metadata",
            category="memory",
            max_retries=2,
        )

    async def execute(
        self,
        key: str,
        content: str,
        memory_type: str = "auto",
        priority: float = 0.5,
        persist: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Store content in AgeMem.

        Args:
            key: Unique identifier
            content: Content to store
            memory_type: Where to store ("stm", "ltm", "auto")
            priority: Retention priority
            persist: Force LTM persistence
            metadata: Optional metadata

        Returns:
            ToolResult with storage confirmation
        """
        try:
            agemem = get_agemem()

            # Map string to MemoryType enum
            mem_type_map = {
                "stm": MemoryType.STM,
                "ltm": MemoryType.LTM,
                "auto": MemoryType.AUTO,
            }
            mem_type = mem_type_map.get(memory_type.lower(), MemoryType.AUTO)

            # Add tool-specific metadata
            full_metadata = metadata or {}
            full_metadata.update({
                "stored_via": "memory_store_tool",
                "stored_at": datetime.now().isoformat(),
            })

            # Store in memory
            success = await agemem.store(
                key=key,
                value=content,
                metadata=full_metadata,
                memory_type=mem_type,
                priority=priority,
                persist=persist,
            )

            if success:
                return ToolResult.success(
                    {
                        "key": key,
                        "stored": True,
                        "memory_type": memory_type,
                        "priority": priority,
                        "persisted": persist or priority >= 0.9,
                        "content_length": len(content),
                    },
                    metadata={
                        "operation": "store",
                    },
                )
            else:
                return ToolResult.error(
                    f"Failed to store memory entry: {key}",
                    metadata={"key": key},
                )

        except Exception as e:
            self.logger.error(f"Memory store error: {e}")
            return ToolResult.error(str(e), metadata={"key": key})


@ToolRegistry.register
class MemoryQueryTool(BaseTool):
    """
    Direct AgeMem semantic search.

    Performs semantic similarity search across memory to find relevant entries.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_query",
            description="Search memory using semantic similarity",
            parameters=[
                ToolParameter(
                    name="query",
                    param_type="string",
                    description="Search query (natural language)",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    param_type="integer",
                    description="Maximum number of results to return",
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=50,
                ),
                ToolParameter(
                    name="memory_type",
                    param_type="string",
                    description="Where to search: 'stm', 'ltm', or 'auto' (both)",
                    required=False,
                    default="auto",
                    enum=["stm", "ltm", "auto"],
                ),
                ToolParameter(
                    name="threshold",
                    param_type="float",
                    description="Minimum similarity threshold (0.0 to 1.0)",
                    required=False,
                    default=0.0,
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
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Search AgeMem using semantic similarity.

        Args:
            query: Natural language search query
            top_k: Maximum results to return
            memory_type: Where to search
            threshold: Minimum similarity threshold

        Returns:
            ToolResult with matching entries
        """
        try:
            agemem = get_agemem()

            # Map string to MemoryType enum
            mem_type_map = {
                "stm": MemoryType.STM,
                "ltm": MemoryType.LTM,
                "auto": MemoryType.AUTO,
            }
            mem_type = mem_type_map.get(memory_type.lower(), MemoryType.AUTO)

            # Perform search
            results = await agemem.search(
                query=query,
                top_k=top_k,
                memory_type=mem_type,
                threshold=threshold if threshold > 0 else None,
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "key": result.get("key", "unknown"),
                    "content": result.get("content", ""),
                    "similarity": result.get("similarity", result.get("priority", 0.0)),
                    "source": result.get("source", memory_type),
                    "metadata": result.get("metadata", {}),
                })

            return ToolResult.success(
                formatted_results,
                metadata={
                    "query": query[:100],
                    "total_results": len(formatted_results),
                    "memory_type": memory_type,
                    "threshold": threshold,
                },
            )

        except Exception as e:
            self.logger.error(f"Memory query error: {e}")
            return ToolResult.error(str(e), metadata={"query": query[:50]})


@ToolRegistry.register
class MemoryRetrieveTool(BaseTool):
    """
    Direct key-based memory retrieval.

    Retrieves a specific memory entry by its key.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

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
                ToolParameter(
                    name="memory_type",
                    param_type="string",
                    description="Where to look: 'stm', 'ltm', or 'auto' (check both)",
                    required=False,
                    default="auto",
                    enum=["stm", "ltm", "auto"],
                ),
            ],
            returns="object",
            returns_description="The memory entry content and metadata",
            category="memory",
            max_retries=1,
        )

    async def execute(
        self,
        key: str,
        memory_type: str = "auto",
        **kwargs: Any,
    ) -> ToolResult:
        """
        Retrieve a memory entry by key.

        Args:
            key: Key to retrieve
            memory_type: Where to look

        Returns:
            ToolResult with entry content or not found
        """
        try:
            agemem = get_agemem()

            # Map string to MemoryType enum
            mem_type_map = {
                "stm": MemoryType.STM,
                "ltm": MemoryType.LTM,
                "auto": MemoryType.AUTO,
            }
            mem_type = mem_type_map.get(memory_type.lower(), MemoryType.AUTO)

            # Retrieve
            content = await agemem.retrieve(key, memory_type=mem_type)

            if content is not None:
                return ToolResult.success(
                    {
                        "key": key,
                        "content": content,
                        "found": True,
                    },
                    metadata={"memory_type": memory_type},
                )
            else:
                return ToolResult.success(
                    {
                        "key": key,
                        "content": None,
                        "found": False,
                    },
                    metadata={"memory_type": memory_type},
                )

        except Exception as e:
            self.logger.error(f"Memory retrieve error: {e}")
            return ToolResult.error(str(e), metadata={"key": key})


@ToolRegistry.register
class MemoryConsolidateTool(BaseTool):
    """
    Force STM -> LTM consolidation.

    Triggers memory consolidation to move important short-term memories
    to long-term storage.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_consolidate",
            description="Force consolidation of important STM entries to LTM",
            parameters=[
                ToolParameter(
                    name="key",
                    param_type="string",
                    description="Specific key to consolidate (optional, all high-priority if not specified)",
                    required=False,
                ),
                ToolParameter(
                    name="min_priority",
                    param_type="float",
                    description="Minimum priority for automatic consolidation",
                    required=False,
                    default=0.8,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            returns="object",
            returns_description="Consolidation results with count of entries moved",
            category="memory",
            max_retries=1,
        )

    async def execute(
        self,
        key: Optional[str] = None,
        min_priority: float = 0.8,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Force memory consolidation.

        Args:
            key: Specific key to consolidate (optional)
            min_priority: Minimum priority for auto-consolidation

        Returns:
            ToolResult with consolidation statistics
        """
        try:
            agemem = get_agemem()

            if key:
                # Consolidate specific key
                success = await agemem.promote_to_ltm(key)
                return ToolResult.success(
                    {
                        "consolidated": 1 if success else 0,
                        "key": key,
                        "success": success,
                    },
                    metadata={"operation": "promote_single"},
                )
            else:
                # Trigger general consolidation
                # This calls the internal consolidation method
                consolidated = await agemem._consolidate()

                return ToolResult.success(
                    {
                        "consolidated": consolidated,
                        "min_priority": min_priority,
                    },
                    metadata={"operation": "consolidate_all"},
                )

        except Exception as e:
            self.logger.error(f"Memory consolidation error: {e}")
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemoryStatsTool(BaseTool):
    """
    Get memory usage statistics.

    Returns detailed statistics about STM and LTM usage.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_stats",
            description="Get memory usage statistics for STM and LTM",
            parameters=[],
            returns="object",
            returns_description="Memory statistics including utilization and entry counts",
            category="memory",
            max_retries=0,
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Get memory usage statistics.

        Returns:
            ToolResult with memory statistics
        """
        try:
            agemem = get_agemem()
            stats = agemem.get_usage_stats()

            # Check if pruning is needed
            should_prune = agemem.should_prune()

            # Get context info
            context_window = agemem.get_context(max_tokens=100)

            return ToolResult.success(
                {
                    "stm": stats.get("stm", {}),
                    "ltm": stats.get("ltm", {}),
                    "combined": stats.get("combined", {}),
                    "should_prune": should_prune,
                    "context_preview": context_window[:200] if context_window else "",
                },
            )

        except Exception as e:
            self.logger.error(f"Memory stats error: {e}")
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemoryDeleteTool(BaseTool):
    """
    Delete a memory entry.

    Removes an entry from STM, LTM, or both.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_delete",
            description="Delete a memory entry by key",
            parameters=[
                ToolParameter(
                    name="key",
                    param_type="string",
                    description="Key of the memory entry to delete",
                    required=True,
                ),
                ToolParameter(
                    name="memory_type",
                    param_type="string",
                    description="Where to delete from: 'stm', 'ltm', or 'auto' (both)",
                    required=False,
                    default="auto",
                    enum=["stm", "ltm", "auto"],
                ),
            ],
            returns="object",
            returns_description="Deletion result",
            category="memory",
            requires_confirmation=True,
            max_retries=1,
        )

    async def execute(
        self,
        key: str,
        memory_type: str = "auto",
        **kwargs: Any,
    ) -> ToolResult:
        """
        Delete a memory entry.

        Args:
            key: Key to delete
            memory_type: Where to delete from

        Returns:
            ToolResult with deletion status
        """
        try:
            agemem = get_agemem()

            # Map string to MemoryType enum
            mem_type_map = {
                "stm": MemoryType.STM,
                "ltm": MemoryType.LTM,
                "auto": MemoryType.AUTO,
            }
            mem_type = mem_type_map.get(memory_type.lower(), MemoryType.AUTO)

            # Delete
            success = await agemem.delete(key, memory_type=mem_type)

            return ToolResult.success(
                {
                    "key": key,
                    "deleted": success,
                    "memory_type": memory_type,
                },
            )

        except Exception as e:
            self.logger.error(f"Memory delete error: {e}")
            return ToolResult.error(str(e), metadata={"key": key})


@ToolRegistry.register
class MemorySummarizeTool(BaseTool):
    """
    Summarize and prune memory content.

    Triggers the summarize-and-prune mechanism for content reduction.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_summarize",
            description="Summarize content to reduce token count",
            parameters=[
                ToolParameter(
                    name="content",
                    param_type="string",
                    description="Content to summarize",
                    required=True,
                ),
                ToolParameter(
                    name="target_ratio",
                    param_type="float",
                    description="Target size as ratio of original (e.g., 0.3 = 30%)",
                    required=False,
                    default=0.3,
                    min_value=0.1,
                    max_value=0.9,
                ),
            ],
            returns="string",
            returns_description="Summarized content",
            category="memory",
            max_retries=1,
        )

    async def execute(
        self,
        content: str,
        target_ratio: float = 0.3,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Summarize content to reduce size.

        Args:
            content: Content to summarize
            target_ratio: Target size ratio

        Returns:
            ToolResult with summarized content
        """
        try:
            agemem = get_agemem()

            # Summarize
            summarized = await agemem.summarize_and_prune(content, target_ratio)

            original_len = len(content)
            summarized_len = len(summarized)
            actual_ratio = summarized_len / original_len if original_len > 0 else 1.0

            return ToolResult.success(
                summarized,
                metadata={
                    "original_length": original_len,
                    "summarized_length": summarized_len,
                    "actual_ratio": actual_ratio,
                    "target_ratio": target_ratio,
                },
            )

        except Exception as e:
            self.logger.error(f"Memory summarize error: {e}")
            return ToolResult.error(str(e))
