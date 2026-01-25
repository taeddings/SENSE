"""
SENSE-v2 AgeMem - The Filing Cabinet
Unified agentic memory system integrating LTM and STM.
Per SYSTEM_PROMPT: Structured knowledge persistence with vector database and context management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from enum import Enum
import logging
import asyncio

from sense.core.base import BaseMemory
from sense.core.config import MemoryConfig, MemoryTier
from sense.memory.ltm import LongTermMemory, LTMEntry
from sense.memory.stm import ShortTermMemory
from sense.memory.embeddings import EmbeddingProvider, SentenceTransformerProvider


class MemoryType(Enum):
    """Type of memory operation."""
    STM = "stm"  # Short-term / working memory
    LTM = "ltm"  # Long-term / persistent memory
    AUTO = "auto"  # Automatically determine


@dataclass
class MemoryEntry:
    """
    Unified memory entry that can exist in STM, LTM, or both.
    """
    key: str
    content: str
    memory_type: MemoryType
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    tier: MemoryTier = MemoryTier.WARM

    # Tracking
    stm_stored: bool = False
    ltm_stored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "tier": self.tier.value,
            "stm_stored": self.stm_stored,
            "ltm_stored": self.ltm_stored,
        }


class AgeMem(BaseMemory):
    """
    AgeMem - The Filing Cabinet
    Unified agentic memory system for SENSE-v2.

    Architecture:
    - Short-Term Memory (STM): Fast, token-limited working memory
    - Long-Term Memory (LTM): Persistent vector database storage

    Features:
    - Automatic tiering (hot -> warm -> cold)
    - Context-aware summarize-and-prune at 80% capacity
    - Semantic search across all memory
    - Memory consolidation (STM -> LTM promotion)

    Per SYSTEM_PROMPT requirements:
    - Vector database for LTM
    - Summarize-and-Prune hook at 80% of model's context limit
    - Structured knowledge persistence
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(config)
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize embedding provider
        self.embedding_provider = embedding_provider or SentenceTransformerProvider()

        # Initialize memory subsystems
        self.stm = ShortTermMemory(config=self.config, summarizer=summarizer)
        self.ltm = LongTermMemory(config=self.config, embedding_provider=self.embedding_provider)

        # Set up STM prune callback for consolidation
        self.stm.set_prune_callback(self._on_stm_prune)

        # Consolidation tracking
        self._consolidation_queue: List[str] = []

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict] = None,
        memory_type: MemoryType = MemoryType.AUTO,
        priority: float = 1.0,
        persist: bool = False,
    ) -> bool:
        """
        Store a value in memory.

        Args:
            key: Unique identifier
            value: Content to store
            metadata: Optional metadata
            memory_type: STM, LTM, or AUTO (default)
            priority: Retention priority for STM
            persist: If True, always store in LTM

        Returns:
            True if successful
        """
        metadata = metadata or {}
        success = True

        # Determine memory type
        if memory_type == MemoryType.AUTO:
            # Auto: STM for recent/temporary, LTM for important/persistent
            if persist or priority >= 0.9 or metadata.get("important", False):
                memory_type = MemoryType.LTM
            else:
                memory_type = MemoryType.STM

        # Store in appropriate memory system
        if memory_type == MemoryType.STM or memory_type == MemoryType.AUTO:
            success = await self.stm.store(key, value, metadata, priority)

        if memory_type == MemoryType.LTM or persist:
            success = await self.ltm.store(key, value, metadata) and success

        return success

    async def retrieve(
        self,
        key: str,
        memory_type: MemoryType = MemoryType.AUTO,
    ) -> Optional[Any]:
        """
        Retrieve a value from memory.

        Args:
            key: Key to retrieve
            memory_type: Where to look (AUTO checks STM first, then LTM)

        Returns:
            Stored content or None
        """
        if memory_type in (MemoryType.STM, MemoryType.AUTO):
            result = await self.stm.retrieve(key)
            if result is not None:
                return result

        if memory_type in (MemoryType.LTM, MemoryType.AUTO):
            return await self.ltm.retrieve(key)

        return None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: MemoryType = MemoryType.AUTO,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memory using semantic similarity.

        Args:
            query: Search query
            top_k: Maximum results
            memory_type: Where to search
            threshold: Minimum similarity

        Returns:
            List of matching entries with scores
        """
        results = []

        # Search STM (keyword-based, fast)
        if memory_type in (MemoryType.STM, MemoryType.AUTO):
            stm_results = await self.stm.search(query, top_k)
            for r in stm_results:
                r["source"] = "stm"
            results.extend(stm_results)

        # Search LTM (semantic, comprehensive)
        if memory_type in (MemoryType.LTM, MemoryType.AUTO):
            ltm_results = await self.ltm.search(query, top_k, threshold)
            for r in ltm_results:
                r["source"] = "ltm"
            results.extend(ltm_results)

        # Deduplicate by key, preferring STM (more recent)
        seen_keys = set()
        unique_results = []
        for r in results:
            if r["key"] not in seen_keys:
                seen_keys.add(r["key"])
                unique_results.append(r)

        # Sort by similarity/priority
        unique_results.sort(
            key=lambda x: x.get("similarity", x.get("priority", 0)),
            reverse=True
        )

        return unique_results[:top_k]

    async def delete(
        self,
        key: str,
        memory_type: MemoryType = MemoryType.AUTO,
    ) -> bool:
        """Delete an entry from memory."""
        success = False

        if memory_type in (MemoryType.STM, MemoryType.AUTO):
            if await self.stm.delete(key):
                success = True

        if memory_type in (MemoryType.LTM, MemoryType.AUTO):
            if await self.ltm.delete(key):
                success = True

        return success

    async def summarize_and_prune(
        self,
        content: str,
        target_ratio: float = 0.3,
    ) -> str:
        """
        Summarize content to reduce token count.
        Delegates to STM's summarizer.
        """
        return await self.stm.summarize_and_prune(content, target_ratio)

    def _on_stm_prune(self, tokens_freed: int, tokens_remaining: int) -> None:
        """
        Callback when STM is pruned.
        Triggers consolidation of important items to LTM.
        """
        self.logger.debug(
            f"STM pruned: {tokens_freed} tokens freed, {tokens_remaining} remaining"
        )
        # Queue consolidation task
        asyncio.create_task(self._consolidate())

    async def _consolidate(self) -> int:
        """
        Consolidate important STM entries to LTM.
        Called after STM prune to preserve valuable information.

        Returns:
            Number of entries consolidated
        """
        consolidated = 0

        # Get STM stats to identify high-priority entries
        stm_stats = self.stm.get_usage_stats()

        # Access STM entries directly for consolidation
        for key, entry in list(self.stm._entries.items()):
            # Consolidate high-priority entries
            if entry.priority >= 0.8:
                # Check if already in LTM
                existing = await self.ltm.retrieve(key)
                if existing is None:
                    await self.ltm.store(
                        key=key,
                        value=entry.content,
                        metadata={
                            **entry.metadata,
                            "consolidated_from": "stm",
                            "original_priority": entry.priority,
                        }
                    )
                    consolidated += 1

        if consolidated > 0:
            self.logger.info(f"Consolidated {consolidated} entries from STM to LTM")

        return consolidated

    async def promote_to_ltm(self, key: str) -> bool:
        """
        Explicitly promote an STM entry to LTM.

        Args:
            key: Key of entry to promote

        Returns:
            True if successful
        """
        content = await self.stm.retrieve(key)
        if content is None:
            return False

        entry = self.stm._entries.get(key)
        metadata = entry.metadata if entry else {}

        return await self.ltm.store(
            key=key,
            value=content,
            metadata={
                **metadata,
                "promoted_from": "stm",
                "promoted_at": datetime.now().isoformat(),
            }
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get combined memory usage statistics."""
        stm_stats = self.stm.get_usage_stats()
        ltm_stats = self.ltm.get_usage_stats()

        return {
            "stm": stm_stats,
            "ltm": ltm_stats,
            "combined": {
                "total_stm_tokens": stm_stats["total_tokens"],
                "total_ltm_entries": ltm_stats["total_entries"],
                "stm_utilization": stm_stats["utilization"],
                "ltm_utilization": ltm_stats["utilization"],
            }
        }

    async def persist(self) -> bool:
        """Persist all memory to disk."""
        return await self.ltm.persist()

    async def load(self) -> bool:
        """Load persisted memory from disk."""
        return await self.ltm.load()

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get current working context from STM.
        Useful for building prompts.
        """
        return self.stm.get_context_window(max_tokens)

    def should_prune(self) -> bool:
        """Check if memory should be pruned."""
        return self.stm._should_prune()

    async def clear_stm(self) -> None:
        """Clear short-term memory."""
        await self.stm.clear()

    async def cleanup(self) -> Dict[str, int]:
        """
        Run cleanup operations on all memory systems.

        Returns:
            Cleanup statistics
        """
        stats = {"ltm_cold_deleted": 0}

        # Clean up old cold tier LTM entries
        deleted = await self.ltm.cleanup_cold_tier()
        stats["ltm_cold_deleted"] = deleted

        return stats


# Convenience function to create AgeMem with default settings
def create_agemem(
    persistence_dir: Optional[str] = None,
    **kwargs
) -> AgeMem:
    """
    Factory function to create AgeMem instance.

    Args:
        persistence_dir: Optional custom persistence directory
        **kwargs: Additional MemoryConfig parameters

    Returns:
        Configured AgeMem instance
    """
    config_kwargs = kwargs.copy()
    if persistence_dir:
        config_kwargs["persistence_dir"] = persistence_dir

    config = MemoryConfig(**config_kwargs)
    return AgeMem(config=config)
