"""
SENSE-v2 Short-Term Memory (STM)
Active context management with summarize-and-prune capability.
Per SYSTEM_PROMPT: Triggers Summarize-and-Prune at 80% of model's context limit.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import logging
from collections import OrderedDict

from sense_v2.core.base import BaseMemory
from sense_v2.core.config import MemoryConfig
from sense_v2.core.schemas import AgentMessage, MessageRole


@dataclass
class STMEntry:
    """A single entry in Short-Term Memory."""
    key: str
    content: str
    token_count: int
    priority: float = 1.0  # Higher = more important to keep
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "content": self.content,
            "token_count": self.token_count,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class ShortTermMemory(BaseMemory):
    """
    Short-Term Memory for managing active conversation context.

    Features:
    - Token-aware capacity management
    - Automatic summarize-and-prune at threshold
    - Priority-based retention
    - Sliding window for recent context

    Per SYSTEM_PROMPT requirements:
    - Triggers at 80% of model's context limit
    - Summarizes to configurable ratio (default 30%)
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(config)
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ordered dict for FIFO-like access with key lookup
        self._entries: OrderedDict[str, STMEntry] = OrderedDict()
        self._total_tokens = 0

        # Custom summarizer function (or use default)
        self._summarizer = summarizer or self._default_summarizer

        # Prune callback for notification
        self._prune_callback: Optional[Callable[[int, int], None]] = None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text. ~4 chars per token."""
        return max(1, len(text) // 4)

    def _default_summarizer(self, content: str) -> str:
        """
        Default extractive summarizer.
        In production, replace with LLM-based summarization.
        """
        sentences = content.replace("\n", ". ").split(". ")
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 3:
            return content

        # Keep key sentences
        target = max(1, int(len(sentences) * self.config.stm_summary_ratio))

        # Select: first, last, and evenly spaced middle sentences
        selected = [sentences[0]]
        if len(sentences) > 2:
            step = max(1, (len(sentences) - 2) // max(1, target - 2))
            middle_indices = range(1, len(sentences) - 1, step)
            selected.extend(sentences[i] for i in list(middle_indices)[:target - 2])
        selected.append(sentences[-1])

        return ". ".join(selected)

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict] = None,
        priority: float = 1.0,
    ) -> bool:
        """
        Store a value in Short-Term Memory.

        Args:
            key: Unique identifier
            value: Content to store
            metadata: Optional metadata
            priority: Retention priority (higher = keep longer)

        Returns:
            True if successful
        """
        content = str(value)
        token_count = self._estimate_tokens(content)

        # Check if we need to prune first
        if self._should_prune(token_count):
            await self._auto_prune()

        # Update existing or add new
        if key in self._entries:
            old_entry = self._entries[key]
            self._total_tokens -= old_entry.token_count

        entry = STMEntry(
            key=key,
            content=content,
            token_count=token_count,
            priority=priority,
            metadata=metadata or {},
        )

        self._entries[key] = entry
        self._entries.move_to_end(key)  # Most recent at end
        self._total_tokens += token_count

        return True

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        if key in self._entries:
            # Move to end (most recently accessed)
            self._entries.move_to_end(key)
            return self._entries[key].content
        return None

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Simple keyword search in STM.
        STM is small enough that semantic search is usually overkill.
        """
        query_lower = query.lower()
        results = []

        for key, entry in self._entries.items():
            if query_lower in entry.content.lower():
                results.append({
                    "key": key,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "priority": entry.priority,
                })

        # Sort by priority and recency
        results.sort(key=lambda x: x["priority"], reverse=True)
        return results[:top_k]

    async def delete(self, key: str) -> bool:
        """Delete an entry from STM."""
        if key in self._entries:
            entry = self._entries.pop(key)
            self._total_tokens -= entry.token_count
            return True
        return False

    async def summarize_and_prune(
        self,
        content: str,
        target_ratio: float = 0.3,
    ) -> str:
        """Summarize content to target ratio."""
        target_ratio = target_ratio or self.config.stm_summary_ratio
        return self._summarizer(content)

    def _should_prune(self, additional_tokens: int = 0) -> bool:
        """
        Check if STM should be pruned.
        Triggers at 80% of max tokens (configurable).
        """
        projected_tokens = self._total_tokens + additional_tokens
        threshold = self.config.stm_max_tokens * self.config.stm_prune_threshold
        return projected_tokens > threshold

    async def _auto_prune(self) -> int:
        """
        Automatically prune STM to make room.
        Uses priority-based eviction.

        Returns:
            Number of tokens freed
        """
        target_tokens = int(self.config.stm_max_tokens * self.config.stm_summary_ratio)
        tokens_freed = 0

        self.logger.info(
            f"Auto-pruning STM: {self._total_tokens} tokens -> target {target_tokens}"
        )

        # Sort entries by priority (lowest first for eviction)
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: (x[1].priority, x[1].created_at)
        )

        entries_to_remove = []
        entries_to_summarize = []

        for key, entry in sorted_entries:
            if self._total_tokens - tokens_freed <= target_tokens:
                break

            if entry.priority < 0.5:
                # Low priority: remove entirely
                entries_to_remove.append(key)
                tokens_freed += entry.token_count
            else:
                # Higher priority: summarize
                entries_to_summarize.append(key)

        # Remove low-priority entries
        for key in entries_to_remove:
            await self.delete(key)

        # Summarize remaining entries
        for key in entries_to_summarize:
            if self._total_tokens - tokens_freed <= target_tokens:
                break

            entry = self._entries[key]
            original_tokens = entry.token_count

            summarized = await self.summarize_and_prune(entry.content)
            new_tokens = self._estimate_tokens(summarized)

            if new_tokens < original_tokens:
                entry.content = summarized
                entry.token_count = new_tokens
                self._total_tokens -= (original_tokens - new_tokens)
                tokens_freed += (original_tokens - new_tokens)

                entry.metadata["summarized"] = True
                entry.metadata["original_tokens"] = original_tokens

        # Notify callback if set
        if self._prune_callback:
            self._prune_callback(tokens_freed, self._total_tokens)

        self.logger.info(f"Pruned {tokens_freed} tokens, now at {self._total_tokens}")
        return tokens_freed

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get STM usage statistics."""
        return {
            "entry_count": len(self._entries),
            "total_tokens": self._total_tokens,
            "max_tokens": self.config.stm_max_tokens,
            "utilization": self._total_tokens / self.config.stm_max_tokens if self.config.stm_max_tokens > 0 else 0,
            "prune_threshold": self.config.stm_prune_threshold,
            "should_prune": self._should_prune(),
        }

    def get_context_window(self, max_tokens: Optional[int] = None) -> str:
        """
        Get recent context as a single string.
        Useful for constructing prompts.
        """
        max_tokens = max_tokens or self.config.stm_max_tokens

        context_parts = []
        token_count = 0

        # Iterate in reverse (most recent first)
        for key in reversed(self._entries):
            entry = self._entries[key]
            if token_count + entry.token_count > max_tokens:
                break
            context_parts.insert(0, entry.content)
            token_count += entry.token_count

        return "\n\n".join(context_parts)

    def set_prune_callback(self, callback: Callable[[int, int], None]) -> None:
        """Set callback for prune events. Args: (tokens_freed, tokens_remaining)."""
        self._prune_callback = callback

    async def clear(self) -> None:
        """Clear all STM entries."""
        self._entries.clear()
        self._total_tokens = 0

    def add_message(self, message: AgentMessage, priority: Optional[float] = None) -> None:
        """
        Convenience method to add an AgentMessage to STM.
        System messages get higher priority by default.
        """
        if priority is None:
            priority_map = {
                MessageRole.SYSTEM: 1.0,
                MessageRole.ASSISTANT: 0.7,
                MessageRole.USER: 0.8,
                MessageRole.TOOL: 0.5,
            }
            priority = priority_map.get(message.role, 0.5)

        import asyncio
        asyncio.create_task(
            self.store(
                key=f"msg_{message.timestamp.timestamp()}",
                value=message.content,
                metadata={"role": message.role.value},
                priority=priority,
            )
        )
