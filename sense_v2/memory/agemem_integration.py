"""
SENSE-v2 ReasoningMemoryManager
Integration layer for reasoning traces with AgeMem memory system.

Part of Sprint 2: The Brain

Handles:
- Storage of ReasoningTrace objects in memory
- Retrieval-augmented reasoning (similar trace lookup)
- Consolidation of reasoning patterns
- Drift-aware memory management
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging
import asyncio

from sense_v2.core.base import BaseMemory
from sense_v2.core.config import MemoryConfig
from sense_v2.memory.agemem import AgeMem, MemoryType, MemoryEntry
from sense_v2.memory.engram_schemas import ReasoningTrace, DriftSnapshot
from sense_v2.memory.embeddings import EmbeddingProvider


@dataclass
class ReasoningMemoryConfig:
    """
    Configuration for reasoning memory integration.

    Extends base memory config with reasoning-specific parameters.
    """
    # Embedding dimensions for reasoning traces
    trace_embedding_dim: int = 384  # Matches sentence-transformer default

    # Retrieval parameters
    max_similar_traces: int = 5
    similarity_threshold: float = 0.7

    # Consolidation parameters
    consolidation_batch_size: int = 10
    min_traces_for_pattern: int = 3

    # Drift sensitivity
    drift_memory_boost: float = 1.2  # Multiplier for high-drift scenarios


class ReasoningMemoryManager(BaseMemory):
    """
    ReasoningMemoryManager - Integration layer for reasoning traces.

    Manages the storage, retrieval, and analysis of reasoning traces
    within the AgeMem memory system.

    Key Features:
    - Store complete reasoning traces for future retrieval
    - Retrieval-augmented reasoning via semantic search
    - Automatic consolidation of successful patterns
    - Drift-aware memory prioritization

    Integration Points:
    - Uses AgeMem for unified STM/LTM storage
    - Leverages FAISS for embedding-based retrieval
    - Connects with PopulationManager for evolutionary feedback
    """

    def __init__(
        self,
        agemem: AgeMem,
        config: Optional[ReasoningMemoryConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        super().__init__(config)
        self.config = config or ReasoningMemoryConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.agemem = agemem
        self.embedding_provider = embedding_provider or self.agemem.embedding_provider

        # Cache for frequently accessed traces
        self._trace_cache: Dict[str, ReasoningTrace] = {}

        # Statistics
        self.traces_stored = 0
        self.retrievals_performed = 0
        self.consolidations_done = 0

    async def store_reasoning_trace(self, trace: ReasoningTrace) -> bool:
        """
        Store a reasoning trace in the memory system.

        Args:
            trace: The ReasoningTrace to store

        Returns:
            bool: True if stored successfully

        The trace is stored with:
        - Problem embedding for retrieval
        - Metadata for filtering and analysis
        - Automatic tiering based on success/drift
        """
        try:
            # Generate embedding if not present
            if trace.problem_embedding is None:
                trace.problem_embedding = await self._generate_problem_embedding(trace)

            # Create memory entry - store problem + answer for semantic search
            searchable_content = f"{trace.problem_description}\n{trace.final_answer}"
            entry = MemoryEntry(
                key=f"reasoning_trace_{trace.trace_id}",
                content=searchable_content,
                memory_type=MemoryType.LTM if trace.should_flag_for_ltm else MemoryType.STM,
                priority=self._calculate_trace_priority(trace),
                metadata={
                    "trace_id": trace.trace_id,
                    "task_id": trace.task_id,
                    "genome_id": trace.genome_id,
                    "generation_id": trace.generation_id,
                    "outcome": trace.outcome,
                    "grounding_score": trace.grounding_score,
                    "reasoning_tokens": trace.reasoning_tokens_used,
                    "execution_time_ms": trace.execution_time_ms,
                    "drift_level": trace.drift_context.drift_level if trace.drift_context else 0.0,
                    "thought_chain_length": len(trace.thought_chain),
                    "tool_calls_count": len(trace.tool_calls),
                },
                tier=self._determine_trace_tier(trace),
            )

            # Store in AgeMem
            success = await self.agemem.store_async(entry)

            if success:
                # Cache the full trace
                self._trace_cache[trace.trace_id] = trace
                self.traces_stored += 1

                # Trigger consolidation if needed
                if self._should_consolidate():
                    await self._consolidate_reasoning_patterns()

            return success

        except Exception as e:
            self.logger.error(f"Failed to store reasoning trace {trace.trace_id}: {e}")
            return False

    async def retrieve_similar_traces(
        self,
        problem_description: str,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[ReasoningTrace, float]]:
        """
        Retrieve similar reasoning traces for retrieval-augmented reasoning.

        Args:
            problem_description: Text description of the current problem
            k: Number of traces to retrieve (default: config.max_similar_traces)
            similarity_threshold: Minimum similarity score (default: config.similarity_threshold)

        Returns:
            List of (ReasoningTrace, similarity_score) tuples
        """
        k = k or self.config.max_similar_traces
        threshold = similarity_threshold or self.config.similarity_threshold

        try:
            # Search AgeMem for similar traces using semantic search on stored content
            # The content includes problem_description, so it should match
            search_results = await self.agemem.search(
                query=problem_description,
                top_k=k * 2,  # Get more candidates for filtering
                memory_type=MemoryType.LTM,
            )

            similar_traces = []
            for result in search_results:
                score = result.get("similarity", 0.0)
                if score >= threshold and "trace_id" in result.get("metadata", {}):
                    trace = await self._load_full_trace(result["metadata"]["trace_id"])
                    if trace:
                        similar_traces.append((trace, score))

            # Sort by score and limit to k
            similar_traces.sort(key=lambda x: x[1], reverse=True)
            result = similar_traces[:k]

            self.retrievals_performed += 1
            return result

        except Exception as e:
            self.logger.error(f"Failed to retrieve similar traces: {e}")
            return []

    async def get_traces_by_genome(self, genome_id: str, limit: int = 50) -> List[ReasoningTrace]:
        """
        Get all reasoning traces for a specific genome.

        Args:
            genome_id: ID of the genome
            limit: Maximum number of traces to return

        Returns:
            List of ReasoningTrace objects
        """
        try:
            # Search with broad query and filter by genome_id in metadata
            # Use empty string to get all, but limit results
            search_results = await self.agemem.search(
                query="",  # Broad search
                top_k=limit * 2,  # Get more to filter
                memory_type=MemoryType.LTM,
            )

            traces = []
            for result in search_results:
                metadata = result.get("metadata", {})
                if metadata.get("genome_id") == genome_id and "trace_id" in metadata:
                    trace = await self._load_full_trace(metadata["trace_id"])
                    if trace:
                        traces.append(trace)
                        if len(traces) >= limit:
                            break

            return traces

        except Exception as e:
            self.logger.error(f"Failed to get traces for genome {genome_id}: {e}")
            return []

    async def get_successful_patterns(self, min_occurrences: Optional[int] = None) -> Dict[str, List[ReasoningTrace]]:
        """
        Extract successful reasoning patterns from stored traces.

        Args:
            min_occurrences: Minimum occurrences to consider a pattern

        Returns:
            Dict mapping pattern signatures to lists of traces
        """
        min_occ = min_occurrences or self.config.min_traces_for_pattern

        try:
            # Search with broad query and filter by success criteria
            search_results = await self.agemem.search(
                query="",  # Broad search
                top_k=1000,  # Get many results
                memory_type=MemoryType.LTM,
            )

            patterns = {}
            for result in search_results:
                metadata = result.get("metadata", {})
                if (metadata.get("outcome") == True and
                    metadata.get("grounding_score", 0.0) >= 0.8 and
                    "trace_id" in metadata):
                    trace = await self._load_full_trace(metadata["trace_id"])
                    if trace:
                        pattern_sig = self._extract_pattern_signature(trace)
                        if pattern_sig not in patterns:
                            patterns[pattern_sig] = []
                        patterns[pattern_sig].append(trace)

            # Filter by minimum occurrences
            return {sig: traces for sig, traces in patterns.items() if len(traces) >= min_occ}

        except Exception as e:
            self.logger.error(f"Failed to extract successful patterns: {e}")
            return {}

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about reasoning memory usage.

        Returns:
            Dict with various statistics
        """
        return {
            "traces_stored": self.traces_stored,
            "retrievals_performed": self.retrievals_performed,
            "consolidations_done": self.consolidations_done,
            "cache_size": len(self._trace_cache),
            "avg_retrieval_success": self.retrievals_performed / max(1, self.traces_stored),
        }

    async def _generate_problem_embedding(self, trace: ReasoningTrace) -> List[float]:
        """Generate embedding for problem description."""
        text = f"{trace.problem_description} {' '.join(trace.thought_chain[:3])}"
        return await self.embedding_provider.embed_text(text)

    def _calculate_trace_priority(self, trace: ReasoningTrace) -> float:
        """Calculate storage priority for a trace."""
        priority = 1.0

        # Boost for successful traces
        if trace.outcome:
            priority += 0.5

        # Boost for well-grounded traces
        priority += trace.grounding_score * 0.3

        # Boost for high-drift contexts (learning opportunities)
        if trace.drift_context and trace.drift_context.drift_level > 0.5:
            priority *= self.config.drift_memory_boost

        return priority

    def _determine_trace_tier(self, trace: ReasoningTrace) -> str:
        """Determine the memory tier for a trace."""
        from sense_v2.core.config import MemoryTier

        if trace.should_flag_for_ltm:
            return MemoryTier.COLD  # Long-term storage
        elif trace.outcome and trace.grounding_score > 0.5:
            return MemoryTier.WARM  # Medium-term
        else:
            return MemoryTier.HOT   # Short-term, may be pruned

    def _should_consolidate(self) -> bool:
        """Check if consolidation should be triggered."""
        return self.traces_stored % self.config.consolidation_batch_size == 0

    async def _consolidate_reasoning_patterns(self) -> None:
        """Consolidate reasoning patterns from recent traces."""
        try:
            # Get recent successful traces from STM
            # Since we can't query STM directly, we'll trigger consolidation differently
            # For now, rely on the patterns extraction from LTM
            patterns = await self.get_successful_patterns()
            if patterns:
                for pattern_sig, traces in patterns.items():
                    await self._store_pattern_summary(pattern_sig, traces)

                self.consolidations_done += 1

        except Exception as e:
            self.logger.error(f"Failed to consolidate reasoning patterns: {e}")

    async def _store_pattern_summary(self, pattern_sig: str, traces: List[ReasoningTrace]) -> None:
        """Store a summary of a successful reasoning pattern."""
        summary = f"Pattern {pattern_sig}: {len(traces)} successful instances"

        entry = MemoryEntry(
            key=f"pattern_{pattern_sig}",
            content=summary,
            memory_type=MemoryType.LTM,
            priority=2.0,  # High priority for patterns
            metadata={
                "pattern_signature": pattern_sig,
                "instance_count": len(traces),
                "avg_grounding_score": sum(t.grounding_score for t in traces) / len(traces),
                "avg_tokens": sum(t.reasoning_tokens_used for t in traces) / len(traces),
            }
        )

        await self.agemem.store_async(entry)

    def _extract_pattern_signature(self, trace: ReasoningTrace) -> str:
        """Extract a signature representing the reasoning pattern."""
        # Simple signature based on thought chain length and tools used
        tool_types = set()
        for call in trace.tool_calls:
            if "tool_name" in call:
                tool_types.add(call["tool_name"])

        return f"len_{len(trace.thought_chain)}_tools_{'_'.join(sorted(tool_types))}"

    async def _load_full_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Load a full ReasoningTrace from cache or reconstruct from metadata."""
        # Check cache first
        if trace_id in self._trace_cache:
            return self._trace_cache[trace_id]

        # For now, return None as full traces need to be cached at storage time
        # In a full implementation, this would reconstruct from stored metadata
        return None