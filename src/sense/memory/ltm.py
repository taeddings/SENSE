"""
SENSE-v2 Long-Term Memory (LTM)
Vector database-backed persistent memory for knowledge storage.
Per SYSTEM_PROMPT: Uses vector database for LTM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import os
import logging
import asyncio
from pathlib import Path
import numpy as np

from sense.core.base import BaseMemory
from sense.core.config import MemoryConfig, MemoryTier
from sense.memory.embeddings import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    batch_cosine_similarity,
)


@dataclass
class LTMEntry:
    """A single entry in Long-Term Memory."""
    key: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tier: MemoryTier = MemoryTier.WARM

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry to dictionary."""
        return {
            "key": self.key,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "tier": self.tier.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> "LTMEntry":
        """Deserialize entry from dictionary."""
        return cls(
            key=data["key"],
            content=data["content"],
            embedding=embedding,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            accessed_at=datetime.fromisoformat(data.get("accessed_at", datetime.now().isoformat())),
            access_count=data.get("access_count", 0),
            tier=MemoryTier(data.get("tier", "warm")),
        )

    def record_access(self):
        """Record an access to this entry."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class LongTermMemory(BaseMemory):
    """
    Long-Term Memory implementation using vector storage.

    Supports multiple backends:
    - ChromaDB (default, recommended for production)
    - In-memory (for testing/development)

    Per SYSTEM_PROMPT requirements:
    - Vector database for semantic search
    - Tiered storage (hot/warm/cold)
    - Optimized for 128GB UMA
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        super().__init__(config)
        self.config = config or MemoryConfig()
        self.embedding_provider = embedding_provider or SentenceTransformerProvider()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Storage
        self._entries: Dict[str, LTMEntry] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._keys: List[str] = []

        # ChromaDB client (lazy initialized)
        self._chroma_client = None
        self._chroma_collection = None

        # Ensure persistence directory exists
        Path(self.config.persistence_dir).mkdir(parents=True, exist_ok=True)

    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        if self._chroma_client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            persist_path = os.path.join(self.config.persistence_dir, "chromadb")
            Path(persist_path).mkdir(parents=True, exist_ok=True)

            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_path,
                anonymized_telemetry=False,
            ))

            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.config.ltm_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self.logger.info(f"Initialized ChromaDB at {persist_path}")

        except ImportError:
            self.logger.warning(
                "ChromaDB not installed, using in-memory storage. "
                "Install with: pip install chromadb"
            )
            self._chroma_client = "fallback"

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Store a value in Long-Term Memory.

        Args:
            key: Unique identifier for the entry
            value: Content to store (will be converted to string)
            metadata: Optional metadata dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            content = str(value)
            embedding = self.embedding_provider.embed(content)[0]

            entry = LTMEntry(
                key=key,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
            )

            self._entries[key] = entry

            # Update in-memory index
            self._update_index(key, embedding)

            # Store in ChromaDB if available
            self._init_chromadb()
            if self._chroma_collection is not None:
                self._chroma_collection.upsert(
                    ids=[key],
                    embeddings=[embedding.tolist()],
                    documents=[content],
                    metadatas=[metadata or {}],
                )

            self.logger.debug(f"Stored LTM entry: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store LTM entry {key}: {e}")
            return False

    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from Long-Term Memory by key.

        Args:
            key: The key to retrieve

        Returns:
            The stored content, or None if not found
        """
        if key in self._entries:
            entry = self._entries[key]
            entry.record_access()
            return entry.content

        # Try ChromaDB
        self._init_chromadb()
        if self._chroma_collection is not None:
            try:
                results = self._chroma_collection.get(ids=[key])
                if results and results["documents"]:
                    return results["documents"][0]
            except Exception:
                pass

        return None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search Long-Term Memory using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold (default from config)

        Returns:
            List of matching entries with similarity scores
        """
        threshold = threshold or self.config.ltm_similarity_threshold
        query_embedding = self.embedding_provider.embed(query)[0]

        results = []

        # Search ChromaDB if available
        self._init_chromadb()
        if self._chroma_collection is not None and self._chroma_client != "fallback":
            try:
                chroma_results = self._chroma_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )

                if chroma_results and chroma_results["documents"]:
                    for i, doc in enumerate(chroma_results["documents"][0]):
                        # ChromaDB returns distance, convert to similarity
                        distance = chroma_results["distances"][0][i]
                        similarity = 1 - distance

                        if similarity >= threshold:
                            results.append({
                                "key": chroma_results["ids"][0][i],
                                "content": doc,
                                "metadata": chroma_results["metadatas"][0][i] if chroma_results["metadatas"] else {},
                                "similarity": similarity,
                            })

                return results

            except Exception as e:
                self.logger.warning(f"ChromaDB search failed, using in-memory: {e}")

        # Fallback to in-memory search
        if self._embeddings is not None and len(self._keys) > 0:
            similarities = batch_cosine_similarity(query_embedding, self._embeddings)

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            for idx in top_indices:
                if similarities[idx] >= threshold:
                    key = self._keys[idx]
                    entry = self._entries.get(key)
                    if entry:
                        entry.record_access()
                        results.append({
                            "key": key,
                            "content": entry.content,
                            "metadata": entry.metadata,
                            "similarity": float(similarities[idx]),
                        })

        return results

    async def delete(self, key: str) -> bool:
        """
        Delete an entry from Long-Term Memory.

        Args:
            key: Key of entry to delete

        Returns:
            True if deleted, False if not found
        """
        if key not in self._entries:
            return False

        del self._entries[key]

        # Remove from index
        if key in self._keys:
            idx = self._keys.index(key)
            self._keys.pop(idx)
            if self._embeddings is not None:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)

        # Remove from ChromaDB
        self._init_chromadb()
        if self._chroma_collection is not None:
            try:
                self._chroma_collection.delete(ids=[key])
            except Exception:
                pass

        self.logger.debug(f"Deleted LTM entry: {key}")
        return True

    async def summarize_and_prune(self, content: str, target_ratio: float = 0.3) -> str:
        """
        Summarize content to reduce token count.
        This is a placeholder - actual summarization would use an LLM.

        Args:
            content: Content to summarize
            target_ratio: Target length ratio (0.3 = 30% of original)

        Returns:
            Summarized content
        """
        # Simple extractive summarization
        # In production, this would call an LLM for abstractive summarization
        sentences = content.split(". ")
        target_count = max(1, int(len(sentences) * target_ratio))

        # Keep first and last sentences, sample from middle
        if len(sentences) <= target_count:
            return content

        selected = [sentences[0]]
        if len(sentences) > 2:
            middle = sentences[1:-1]
            step = max(1, len(middle) // (target_count - 2))
            selected.extend(middle[::step][:target_count - 2])
        if len(sentences) > 1:
            selected.append(sentences[-1])

        return ". ".join(selected)

    def _update_index(self, key: str, embedding: np.ndarray):
        """Update in-memory embedding index."""
        if key in self._keys:
            idx = self._keys.index(key)
            self._embeddings[idx] = embedding
        else:
            self._keys.append(key)
            if self._embeddings is None:
                self._embeddings = embedding.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, embedding])

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_entries = len(self._entries)

        # Calculate tier distribution
        tier_counts = {tier.value: 0 for tier in MemoryTier}
        for entry in self._entries.values():
            tier_counts[entry.tier.value] += 1

        # Calculate memory size estimate
        embedding_size = 0
        if self._embeddings is not None:
            embedding_size = self._embeddings.nbytes

        return {
            "total_entries": total_entries,
            "max_entries": self.config.ltm_max_entries,
            "utilization": total_entries / self.config.ltm_max_entries if self.config.ltm_max_entries > 0 else 0,
            "tier_distribution": tier_counts,
            "embedding_dimension": self.embedding_provider.dimension,
            "embedding_memory_bytes": embedding_size,
        }

    async def persist(self) -> bool:
        """Persist in-memory entries to disk."""
        try:
            persist_path = os.path.join(self.config.persistence_dir, "ltm_entries.json")
            embedding_path = os.path.join(self.config.persistence_dir, "ltm_embeddings.npy")

            # Save entries metadata
            entries_data = {key: entry.to_dict() for key, entry in self._entries.items()}
            with open(persist_path, "w") as f:
                json.dump(entries_data, f, indent=2)

            # Save embeddings
            if self._embeddings is not None:
                np.save(embedding_path, self._embeddings)

            self.logger.info(f"Persisted {len(self._entries)} LTM entries")
            return True

        except Exception as e:
            self.logger.error(f"Failed to persist LTM: {e}")
            return False

    async def load(self) -> bool:
        """Load persisted entries from disk."""
        try:
            persist_path = os.path.join(self.config.persistence_dir, "ltm_entries.json")
            embedding_path = os.path.join(self.config.persistence_dir, "ltm_embeddings.npy")

            if not os.path.exists(persist_path):
                return True  # Nothing to load

            # Load entries metadata
            with open(persist_path, "r") as f:
                entries_data = json.load(f)

            # Load embeddings
            embeddings = None
            if os.path.exists(embedding_path):
                embeddings = np.load(embedding_path)

            # Reconstruct entries
            self._entries = {}
            self._keys = []
            for i, (key, data) in enumerate(entries_data.items()):
                emb = embeddings[i] if embeddings is not None and i < len(embeddings) else None
                self._entries[key] = LTMEntry.from_dict(data, emb)
                self._keys.append(key)

            self._embeddings = embeddings

            self.logger.info(f"Loaded {len(self._entries)} LTM entries")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load LTM: {e}")
            return False

    async def cleanup_cold_tier(self, max_age_hours: int = 168) -> int:
        """
        Clean up cold tier entries older than max_age.

        Args:
            max_age_hours: Maximum age in hours before deletion

        Returns:
            Number of entries deleted
        """
        deleted = 0
        cutoff = datetime.now()

        keys_to_delete = []
        for key, entry in self._entries.items():
            if entry.tier == MemoryTier.COLD:
                age_hours = (cutoff - entry.accessed_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            await self.delete(key)
            deleted += 1

        if deleted > 0:
            self.logger.info(f"Cleaned up {deleted} cold tier entries")

        return deleted
