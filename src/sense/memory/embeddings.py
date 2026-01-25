"""
SENSE-v2 Embedding Provider
Handles text embeddings for semantic memory search.
Optimized for 128GB UMA architecture.
"""

from typing import List, Optional, Union
import numpy as np
import logging
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embedding provider using sentence-transformers.
    Default model: all-MiniLM-L6-v2 (384 dimensions).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.device = device
        self._batch_size = batch_size
        self._model = None
        self._dimension = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                self.logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                self.logger.warning(
                    "sentence-transformers not installed, using fallback embeddings"
                )
                self._model = "fallback"
                self._dimension = 384

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        self._load_model()

        if self._model == "fallback":
            return self._fallback_embed(text)

        if isinstance(text, str):
            text = [text]

        embeddings = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )

        return embeddings

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        self._load_model()
        batch_size = batch_size or self._batch_size

        if self._model == "fallback":
            return self._fallback_embed(texts)

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
            batch_size=batch_size,
        )

        return embeddings

    def _fallback_embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Fallback embedding using simple hash-based approach.
        Used when sentence-transformers is not available.
        """
        if isinstance(text, str):
            text = [text]

        embeddings = []
        for t in text:
            # Simple deterministic embedding based on character hashing
            np.random.seed(hash(t) % (2**32))
            emb = np.random.randn(self._dimension).astype(np.float32)
            emb = emb / np.linalg.norm(emb)  # Normalize
            embeddings.append(emb)

        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        self._load_model()
        return self._dimension


class CachedEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider with caching for frequently accessed texts.
    Reduces redundant computation for repeated queries.
    """

    def __init__(
        self,
        base_provider: EmbeddingProvider,
        cache_size: int = 10000,
    ):
        self.base_provider = base_provider
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings with caching."""
        if isinstance(text, str):
            text = [text]

        results = []
        uncached = []
        uncached_indices = []

        for i, t in enumerate(text):
            cache_key = hash(t)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
                self._update_access(cache_key)
            else:
                results.append(None)
                uncached.append(t)
                uncached_indices.append(i)

        if uncached:
            new_embeddings = self.base_provider.embed(uncached)
            for idx, emb in zip(uncached_indices, new_embeddings):
                cache_key = hash(text[idx])
                self._add_to_cache(cache_key, emb)
                results[idx] = emb

        return np.array(results)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for batch with caching."""
        return self.embed(texts)

    def _add_to_cache(self, key: int, value: np.ndarray):
        """Add item to cache with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # Evict least recently used
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = value
        self._access_order.append(key)

    def _update_access(self, key: int):
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.base_provider.dimension

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_size": self.cache_size,
            "utilization": len(self._cache) / self.cache_size if self.cache_size > 0 else 0,
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple vectors.
    Optimized for batch operations.
    """
    # Normalize query
    query_norm = query / (np.linalg.norm(query) + 1e-8)

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-8)

    # Batch dot product
    similarities = np.dot(vectors_norm, query_norm)

    return similarities
