"""
SENSE v4.0: Knowledge RAG Module

Semantic knowledge retrieval backed by vector embeddings.
Provides context enrichment and fact-checking capabilities.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from sense.memory.embeddings import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    batch_cosine_similarity
)
from sense.memory.bridge import UniversalMemory


@dataclass
class Document:
    """
    Document for vector storage.

    Attributes:
        content: The document text
        metadata: Additional document metadata
        embedding: Cached embedding vector
        source: Source of the document
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    source: str = "unknown"

    def __hash__(self):
        """Make document hashable for deduplication."""
        return hash(self.content)

    def __eq__(self, other):
        """Documents are equal if content matches."""
        if not isinstance(other, Document):
            return False
        return self.content == other.content


@dataclass
class SearchResult:
    """
    Search result with similarity score.

    Attributes:
        document: The retrieved document
        score: Similarity score (0.0 - 1.0)
        rank: Result ranking position
    """
    document: Document
    score: float
    rank: int = 0


@dataclass
class FactCheckResult:
    """
    Fact-checking result.

    Attributes:
        claim: The claim being checked
        confidence: Confidence in the fact-check (0.0 - 1.0)
        supporting_evidence: Evidence that supports the claim
        contradicting_evidence: Evidence that contradicts the claim
        verdict: "supported", "contradicted", or "uncertain"
    """
    claim: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    verdict: str = "uncertain"


class NumpyFallbackIndex:
    """
    Numpy-based fallback for when FAISS is unavailable.

    Simple brute-force similarity search.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.ntotal = 0

    def add(self, embeddings: np.ndarray):
        """Add vectors to index."""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        self.vectors.append(embeddings)
        self.ntotal += embeddings.shape[0]

    def search(self, query: np.ndarray, k: int = 5):
        """
        Search for top-k similar vectors.

        Returns:
            (similarities, indices) tuple
        """
        if not self.vectors:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)

        # Concatenate all vectors
        all_vectors = np.vstack(self.vectors)

        # Compute similarities
        similarities = batch_cosine_similarity(query, all_vectors)

        # Get top-k
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = similarities[top_indices]

        return top_scores.reshape(1, -1), top_indices.reshape(1, -1)


class VectorStore:
    """
    FAISS-backed vector store with graceful fallback.

    Uses existing EmbeddingProvider from memory/embeddings.py
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        dimension: int = 384,
        use_faiss: bool = True
    ):
        """
        Initialize vector store.

        Args:
            embedding_provider: Provider for text embeddings
            dimension: Embedding dimension
            use_faiss: Whether to use FAISS (falls back to numpy if unavailable)
        """
        self.embedder = embedding_provider or SentenceTransformerProvider()
        self.dimension = dimension
        self.documents: List[Document] = []
        self.logger = logging.getLogger("Intelligence.VectorStore")

        # Try to initialize FAISS
        self.index = None
        if use_faiss:
            try:
                import faiss
                self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim)
                self.logger.info("FAISS index initialized")
            except ImportError:
                self.logger.warning("FAISS not available, using numpy fallback")
                self.index = NumpyFallbackIndex(dimension)
        else:
            self.logger.info("Using numpy fallback (FAISS disabled)")
            self.index = NumpyFallbackIndex(dimension)

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to vector store.

        Args:
            docs: List of documents to add
        """
        if not docs:
            return

        # Deduplicate
        unique_docs = list(set(docs))

        # Generate embeddings if not cached
        texts_to_embed = []
        for doc in unique_docs:
            if doc.embedding is None:
                texts_to_embed.append(doc.content)

        if texts_to_embed:
            embeddings = self.embedder.embed(texts_to_embed)
            embed_idx = 0
            for doc in unique_docs:
                if doc.embedding is None:
                    doc.embedding = embeddings[embed_idx]
                    embed_idx += 1

        # Add to index
        embeddings_matrix = np.vstack([doc.embedding for doc in unique_docs])

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        embeddings_matrix = embeddings_matrix / (norms + 1e-8)

        self.index.add(embeddings_matrix.astype(np.float32))
        self.documents.extend(unique_docs)

        self.logger.info(f"Added {len(unique_docs)} documents to vector store (total: {len(self.documents)})")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Semantic search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # Edge case: empty store
        if len(self.documents) == 0:
            return []

        # Embed query
        query_embedding = self.embedder.embed(query)

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Search
        top_k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_norm.astype(np.float32), top_k)

        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < len(self.documents):  # Safety check
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank
                ))

        return results

    def clear(self):
        """Clear all documents from the store."""
        self.documents = []
        # Reinitialize index
        if hasattr(self.index, 'reset'):
            self.index.reset()
        else:
            # Recreate numpy fallback
            self.index = NumpyFallbackIndex(self.dimension)


class KnowledgeRAG:
    """
    Retrieval-Augmented Generation pipeline.

    Enriches prompts with relevant context from vector store and memory.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        memory: Optional[UniversalMemory] = None,
        max_context_tokens: int = 500
    ):
        """
        Initialize Knowledge RAG.

        Args:
            vector_store: Vector store for semantic search
            memory: UniversalMemory for episodic recall
            max_context_tokens: Maximum tokens for retrieved context
        """
        self.store = vector_store or VectorStore()
        self.memory = memory or UniversalMemory()
        self.max_context_tokens = max_context_tokens
        self.logger = logging.getLogger("Intelligence.KnowledgeRAG")

    def retrieve_context(self, task: str, top_k: int = 5) -> str:
        """
        Retrieve relevant context for a task.

        Process:
        1. Search vector store for semantic matches
        2. Merge with UniversalMemory recalls
        3. Rank and deduplicate
        4. Truncate to max_tokens

        Args:
            task: The task to retrieve context for
            top_k: Number of results from vector store

        Returns:
            Formatted context string
        """
        context_pieces = []

        # 1. Vector store search
        if len(self.store.documents) > 0:
            search_results = self.store.search(task, top_k=top_k)
            for result in search_results:
                if result.score > 0.3:  # Relevance threshold
                    context_pieces.append({
                        'content': result.document.content,
                        'score': result.score,
                        'source': 'vector_store'
                    })

        # 2. Memory recall
        memory_results = self.memory.recall(task)
        for mem in memory_results:
            context_pieces.append({
                'content': mem,
                'score': 0.8,  # Assume high relevance from memory
                'source': 'episodic_memory'
            })

        # 3. Deduplicate
        seen_content = set()
        unique_pieces = []
        for piece in context_pieces:
            content_lower = piece['content'].lower()
            if content_lower not in seen_content:
                seen_content.add(content_lower)
                unique_pieces.append(piece)

        # 4. Sort by score
        unique_pieces.sort(key=lambda x: x['score'], reverse=True)

        # 5. Truncate to token budget
        formatted_context = self._format_context(unique_pieces)

        return formatted_context

    def _format_context(self, pieces: List[Dict]) -> str:
        """
        Format context pieces into a single string with token budget.

        Approximate tokens as words * 1.3
        """
        lines = ["### RETRIEVED CONTEXT:"]
        current_tokens = 3  # "### RETRIEVED CONTEXT:"

        for piece in pieces:
            content = piece['content']
            source = piece['source']

            # Estimate tokens
            word_count = len(content.split())
            estimated_tokens = int(word_count * 1.3)

            if current_tokens + estimated_tokens > self.max_context_tokens:
                break  # Budget exhausted

            lines.append(f"- [{source}] {content}")
            current_tokens += estimated_tokens

        if len(lines) == 1:
            return ""  # No context retrieved

        return '\n'.join(lines)

    def fact_check(self, claim: str, sources: List[str]) -> FactCheckResult:
        """
        Cross-reference a claim against known facts.

        Args:
            claim: The claim to verify
            sources: List of source texts to check against

        Returns:
            FactCheckResult with verdict and evidence
        """
        # Add sources as temporary documents
        temp_docs = [Document(content=src, source="fact_check") for src in sources]

        # Temporarily add to vector store
        original_doc_count = len(self.store.documents)
        self.store.add_documents(temp_docs)

        # Search for relevant evidence
        results = self.store.search(claim, top_k=5)

        supporting = []
        contradicting = []

        for result in results:
            if result.score > 0.7:
                # High similarity = likely supporting
                supporting.append(result.document.content)
            elif result.score < 0.3:
                # Low similarity = possibly contradicting
                contradicting.append(result.document.content)

        # Remove temporary documents (restore original state)
        self.store.documents = self.store.documents[:original_doc_count]

        # Determine verdict
        if len(supporting) > len(contradicting):
            verdict = "supported"
            confidence = 0.7 + (len(supporting) * 0.05)
        elif len(contradicting) > len(supporting):
            verdict = "contradicted"
            confidence = 0.7 + (len(contradicting) * 0.05)
        else:
            verdict = "uncertain"
            confidence = 0.5

        confidence = min(1.0, confidence)

        return FactCheckResult(
            claim=claim,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            verdict=verdict
        )

    def add_knowledge(self, content: str, source: str = "user", metadata: Optional[Dict] = None):
        """
        Add knowledge to the RAG system.

        Args:
            content: Knowledge content
            source: Source of the knowledge
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}

        doc = Document(
            content=content,
            source=source,
            metadata=metadata
        )

        self.store.add_documents([doc])
        self.logger.info(f"Added knowledge from {source}: {content[:50]}...")
