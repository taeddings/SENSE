"""AgeMem LTM for SENSE v3.0

Procedural RAG: Indexes successful workflows for retrieval.
"""

from typing import List, Dict, Any, Optional
import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # Optional embedding
    from faiss import IndexFlatL2  # Optional vector search
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger("AgeMem")

class AgeMem:
    """
    AgeMem: Adaptive memory with STM/LTM tiering and procedural RAG.
    Retrieves past successful workflows for similar goals.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stm = []  # Short-term: recent tasks
        self.ltm = []  # Long-term: persistent
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') if config.get('use_embeddings', True) and EMBEDDINGS_AVAILABLE else None
        self.index = IndexFlatL2(384) if self.embedder else None  # FAISS for vector search
        self.logger = logger

    def add_memory(self, task: str, plan: str, result: Any, success: bool):
        """Add to STM/LTM based on success and age."""
        memory = {'task': task, 'plan': plan, 'result': result, 'success': success}
        self.stm.append(memory)
        if success:
            self.ltm.append(memory)
            self._index_memory(memory)
        self._prune_stm()

    def _index_memory(self, memory: Dict[str, Any]):
        if self.embedder:
            embedding = self.embedder.encode(memory['task'])
            self.index.add(np.array([embedding]).astype('float32'))

    def _prune_stm(self):
        """Prune STM if over limit."""
        if len(self.stm) > self.config.get('stm_max_entries', 100):
            self.stm = self.stm[-self.config.get('stm_max_entries', 100):]

    async def retrieve_similar(self, task: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve similar successful workflows (procedural RAG)."""
        similarities = self._semantic_search(task, k)
        successful = [m for m in similarities if m['success']]
        self.logger.debug(f"Retrieved {len(successful)} similar memories for '{task[:50]}...' ")
        return successful

    def _semantic_search(self, task: str, k: int) -> List[Dict[str, Any]]:
        if not self.embedder:
            # Keyword fallback
            return [m for m in self.ltm if task.lower() in m['task'].lower()][:k]
        embedding = self.embedder.encode(task)
        distances, indices = self.index.search(np.array([embedding]).astype('float32'), k)
        return [self.ltm[i] for i in indices[0] if i < len(self.ltm)]

# Export
__all__ = ['AgeMem']