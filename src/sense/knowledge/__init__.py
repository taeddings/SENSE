"""
SENSE v4.0 Knowledge System.

Components:
- WebSearchEngine: Interfaces for search APIs
- KnowledgeRAG: Vector-based retrieval
- FactChecker: Verification against external sources
"""

from .web_search import WebSearchEngine, SearchResult
from .retrieval_system import KnowledgeRAG
from .fact_checker import FactChecker

class KnowledgeSystem:
    """
    Coordinator for Internet-Scale Knowledge Integration.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.rag = KnowledgeRAG(config)
        self.fact_checker = FactChecker(config)
        
    async def gather_context(self, task: str) -> str:
        """Retrieve external context for a task."""
        return await self.rag.retrieve_and_augment(task)
        
    async def verify_plan(self, plan: str) -> dict:
        """Check facts within a generated plan."""
        # Simple extraction of first line as claim for now
        claim = plan.split('\n')[0][:100]
        return await self.fact_checker.verify_claim(claim)

