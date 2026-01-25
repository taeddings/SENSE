"""
Tool Discovery Engine.

Searches external sources (PyPI, GitHub) for Python libraries matching task requirements.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import asyncio
import re

@dataclass
class DiscoveredTool:
    """A discovered external tool/library."""
    name: str
    source: str  # "pypi", "github"
    description: str
    install_command: str
    popularity_score: float  # 0-1 normalized
    documentation_url: Optional[str] = None

class DiscoveryEngine:
    """
    Automatically discovers tools based on task requirements.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("DiscoveryEngine")
        self.cache = {}

    async def discover_tools(self, task: str, max_results: int = 3) -> List[DiscoveredTool]:
        """
        Discover tools relevant to the task.
        """
        requirements = self._extract_requirements(task)
        tools = []

        for req in requirements:
            if req in self.cache:
                tools.extend(self.cache[req])
                continue
            
            # Simulate PyPI search
            results = await self._search_pypi(req)
            self.cache[req] = results
            tools.extend(results)

        # Deduplicate and sort
        unique_tools = {t.name: t for t in tools}.values()
        sorted_tools = sorted(unique_tools, key=lambda t: t.popularity_score, reverse=True)
        
        return list(sorted_tools)[:max_results]

    def _extract_requirements(self, task: str) -> List[str]:
        """Extract keywords/requirements from task string."""
        task_lower = task.lower()
        reqs = []
        
        patterns = {
            r'image|photo|jpg|png': 'image-processing',
            r'graph|plot|chart': 'visualization',
            r'data|csv|pandas': 'data-analysis',
            r'web|scrape|http': 'web-scraping',
            r'api|request': 'http-client',
            r'math|calc': 'numpy',
        }
        
        for pattern, keyword in patterns.items():
            if re.search(pattern, task_lower):
                reqs.append(keyword)
                
        return reqs if reqs else ['general-utility']

    async def _search_pypi(self, query: str) -> List[DiscoveredTool]:
        """Search PyPI (Stub)."""
        await asyncio.sleep(0.5)
        
        # Stub results based on query
        if query == 'image-processing':
            return [
                DiscoveredTool("Pillow", "pypi", "Python Imaging Library", "pip install Pillow", 0.95),
                DiscoveredTool("opencv-python", "pypi", "Computer Vision", "pip install opencv-python", 0.90)
            ]
        elif query == 'visualization':
            return [
                DiscoveredTool("matplotlib", "pypi", "Plotting library", "pip install matplotlib", 0.95),
                DiscoveredTool("seaborn", "pypi", "Statistical data visualization", "pip install seaborn", 0.85)
            ]
        elif query == 'data-analysis':
             return [DiscoveredTool("pandas", "pypi", "Data analysis", "pip install pandas", 0.99)]
        
        return []
