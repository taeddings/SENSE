"""
Integration Manager.

Coordinates discovery, generation, and loading of external tools.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from .discovery_engine import DiscoveryEngine
from .wrapper_generator import WrapperGenerator

class IntegrationManager:
    """
    Manages end-to-end tool integration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("IntegrationManager")
        self.discovery = DiscoveryEngine(config)
        self.generator = WrapperGenerator()
        self.plugins_dir = Path("src/sense/plugins/user_defined") # Re-use user_defined for now

    async def auto_integrate(self, task: str) -> List[str]:
        """
        Discover and integrate tools for a task.
        Returns list of generated plugin names.
        """
        # 1. Discover
        tools = await self.discovery.discover_tools(task)
        if not tools:
            return []

        integrated = []
        for tool in tools:
            # 2. Generate Wrapper
            code = self.generator.generate_wrapper(tool)
            
            # 3. Save (Stub: In real impl, write to file and hot-load)
            # For this prototype, we just simulate the save
            # filename = f"{tool.name.lower().replace('-', '_')}_plugin.py"
            # (self.plugins_dir / filename).write_text(code)
            
            self.logger.info(f"Generated wrapper for {tool.name}")
            integrated.append(tool.name)

        return integrated
