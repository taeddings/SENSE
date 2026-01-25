import glob
import os
import logging
from typing import List
from sense.core.plugins.interface import PluginABC
from sense.config import ENABLE_HARVESTED_TOOLS

def load_all_plugins() -> List[PluginABC]:
    plugins = []
    logger = logging.getLogger("PluginLoader")
    
    # ... existing loading logic for standard plugins would go here ...
    
    if ENABLE_HARVESTED_TOOLS:
        from sense.tools.harvested.adapter import AgentZeroToolAdapter
        
        # Scan the harvested directory
        # Adjust path to be relative to project root or absolute
        harvested_path = "src/sense/tools/harvested/default/*"
        harvested_paths = glob.glob(harvested_path)
        
        for path in harvested_paths:
            if os.path.isdir(path):
                try:
                    plugin = AgentZeroToolAdapter(path)
                    plugins.append(plugin)
                    logger.info(f"✅ Harvested Tool Loaded: {plugin.name}")
                    print(f"✅ Harvested Tool Loaded: {plugin.name}") # Visual feedback
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {path}: {e}")
                    print(f"⚠️ Failed to load {path}: {e}")

    return plugins
