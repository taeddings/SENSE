import os
import sys
import logging
from sense.core.plugins.interface import PluginABC
from sense.tools.harvested.adapter import AgentZeroToolAdapter

# Define where tools live
# Since loader.py is in src/sense/core/plugins/, we go up 3 levels to src/sense/
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tools", "harvested")

def load_all_plugins():
    """
    INTELLIGENT TOOL SCANNER
    ------------------------
    Recursively scans the 'harvested' directory to discover and register tools.
    
    Discovery Logic:
    1. Root Files: Any .py file in 'harvested/' is a tool (excluding system files).
    2. Tool Bundles: Any folder in 'harvested/' is scanned for an entry point:
       - Priority 1: A script named same as folder (e.g., ddg_search/ddg_search.py)
       - Priority 2: main.py
       - Priority 3: __init__.py (if it contains a class, handled by adapter)
    
    Returns:
        List[PluginABC]: A list of ready-to-use tool adapters.
    """
    logger = logging.getLogger("PluginLoader")
    plugins = []
    
    # Create dir if not exists
    if not os.path.exists(TOOLS_DIR):
        os.makedirs(TOOLS_DIR, exist_ok=True)
        return plugins

    # Get list of all items in the tools directory
    items = os.listdir(TOOLS_DIR)
    
    for item in items:
        full_path = os.path.join(TOOLS_DIR, item)
        
        # SKIP SYSTEM FILES
        if item.startswith("_") or item in ["adapter.py", "requirements.txt"]:
            continue

        entry_point = None
        tool_name = None

        # --- CASE A: IT IS A FOLDER (Tool Bundle) ---
        if os.path.isdir(full_path):
            tool_name = item # Name of the tool is the folder name
            
            # Heuristic 1: Look for exact match (best practice)
            # e.g. tools/ddg_search/ddg_search.py
            exact_match = os.path.join(full_path, f"{item}.py")
            if os.path.exists(exact_match):
                entry_point = full_path # Adapter handles the folder path
            
            # Heuristic 2: Look for main.py
            elif os.path.exists(os.path.join(full_path, "main.py")):
                entry_point = full_path

            # Heuristic 3: Look for any single .py file if it's the only one
            else:
                py_files = [f for f in os.listdir(full_path) if f.endswith(".py") and not f.startswith("_")]
                if len(py_files) == 1:
                    # If there's only one script, that must be it
                    # We pass the folder, Adapter finds the script
                    entry_point = full_path
        
        # --- CASE B: IT IS A FILE (Standalone Tool) ---
        elif os.path.isfile(full_path) and item.endswith(".py"):
            # e.g. tools/yt_download.py
            tool_name = os.path.splitext(item)[0]
            # For flat files in root, we point to the parent dir.
            entry_point = TOOLS_DIR

        # --- REGISTER ---
        if entry_point and tool_name:
            try:
                # Initialize the Universal Adapter
                # We tell it: "Here is where the tool lives."
                plugin_instance = AgentZeroToolAdapter(entry_point)
                
                # FORCE THE NAME:
                # This ensures [ddg_search] maps to this adapter, regardless of filename
                plugin_instance.name = tool_name
                
                plugins.append(plugin_instance)
                logger.info(f"✅ Harvested Tool Loaded: {tool_name}")
                print(f"✅ Harvested Tool Loaded: {tool_name}") # Visual feedback for CLI
                
            except Exception as e:
                logger.error(f"❌ Failed to load tool {tool_name}: {e}")

    return plugins