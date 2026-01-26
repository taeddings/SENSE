"""
Wrapper Generator.

Auto-generates PluginABC wrappers for external libraries.
"""

from typing import Dict, Any, List
from .discovery_engine import DiscoveredTool

class WrapperGenerator:
    """
    Generates Python code for a SENSE Plugin wrapper.
    """

    WRAPPER_TEMPLATE = '''"""
Auto-generated wrapper for {name}
Source: {source}
"""
from typing import Any, Dict
from sense.core.plugins.interface import PluginABC
import logging

try:
    import {module_name}
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

class {class_name}(PluginABC):
    """Wrapper for {name}: {description}"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.logger = logging.getLogger("{class_name}")
        if not AVAILABLE:
            self.logger.warning("{name} not installed. Run: {install_command}")

    def execute(self, func_name: str, *args, **kwargs) -> Any:
        """Execute a function from the library."""
        if not AVAILABLE:
            raise ImportError("{name} is not installed.")
            
        if hasattr({module_name}, func_name):
            func = getattr({module_name}, func_name)
            return func(*args, **kwargs)
        raise AttributeError(f"Function {{func_name}} not found in {module_name}")

    def get_manifest(self) -> Dict[str, Any]:
        return {{
            "name": "{name}",
            "type": "external",
            "source": "{source}",
            "installed": AVAILABLE
        }}
'''

    def generate_wrapper(self, tool: DiscoveredTool) -> str:
        """Generate the wrapper code string."""
        # Simple heuristic for module name (e.g., opencv-python -> cv2)
        module_name = tool.name.lower().replace('-', '_')
        if tool.name == "opencv-python": module_name = "cv2"
        if tool.name == "Pillow": module_name = "PIL"
        
        class_name = tool.name.replace('-', '').replace('_', '').capitalize() + "Plugin"
        
        return self.WRAPPER_TEMPLATE.format(
            name=tool.name,
            source=tool.source,
            module_name=module_name,
            class_name=class_name,
            description=tool.description,
            install_command=tool.install_command
        )
