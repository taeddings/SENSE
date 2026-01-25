import os
import sys
import subprocess
import logging
from typing import Any, Dict, AsyncIterator, Optional
from sense.core.plugins.interface import PluginABC, SensorReading

class AgentZeroToolAdapter(PluginABC):
    """
    Wraps an Agent Zero 'Instrument' using strict CLI isolation.
    Does NOT import the python file to prevent side-effects (like sys.exit).
    """
    def __init__(self, tool_path: str):
        super().__init__()
        self.tool_path = tool_path
        self.name = os.path.basename(tool_path)
        self.logger = logging.getLogger(f"Adapter.{self.name}")
        # We DO NOT import the module. It is too dangerous.
        self.module = None 

    def get_manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": "1.2.0",
            "capability": "actuator",
            "description": f"Sandboxed Agent Zero Tool: {self.name}",
            "sensors": [],
            "actuators": [self.name]
        }

    async def stream_auxiliary_input(self) -> AsyncIterator[SensorReading]:
        if False: yield None

    def emergency_stop(self) -> None:
        pass

    def safety_policy(self) -> Dict[str, Any]:
        return {}

    def get_grounding_truth(self, query: str) -> Optional[float]:
        return None

    def execute(self, *args, **kwargs):
        """
        Executes the tool via Subprocess (CLI) only.
        This prevents the tool from crashing the main agent process.
        """
        # Extract argument (supports 'arg', 'url', 'input', 'query')
        arg = kwargs.get('arg') or kwargs.get('url') or kwargs.get('input') or kwargs.get('query')
        if not arg and args:
            arg = args[0]
            
        if not arg:
            return "Error: No input argument provided to tool."

        self.logger.info(f"Executing {self.name} via CLI with arg: {arg}")

        # Construct Command
        script_path = os.path.join(self.tool_path, f"{self.name}.py")
        
        # Verify file exists
        if not os.path.exists(script_path):
             # Search for the main .py file if name doesn't match directory
            possible_files = [f for f in os.listdir(self.tool_path) if f.endswith('.py') and f != "__init__.py"]
            if possible_files:
                script_path = os.path.join(self.tool_path, possible_files[0])
            else:
                return f"Error: Could not find python script for {self.name}"

        cmd = [sys.executable, script_path, str(arg)]
        
        try:
            # CAPTURE OUTPUT
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            output = result.stdout + "\n" + result.stderr
            
            # --- CLEANUP HEURISTICS ---
            # Filter out progress bars and noise
            clean_lines = []
            for line in output.split('\n'):
                # Strip typical download bars
                if "[download]" in line and "%" in line:
                    continue
                if line.strip() == "":
                    continue
                clean_lines.append(line)
            
            clean_output = "\n".join(clean_lines)
            
            if result.returncode == 0:
                return clean_output.strip() or "Tool executed (No Output)."
            else:
                return f"Tool Error: {result.stderr}\nOutput: {clean_output}"

        except subprocess.TimeoutExpired:
            return "Error: Tool execution timed out (60s)."
        except Exception as e:
            return f"Execution Exception: {e}"
