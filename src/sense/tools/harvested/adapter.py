import os
import sys
import subprocess
import logging
from typing import Any, Dict, AsyncIterator, Optional
from sense.core.plugins.interface import PluginABC, SensorReading

class AgentZeroToolAdapter(PluginABC):
    """
    Wraps an Agent Zero 'Instrument' using strict CLI isolation.
    Runs all tools in the public Android Download workspace.
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
            "version": "1.4.0",
            "capability": "actuator",
            "description": f"Universal Workspace Agent Zero Tool: {self.name}",
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
        Executes the tool via Subprocess (CLI) in the User's Download Workspace.
        """
        # Extract argument (supports 'arg', 'url', 'input', 'query')
        arg = kwargs.get('arg') or kwargs.get('url') or kwargs.get('input') or kwargs.get('query')
        if not arg and args:
            arg = args[0]
            
        if not arg:
            return "Error: No input argument provided to tool."

        self.logger.info(f"Executing {self.name} via CLI with arg: {arg}")

        # 1. RESOLVE SCRIPT (Universal Search)
        script_path = os.path.join(self.tool_path, f"{self.name}.py")
        if not os.path.exists(script_path):
            possible_files = [f for f in os.listdir(self.tool_path) if f.endswith('.py') and f != "__init__.py"]
            if possible_files:
                script_path = os.path.join(self.tool_path, possible_files[0])
            else:
                return f"Error: Could not find python script for {self.name}"

        # 2. CONVERT TO ABSOLUTE PATH
        # Ensures script is found even if we change the cwd below.
        script_path = os.path.abspath(script_path)

        # 3. DEFINE WORKSPACE
        # All tools run inside the public Download folder for instant access.
        workspace = "/sdcard/Download"
        
        # Ensure workspace exists
        if not os.path.exists(workspace):
            try:
                os.makedirs(workspace, exist_ok=True)
            except Exception as e:
                # Fallback to current dir if permission fails
                self.logger.warning(f"Could not write to {workspace}: {e}")
                workspace = None

        # 4. EXECUTE
        cmd = [sys.executable, script_path, str(arg)]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120, # Generous timeout for media tools
                cwd=workspace # <--- Run *inside* the download folder
            )
            
            output = result.stdout + "\n" + result.stderr
            
            # 5. CLEANUP NOISE
            clean_lines = []
            for line in output.split('\n'):
                # Heuristics to remove progress bars
                if "%" in line and ("ETA" in line or "at" in line): continue
                if line.strip() == "": continue
                clean_lines.append(line)
            
            clean_output = "\n".join(clean_lines)
            
            if result.returncode == 0:
                return clean_output.strip() or "Tool executed (No Output)."
            else:
                return f"Tool Error: {result.stderr}\nOutput: {clean_output}"

        except subprocess.TimeoutExpired:
            return "Error: Tool execution timed out (120s)."
        except Exception as e:
            return f"Execution Exception: {e}"
