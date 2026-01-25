"""Bridge: Safe OS Interaction for SENSE"""

from typing import List, Optional, Dict
import subprocess
import logging

COMMAND_WHITELIST = ["ls", "cat", "echo", "pwd", "python", "pip", "grep", "mkdir", "touch", "rmdir"]

class EmergencyStop:
    """
    Singleton for emergency stop.
    """
    _stopped = False

    @classmethod
    def check(cls) -> bool:
        return cls._stopped

    @classmethod
    def stop(cls):
        cls._stopped = True
        logging.warning("Emergency stop activated")

    @classmethod
    def reset(cls):
        cls._stopped = False

class Bridge:
    """
    Safe OS interaction layer.
    """
    def __init__(self, whitelist: Optional[List[str]] = None):
        self.whitelist = whitelist or COMMAND_WHITELIST
        self.logger = logging.getLogger("Bridge")

    def execute(self, command: str, timeout: int = 30) -> Dict[str, str]:
        """
        Execute whitelisted command safely.
        """
        if EmergencyStop.check():
            raise Exception("Emergency stop activated")
        cmd_parts = command.split()
        if cmd_parts[0] not in self.whitelist:
            raise ValueError(f"Command {cmd_parts[0]} not whitelisted")
        try:
            result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=timeout)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": str(result.returncode)
            }
        except Exception as e:
            self.logger.error(f"Bridge execute failed: {e}")
            return {"error": str(e)}

# Example usage in Worker
# bridge = Bridge()
# result = bridge.execute("ls -la")
# if result["returncode"] == "0":
#     print(result["stdout"])
# else:
#     print(result["stderr"])