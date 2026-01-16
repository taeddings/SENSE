"""
SENSE-v2 Terminal Tool
Schema-based tool for terminal/shell command execution.
"""

from typing import Any, Dict, List, Optional
import asyncio
import os
import logging
from datetime import datetime

from sense_v2.core.base import BaseTool, ToolRegistry
from sense_v2.core.schemas import (
    ToolSchema,
    ToolParameter,
    ToolResult,
    ToolResultStatus,
)


@ToolRegistry.register
class TerminalTool(BaseTool):
    """
    Terminal execution tool.

    Per SYSTEM_PROMPT requirements:
    - Exposed as Schema-based Python Tool
    - Includes feedback mechanism for self-correction (stderr parsing)
    - Reward based on exit codes
    """

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cwd = os.getcwd()

        # Blocked commands for safety
        self._blocked_patterns = [
            "rm -rf /",
            "mkfs",
            ":(){:|:&};:",  # Fork bomb
            "dd if=/dev/zero of=/dev/sda",
        ]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="terminal_exec",
            description="Execute a shell command and return the output",
            parameters=[
                ToolParameter(
                    name="command",
                    param_type="string",
                    description="The shell command to execute",
                    required=True,
                ),
                ToolParameter(
                    name="timeout",
                    param_type="integer",
                    description="Execution timeout in seconds",
                    required=False,
                    default=60,
                    min_value=1,
                    max_value=300,
                ),
                ToolParameter(
                    name="cwd",
                    param_type="string",
                    description="Working directory for command execution",
                    required=False,
                ),
            ],
            returns="object",
            returns_description="Object containing stdout, stderr, and exit_code",
            category="terminal",
            requires_confirmation=False,
            timeout_seconds=60,
            max_retries=2,
        )

    def _is_blocked(self, command: str) -> bool:
        """Check if command matches blocked patterns."""
        cmd_lower = command.lower()
        return any(pattern in cmd_lower for pattern in self._blocked_patterns)

    async def execute(
        self,
        command: str,
        timeout: int = 60,
        cwd: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute a shell command.

        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds
            cwd: Working directory (defaults to current)

        Returns:
            ToolResult with stdout, stderr, exit_code
        """
        start_time = datetime.now()

        # Safety check
        if self._is_blocked(command):
            return ToolResult.error(
                "Command blocked for safety reasons",
                metadata={"command": command[:50]},
            )

        working_dir = cwd or self.cwd

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            result = ToolResult.from_process(
                returncode=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                execution_time_ms=execution_time,
            )

            # Add metadata
            result.metadata["command"] = command[:100]
            result.metadata["cwd"] = working_dir

            # Log for debugging
            self.logger.debug(
                f"Executed: {command[:50]}... exit_code={process.returncode}"
            )

            return result

        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TIMEOUT,
                error=f"Command timed out after {timeout}s",
                metadata={"command": command[:100]},
            )
        except FileNotFoundError:
            return ToolResult.error(
                f"Working directory not found: {working_dir}",
            )
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return ToolResult.error(str(e))


@ToolRegistry.register
class TerminalInteractiveTool(BaseTool):
    """
    Tool for interactive terminal sessions (experimental).
    """

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self._sessions: Dict[str, Any] = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="terminal_interactive",
            description="Start or interact with a terminal session",
            parameters=[
                ToolParameter(
                    name="action",
                    param_type="string",
                    description="Action: start, send, read, close",
                    required=True,
                    enum=["start", "send", "read", "close"],
                ),
                ToolParameter(
                    name="session_id",
                    param_type="string",
                    description="Session identifier",
                    required=False,
                ),
                ToolParameter(
                    name="input",
                    param_type="string",
                    description="Input to send to session",
                    required=False,
                ),
            ],
            category="terminal",
            requires_confirmation=True,
            max_retries=1,
        )

    async def execute(
        self,
        action: str,
        session_id: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute interactive terminal action."""
        if action == "start":
            new_id = f"session_{int(datetime.now().timestamp())}"
            # Placeholder for actual PTY implementation
            self._sessions[new_id] = {"status": "active"}
            return ToolResult.success({"session_id": new_id, "status": "started"})

        elif action == "close" and session_id:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return ToolResult.success({"session_id": session_id, "status": "closed"})
            return ToolResult.error(f"Session not found: {session_id}")

        return ToolResult.error(f"Invalid action or missing parameters")
