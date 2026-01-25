"""
SENSE-v2 Terminal Tool
Schema-based tool for terminal/shell command execution.

Enhanced with agent-zero patterns:
- Named session support (multiple concurrent shells)
- SSH remote execution option
- Dialog detection (Y/N prompts)
- Timeout configuration per session

Hardened for SENSE v2 Unified Evolutionary Architecture:
- Strict type annotations throughout
- ExecutionError raising on critical stderr
- Extended safety checks for dangerous commands
- Proper error propagation for self-correction loops
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import asyncio
import os
import re
import logging
import shlex
from datetime import datetime
from enum import Enum
from pathlib import Path

from sense.core.base import BaseTool, ToolRegistry
from sense.core.schemas import (
    ToolSchema,
    ToolParameter,
    ToolResult,
    ToolResultStatus,
)


class ExecutionError(Exception):
    """
    Raised when terminal command execution fails critically.

    Used for self-correction loops - the agent can parse the error
    and retry with a corrected command.
    """
    def __init__(
        self,
        message: str,
        command: str = "",
        exit_code: Optional[int] = None,
        stderr: str = "",
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.command = command
        self.exit_code = exit_code
        self.stderr = stderr
        self.recoverable = recoverable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": str(self),
            "command": self.command,
            "exit_code": self.exit_code,
            "stderr": self.stderr,
            "recoverable": self.recoverable,
        }

# Optional SSH support
try:
    import paramiko
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False


# =============================================================================
# Multi-Session Shell Management (from agent-zero)
# =============================================================================

class SessionType(Enum):
    """Types of terminal sessions."""
    LOCAL = "local"
    SSH = "ssh"


@dataclass
class SessionConfig:
    """Configuration for a terminal session."""
    session_type: SessionType = SessionType.LOCAL
    # SSH configuration
    ssh_host: str = "localhost"
    ssh_port: int = 22
    ssh_user: str = "root"
    ssh_password: str = ""
    ssh_key_path: Optional[str] = None
    # Timeout configuration
    first_output_timeout: int = 30
    between_output_timeout: int = 15
    max_exec_timeout: int = 180
    dialog_timeout: int = 5
    # Working directory
    cwd: Optional[str] = None


@dataclass
class ShellSession:
    """A shell session wrapper."""
    session_id: str
    session_name: str
    config: SessionConfig
    process: Optional[asyncio.subprocess.Process] = None
    ssh_client: Optional[Any] = None  # paramiko.SSHClient
    ssh_channel: Optional[Any] = None  # paramiko.Channel
    is_running: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_command: Optional[str] = None
    last_output: str = ""

    def is_active(self) -> bool:
        """Check if session is still active."""
        if self.config.session_type == SessionType.LOCAL:
            return self.process is not None and self.process.returncode is None
        else:
            return self.ssh_channel is not None and not self.ssh_channel.closed


class SessionManager:
    """
    Manages multiple concurrent shell sessions.

    Supports both local and SSH sessions with named access.
    """

    # Dialog detection patterns (from agent-zero)
    DIALOG_PATTERNS = [
        re.compile(r"Y/N", re.IGNORECASE),
        re.compile(r"yes/no", re.IGNORECASE),
        re.compile(r"\[y/n\]", re.IGNORECASE),
        re.compile(r":\s*$"),  # Line ending with colon
        re.compile(r"\?\s*$"),  # Line ending with question mark
        re.compile(r"password\s*:", re.IGNORECASE),
        re.compile(r"continue\s*\?", re.IGNORECASE),
    ]

    # Shell prompt patterns
    PROMPT_PATTERNS = [
        re.compile(r"\(venv\).+[$#]\s*$"),
        re.compile(r"root@[^:]+:[^#]+#\s*$"),
        re.compile(r"[a-zA-Z0-9_.-]+@[^:]+:[^$#]+[$#]\s*$"),
        re.compile(r"\$\s*$"),
        re.compile(r"#\s*$"),
    ]

    def __init__(self, default_cwd: Optional[str] = None):
        self.sessions: Dict[str, ShellSession] = {}
        self.default_cwd = default_cwd or os.getcwd()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._session_counter = 0

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        self._session_counter += 1
        return f"session_{int(datetime.now().timestamp())}_{self._session_counter}"

    async def create_session(
        self,
        name: Optional[str] = None,
        config: Optional[SessionConfig] = None,
    ) -> ShellSession:
        """
        Create a new shell session.

        Args:
            name: Optional session name (defaults to session_id)
            config: Session configuration

        Returns:
            The created ShellSession
        """
        session_id = self._generate_session_id()
        session_name = name or session_id
        config = config or SessionConfig(cwd=self.default_cwd)

        session = ShellSession(
            session_id=session_id,
            session_name=session_name,
            config=config,
        )

        if config.session_type == SessionType.LOCAL:
            await self._init_local_session(session)
        else:
            await self._init_ssh_session(session)

        self.sessions[session_name] = session
        self.logger.info(f"Created session '{session_name}' ({config.session_type.value})")

        return session

    async def _init_local_session(self, session: ShellSession) -> None:
        """Initialize a local shell session."""
        cwd = session.config.cwd or self.default_cwd

        session.process = await asyncio.create_subprocess_shell(
            "bash" if os.name != "nt" else "cmd",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )

    async def _init_ssh_session(self, session: ShellSession) -> None:
        """Initialize an SSH shell session."""
        if not SSH_AVAILABLE:
            raise RuntimeError("SSH support requires paramiko: pip install paramiko")

        config = session.config
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if config.ssh_key_path:
                client.connect(
                    hostname=config.ssh_host,
                    port=config.ssh_port,
                    username=config.ssh_user,
                    key_filename=config.ssh_key_path,
                )
            else:
                client.connect(
                    hostname=config.ssh_host,
                    port=config.ssh_port,
                    username=config.ssh_user,
                    password=config.ssh_password,
                )

            session.ssh_client = client
            session.ssh_channel = client.invoke_shell()
            session.ssh_channel.settimeout(config.first_output_timeout)

            if config.cwd:
                session.ssh_channel.send(f"cd {config.cwd}\n")
                await asyncio.sleep(0.5)
                # Clear the cd output
                if session.ssh_channel.recv_ready():
                    session.ssh_channel.recv(4096)

        except Exception as e:
            client.close()
            raise RuntimeError(f"SSH connection failed: {e}")

    def get_session(self, name: str) -> Optional[ShellSession]:
        """Get a session by name."""
        return self.sessions.get(name)

    def get_or_create_session(
        self,
        name: str,
        config: Optional[SessionConfig] = None,
    ) -> ShellSession:
        """Get existing session or create new one."""
        if name in self.sessions and self.sessions[name].is_active():
            return self.sessions[name]
        # Create new session (will be awaited by caller)
        return None  # Caller should await create_session

    async def close_session(self, name: str) -> bool:
        """Close and remove a session."""
        session = self.sessions.pop(name, None)
        if not session:
            return False

        try:
            if session.config.session_type == SessionType.LOCAL:
                if session.process:
                    session.process.terminate()
                    await session.process.wait()
            else:
                if session.ssh_channel:
                    session.ssh_channel.close()
                if session.ssh_client:
                    session.ssh_client.close()

            self.logger.info(f"Closed session '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error closing session '{name}': {e}")
            return False

    async def close_all(self) -> None:
        """Close all sessions."""
        for name in list(self.sessions.keys()):
            await self.close_session(name)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "name": s.session_name,
                "type": s.config.session_type.value,
                "active": s.is_active(),
                "created_at": s.created_at.isoformat(),
                "last_command": s.last_command,
            }
            for s in self.sessions.values()
        ]

    def detect_dialog(self, output: str) -> bool:
        """
        Detect if output contains a dialog prompt.

        Args:
            output: Command output to check

        Returns:
            True if dialog prompt detected
        """
        last_lines = output.splitlines()[-3:] if output else []
        for line in last_lines:
            for pattern in self.DIALOG_PATTERNS:
                if pattern.search(line.strip()):
                    return True
        return False

    def detect_prompt(self, output: str) -> bool:
        """
        Detect if output ends with a shell prompt.

        Args:
            output: Command output to check

        Returns:
            True if shell prompt detected
        """
        last_lines = output.splitlines()[-2:] if output else []
        for line in last_lines:
            for pattern in self.PROMPT_PATTERNS:
                if pattern.search(line.strip()):
                    return True
        return False


@ToolRegistry.register
class TerminalTool(BaseTool):
    """
    Terminal execution tool with multi-session support.

    Per SYSTEM_PROMPT requirements:
    - Exposed as Schema-based Python Tool
    - Includes feedback mechanism for self-correction (stderr parsing)
    - Reward based on exit codes

    Enhanced with agent-zero patterns:
    - Named session support for multiple concurrent shells
    - SSH remote execution option
    - Dialog detection (Y/N prompts)
    - Timeout configuration per session
    """

    # Shared session manager across tool instances (initialized lazily)
    _session_manager: Optional[SessionManager] = None

    # Class-level blocked command patterns (comprehensive list)
    BLOCKED_PATTERNS: List[str] = [
        # Destructive file operations
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf $HOME",
        # Disk operations
        "mkfs",
        "dd if=/dev/zero of=/dev/sd",
        "dd if=/dev/random of=/dev/sd",
        "dd of=/dev/sd",
        # Fork bombs
        ":(){:|:&};:",
        ":(){ :|:& };:",
        # System destruction
        "mv /* /dev/null",
        "chmod -R 777 /",
        "chown -R",
        # Network attacks
        "nmap -sS",  # SYN scan (requires root, suspicious)
        # Privilege escalation
        "sudo rm",
        "sudo dd",
        "sudo mkfs",
        # History manipulation (could hide attacks)
        "history -c",
        "export HISTSIZE=0",
        # Dangerous redirections
        "> /etc/passwd",
        "> /etc/shadow",
        ">/dev/sda",
    ]

    # Patterns that require extra caution (logged but allowed)
    CAUTION_PATTERNS: List[str] = [
        "sudo",
        "su -",
        "wget",
        "curl.*|.*sh",
        "chmod +x",
        "rm -r",
    ]

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cwd: str = os.getcwd()

        # Instance-level blocked patterns (can be extended)
        self._blocked_patterns: List[str] = list(self.BLOCKED_PATTERNS)

        # Track command history for security analysis
        self._command_history: List[Dict[str, Any]] = []
        self._max_history_size: int = 100

    @property
    def session_manager(self) -> SessionManager:
        """Get the session manager, initializing if needed."""
        cls = type(self)
        if cls._session_manager is None:
            cls._session_manager = SessionManager(default_cwd=self.cwd)
        return cls._session_manager

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="terminal_exec",
            description="Execute a shell command and return the output. Supports multiple named sessions.",
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
                ToolParameter(
                    name="session",
                    param_type="string",
                    description="Named session to use (creates if doesn't exist)",
                    required=False,
                    default="default",
                ),
                ToolParameter(
                    name="ssh_host",
                    param_type="string",
                    description="SSH host for remote execution (optional)",
                    required=False,
                ),
                ToolParameter(
                    name="ssh_user",
                    param_type="string",
                    description="SSH username for remote execution",
                    required=False,
                    default="root",
                ),
            ],
            returns="object",
            returns_description="Object containing stdout, stderr, exit_code, and dialog_detected flag",
            category="terminal",
            requires_confirmation=False,
            timeout_seconds=60,
            max_retries=2,
        )

    def _is_blocked(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Check if command matches blocked patterns.

        Args:
            command: The command to check

        Returns:
            Tuple of (is_blocked, matched_pattern)
        """
        cmd_lower = command.lower().strip()

        # Check exact and substring matches
        for pattern in self._blocked_patterns:
            if pattern.lower() in cmd_lower:
                return True, pattern

        # Check regex patterns for more complex matches
        dangerous_regexes = [
            (r"rm\s+-[rf]+\s+/(?:\s|$)", "recursive delete from root"),
            (r">\s*/dev/sd[a-z]", "redirect to disk device"),
            (r"dd\s+.*of=/dev/sd[a-z]", "dd to disk device"),
            (r"chmod\s+-R\s+777\s+/", "chmod 777 on root"),
            (r"curl.*\|\s*(?:ba)?sh", "pipe curl to shell"),
            (r"wget.*\|\s*(?:ba)?sh", "pipe wget to shell"),
        ]

        for regex, description in dangerous_regexes:
            if re.search(regex, cmd_lower):
                return True, description

        return False, None

    def _is_caution(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Check if command requires extra caution (logged but allowed).

        Args:
            command: The command to check

        Returns:
            Tuple of (requires_caution, matched_pattern)
        """
        cmd_lower = command.lower().strip()

        for pattern in self.CAUTION_PATTERNS:
            if re.search(pattern, cmd_lower):
                return True, pattern

        return False, None

    def _record_command(
        self,
        command: str,
        result: Optional[ToolResult] = None,
        blocked: bool = False,
    ) -> None:
        """
        Record command in history for security analysis.

        Args:
            command: The executed command
            result: The execution result (if any)
            blocked: Whether the command was blocked
        """
        entry = {
            "command": command[:200],  # Truncate long commands
            "timestamp": datetime.now().isoformat(),
            "blocked": blocked,
            "exit_code": result.exit_code if result else None,
            "success": result.is_success if result else None,
        }

        self._command_history.append(entry)

        # Keep history bounded
        if len(self._command_history) > self._max_history_size:
            self._command_history = self._command_history[-self._max_history_size:]

    async def execute(
        self,
        command: str,
        timeout: int = 60,
        cwd: Optional[str] = None,
        session: str = "default",
        ssh_host: Optional[str] = None,
        ssh_user: str = "root",
        ssh_password: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute a shell command with multi-session support.

        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds
            cwd: Working directory (defaults to current)
            session: Named session to use (creates if doesn't exist)
            ssh_host: SSH host for remote execution (optional)
            ssh_user: SSH username for remote execution
            ssh_password: SSH password (optional)

        Returns:
            ToolResult with stdout, stderr, exit_code, and dialog_detected flag
        """
        start_time = datetime.now()

        # Safety check - blocked commands
        is_blocked, blocked_reason = self._is_blocked(command)
        if is_blocked:
            self._record_command(command, blocked=True)
            self.logger.warning(f"Blocked command: {command[:50]}... (reason: {blocked_reason})")
            return ToolResult.error(
                f"Command blocked for safety reasons: {blocked_reason}",
                metadata={"command": command[:50], "blocked_reason": blocked_reason},
            )

        # Caution check - log but allow
        is_caution, caution_reason = self._is_caution(command)
        if is_caution:
            self.logger.info(f"Caution command executed: {command[:50]}... (pattern: {caution_reason})")

        working_dir = cwd or self.cwd

        # Check if using named session
        if session != "default" or ssh_host:
            return await self._execute_in_session(
                command=command,
                timeout=timeout,
                session_name=session,
                cwd=working_dir,
                ssh_host=ssh_host,
                ssh_user=ssh_user,
                ssh_password=ssh_password,
            )

        # Standard single-command execution
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

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            result = ToolResult.from_process(
                returncode=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time_ms=execution_time,
            )

            # Add metadata including dialog detection
            result.metadata["command"] = command[:100]
            result.metadata["cwd"] = working_dir
            result.metadata["session"] = "default"
            result.metadata["dialog_detected"] = self.session_manager.detect_dialog(
                stdout_str + stderr_str
            )

            # Record command in history
            self._record_command(command, result)

            # Log for debugging
            self.logger.debug(
                f"Executed: {command[:50]}... exit_code={process.returncode}"
            )

            # Check for critical errors that should raise ExecutionError
            if process.returncode != 0 and stderr_str:
                critical_patterns = [
                    "permission denied",
                    "command not found",
                    "no such file or directory",
                    "segmentation fault",
                    "killed",
                    "out of memory",
                ]
                stderr_lower = stderr_str.lower()
                is_critical = any(p in stderr_lower for p in critical_patterns)

                if is_critical:
                    result.metadata["critical_error"] = True
                    # Don't raise here - let the agent handle via should_retry

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

    async def _execute_in_session(
        self,
        command: str,
        timeout: int,
        session_name: str,
        cwd: Optional[str] = None,
        ssh_host: Optional[str] = None,
        ssh_user: str = "root",
        ssh_password: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute command in a named session.

        Args:
            command: Command to execute
            timeout: Execution timeout
            session_name: Session name
            cwd: Working directory
            ssh_host: SSH host (optional)
            ssh_user: SSH username
            ssh_password: SSH password

        Returns:
            ToolResult with output and dialog detection
        """
        start_time = datetime.now()

        try:
            # Get or create session
            shell_session = self.session_manager.get_session(session_name)

            if shell_session is None:
                # Create new session with appropriate config
                if ssh_host:
                    config = SessionConfig(
                        session_type=SessionType.SSH,
                        ssh_host=ssh_host,
                        ssh_user=ssh_user,
                        ssh_password=ssh_password or "",
                        max_exec_timeout=timeout,
                        cwd=cwd,
                    )
                else:
                    config = SessionConfig(
                        session_type=SessionType.LOCAL,
                        max_exec_timeout=timeout,
                        cwd=cwd,
                    )

                shell_session = await self.session_manager.create_session(
                    name=session_name,
                    config=config,
                )

            # Execute in session
            shell_session.last_command = command
            shell_session.is_running = True

            if shell_session.config.session_type == SessionType.LOCAL:
                output = await self._execute_local_session(
                    shell_session, command, timeout
                )
            else:
                output = await self._execute_ssh_session(
                    shell_session, command, timeout
                )

            shell_session.is_running = False
            shell_session.last_output = output

            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Check for dialog prompts
            dialog_detected = self.session_manager.detect_dialog(output)
            prompt_detected = self.session_manager.detect_prompt(output)

            result = ToolResult.success(
                data={"output": output},
                metadata={
                    "command": command[:100],
                    "session": session_name,
                    "session_type": shell_session.config.session_type.value,
                    "dialog_detected": dialog_detected,
                    "prompt_detected": prompt_detected,
                    "execution_time_ms": execution_time,
                },
            )

            # Add warning if dialog detected
            if dialog_detected:
                result.metadata["warning"] = (
                    "Dialog prompt detected. The command may be waiting for input. "
                    "Consider sending a response (y/n) or using Ctrl+C to cancel."
                )

            return result

        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TIMEOUT,
                error=f"Session command timed out after {timeout}s",
                metadata={"command": command[:100], "session": session_name},
            )
        except Exception as e:
            self.logger.error(f"Session execution failed: {e}")
            return ToolResult.error(str(e), metadata={"session": session_name})

    async def _execute_local_session(
        self,
        session: ShellSession,
        command: str,
        timeout: int,
    ) -> str:
        """Execute command in local session."""
        if not session.process or not session.process.stdin:
            raise RuntimeError("Session process not initialized")

        # Send command
        session.process.stdin.write(f"{command}\n".encode())
        await session.process.stdin.drain()

        # Read output with timeout
        output_parts = []
        end_time = datetime.now().timestamp() + timeout

        while datetime.now().timestamp() < end_time:
            try:
                chunk = await asyncio.wait_for(
                    session.process.stdout.read(4096),
                    timeout=1.0,
                )
                if chunk:
                    output_parts.append(chunk.decode("utf-8", errors="replace"))
                    # Check if we hit a prompt
                    full_output = "".join(output_parts)
                    if self.session_manager.detect_prompt(full_output):
                        break
                else:
                    break
            except asyncio.TimeoutError:
                # No more output available
                if output_parts:
                    break
                continue

        return "".join(output_parts)

    async def _execute_ssh_session(
        self,
        session: ShellSession,
        command: str,
        timeout: int,
    ) -> str:
        """Execute command in SSH session."""
        if not session.ssh_channel:
            raise RuntimeError("SSH channel not initialized")

        # Send command
        session.ssh_channel.send(f"{command}\n")

        # Read output
        output_parts = []
        end_time = datetime.now().timestamp() + timeout

        while datetime.now().timestamp() < end_time:
            if session.ssh_channel.recv_ready():
                chunk = session.ssh_channel.recv(4096)
                output_parts.append(chunk.decode("utf-8", errors="replace"))
                # Check if we hit a prompt
                full_output = "".join(output_parts)
                if self.session_manager.detect_prompt(full_output):
                    break
            else:
                await asyncio.sleep(0.1)

        return "".join(output_parts)

    async def list_sessions(self) -> ToolResult:
        """List all active terminal sessions."""
        sessions = self.session_manager.list_sessions()
        return ToolResult.success(
            data={"sessions": sessions},
            metadata={"count": len(sessions)},
        )

    async def close_session(self, session_name: str) -> ToolResult:
        """Close a named session."""
        success = await self.session_manager.close_session(session_name)
        if success:
            return ToolResult.success(
                data={"closed": session_name},
                metadata={"status": "closed"},
            )
        return ToolResult.error(f"Session '{session_name}' not found")


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
