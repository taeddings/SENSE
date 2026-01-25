"""
SENSE-v2 Sub-Agents for Agent Zero
Specialized agents for Terminal, FileSystem, and Browser operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import asyncio
import subprocess
import os
from pathlib import Path
from datetime import datetime

from sense_v2.core.base import BaseAgent, BaseTool
from sense_v2.core.schemas import AgentMessage, MessageRole, ToolResult, ToolResultStatus
from sense_v2.core.config import OrchestrationConfig


class TerminalAgent(BaseAgent):
    """
    Terminal Sub-Agent for executing shell commands.

    Per SYSTEM_PROMPT requirements:
    - Implements self-correction loop via stderr parsing
    - Handles command execution with proper error handling
    - Supports timeout and retry mechanisms
    """

    def __init__(self, config: Optional[OrchestrationConfig] = None):
        super().__init__(name="TerminalAgent", config=config)
        self.config = config or OrchestrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Command history for context
        self.command_history: List[Dict[str, Any]] = []
        self.max_history = 50

        # Working directory
        self.cwd = os.getcwd()

    async def execute_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        shell: bool = True,
    ) -> ToolResult:
        """
        Execute a shell command with proper error handling.

        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds
            shell: Use shell execution

        Returns:
            ToolResult with stdout/stderr
        """
        timeout = timeout or self.config.task_timeout_seconds
        start_time = datetime.now()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
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

            # Store in history
            self._record_command(command, result)

            return result

        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TIMEOUT,
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult.error(str(e))

    def _record_command(self, command: str, result: ToolResult) -> None:
        """Record command in history."""
        self.command_history.append({
            "command": command,
            "success": result.is_success,
            "exit_code": result.exit_code,
            "timestamp": datetime.now().isoformat(),
        })

        # Prune history
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]

    def analyze_error(self, stderr: str) -> Optional[str]:
        """
        Analyze stderr to suggest corrections.
        Implements self-correction loop per SYSTEM_PROMPT.
        """
        stderr_lower = stderr.lower()

        error_patterns = {
            "command not found": "Ensure the command is installed or check PATH",
            "permission denied": "Try with sudo or check file permissions",
            "no such file or directory": "Verify the path exists",
            "syntax error": "Check command syntax",
            "connection refused": "Verify the service is running",
            "disk quota exceeded": "Free up disk space",
            "too many open files": "Close some files or increase ulimit",
        }

        for pattern, suggestion in error_patterns.items():
            if pattern in stderr_lower:
                return suggestion

        return None

    def set_cwd(self, path: str) -> bool:
        """Set working directory."""
        if os.path.isdir(path):
            self.cwd = path
            return True
        return False

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process message as command execution request."""
        result = await self.execute_command(message.content)

        if result.is_success:
            return AgentMessage.assistant(result.stdout or "Command completed successfully")
        else:
            suggestion = self.analyze_error(result.stderr or "")
            error_msg = result.error or result.stderr or "Command failed"
            if suggestion:
                error_msg += f"\nSuggestion: {suggestion}"
            return AgentMessage.assistant(error_msg)

    async def run(self) -> None:
        """Main agent loop."""
        self._is_running = True
        self.logger.info("TerminalAgent started")
        while self._is_running:
            await asyncio.sleep(1)


class FileSystemAgent(BaseAgent):
    """
    FileSystem Sub-Agent for file operations.

    Handles:
    - Reading/writing files
    - Directory operations
    - File search and listing
    """

    def __init__(self, config: Optional[OrchestrationConfig] = None):
        super().__init__(name="FileSystemAgent", config=config)
        self.config = config or OrchestrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Operation history
        self.operation_history: List[Dict[str, Any]] = []

    async def read_file(self, path: str, encoding: str = "utf-8") -> ToolResult:
        """Read file contents."""
        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return ToolResult.error(f"File not found: {path}")

            if not file_path.is_file():
                return ToolResult.error(f"Not a file: {path}")

            content = file_path.read_text(encoding=encoding)

            self._record_operation("read", str(file_path), True)

            return ToolResult.success(
                content,
                metadata={"path": str(file_path), "size": len(content)},
            )

        except Exception as e:
            self._record_operation("read", path, False, str(e))
            return ToolResult.error(str(e))

    async def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> ToolResult:
        """Write content to file."""
        try:
            file_path = Path(path).expanduser().resolve()

            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            file_path.write_text(content, encoding=encoding)

            self._record_operation("write", str(file_path), True)

            return ToolResult.success(
                f"Written {len(content)} bytes to {file_path}",
                metadata={"path": str(file_path), "size": len(content)},
            )

        except Exception as e:
            self._record_operation("write", path, False, str(e))
            return ToolResult.error(str(e))

    async def list_directory(
        self,
        path: str,
        pattern: str = "*",
        recursive: bool = False,
    ) -> ToolResult:
        """List directory contents."""
        try:
            dir_path = Path(path).expanduser().resolve()

            if not dir_path.exists():
                return ToolResult.error(f"Directory not found: {path}")

            if not dir_path.is_dir():
                return ToolResult.error(f"Not a directory: {path}")

            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))

            file_info = []
            for f in files[:100]:  # Limit results
                try:
                    stat = f.stat()
                    file_info.append({
                        "name": f.name,
                        "path": str(f),
                        "is_dir": f.is_dir(),
                        "size": stat.st_size if f.is_file() else 0,
                    })
                except Exception:
                    continue

            self._record_operation("list", str(dir_path), True)

            return ToolResult.success(
                file_info,
                metadata={"path": str(dir_path), "count": len(file_info)},
            )

        except Exception as e:
            self._record_operation("list", path, False, str(e))
            return ToolResult.error(str(e))

    async def file_exists(self, path: str) -> ToolResult:
        """Check if file or directory exists."""
        try:
            file_path = Path(path).expanduser().resolve()
            exists = file_path.exists()

            return ToolResult.success(
                exists,
                metadata={
                    "path": str(file_path),
                    "is_file": file_path.is_file() if exists else None,
                    "is_dir": file_path.is_dir() if exists else None,
                },
            )

        except Exception as e:
            return ToolResult.error(str(e))

    async def delete(self, path: str, recursive: bool = False) -> ToolResult:
        """Delete file or directory."""
        try:
            target = Path(path).expanduser().resolve()

            if not target.exists():
                return ToolResult.error(f"Path not found: {path}")

            if target.is_file():
                target.unlink()
            elif target.is_dir():
                if recursive:
                    import shutil
                    shutil.rmtree(target)
                else:
                    target.rmdir()

            self._record_operation("delete", str(target), True)

            return ToolResult.success(f"Deleted: {target}")

        except Exception as e:
            self._record_operation("delete", path, False, str(e))
            return ToolResult.error(str(e))

    def _record_operation(
        self,
        operation: str,
        path: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record file operation."""
        self.operation_history.append({
            "operation": operation,
            "path": path,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process message as file operation request."""
        content = message.content.lower()

        # Simple parsing of file operation requests
        if "read" in content:
            # Extract path (simplified)
            parts = message.content.split()
            if len(parts) >= 2:
                result = await self.read_file(parts[-1])
                return AgentMessage.assistant(str(result.output) if result.is_success else result.error)

        elif "list" in content:
            parts = message.content.split()
            path = parts[-1] if len(parts) >= 2 else "."
            result = await self.list_directory(path)
            return AgentMessage.assistant(str(result.output) if result.is_success else result.error)

        return AgentMessage.assistant("Unknown file operation")

    async def run(self) -> None:
        """Main agent loop."""
        self._is_running = True
        self.logger.info("FileSystemAgent started")
        while self._is_running:
            await asyncio.sleep(1)


class BrowserAgent(BaseAgent):
    """
    Browser Sub-Agent for web operations.

    Handles:
    - HTTP requests
    - Web page fetching
    - API calls
    """

    def __init__(self, config: Optional[OrchestrationConfig] = None):
        super().__init__(name="BrowserAgent", config=config)
        self.config = config or OrchestrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Request history
        self.request_history: List[Dict[str, Any]] = []

        # Session (lazy init)
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                self.logger.warning("aiohttp not available")
                return None
        return self._session

    async def fetch_url(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        timeout: int = 30,
    ) -> ToolResult:
        """Fetch URL with specified method."""
        start_time = datetime.now()

        try:
            session = await self._get_session()

            if session is None:
                # Fallback to urllib
                import urllib.request
                import urllib.error

                req = urllib.request.Request(url, headers=headers or {})
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        content = response.read().decode("utf-8", errors="replace")
                        status = response.status

                        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

                        self._record_request(url, method, status, True)

                        return ToolResult.success(
                            content,
                            metadata={"url": url, "status": status},
                            execution_time_ms=execution_time,
                        )
                except urllib.error.HTTPError as e:
                    self._record_request(url, method, e.code, False)
                    return ToolResult.error(f"HTTP {e.code}: {e.reason}")
                except urllib.error.URLError as e:
                    self._record_request(url, method, 0, False)
                    return ToolResult.error(str(e.reason))

            # Use aiohttp
            async with session.request(
                method,
                url,
                headers=headers,
                json=data if method in ["POST", "PUT", "PATCH"] else None,
                timeout=timeout,
            ) as response:
                content = await response.text()
                status = response.status

                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

                self._record_request(url, method, status, status < 400)

                if status >= 400:
                    return ToolResult.error(
                        f"HTTP {status}",
                        metadata={"url": url, "status": status},
                    )

                return ToolResult.success(
                    content,
                    metadata={"url": url, "status": status},
                    execution_time_ms=execution_time,
                )

        except asyncio.TimeoutError:
            self._record_request(url, method, 0, False, "timeout")
            return ToolResult(status=ToolResultStatus.TIMEOUT, error="Request timed out")
        except Exception as e:
            self._record_request(url, method, 0, False, str(e))
            return ToolResult.error(str(e))

    async def fetch_json(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
    ) -> ToolResult:
        """Fetch URL and parse as JSON."""
        headers = headers or {}
        headers["Accept"] = "application/json"

        result = await self.fetch_url(url, method, headers, data)

        if not result.is_success:
            return result

        try:
            import json
            parsed = json.loads(result.output)
            result.output = parsed
            return result
        except Exception as e:
            return ToolResult.error(f"JSON parse error: {e}")

    def _record_request(
        self,
        url: str,
        method: str,
        status: int,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record HTTP request."""
        self.request_history.append({
            "url": url,
            "method": method,
            "status": status,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process message as web request."""
        content = message.content

        # Extract URL from message
        import re
        url_match = re.search(r'https?://\S+', content)

        if url_match:
            url = url_match.group(0)
            result = await self.fetch_url(url)
            if result.is_success:
                # Truncate long responses
                output = str(result.output)[:2000]
                return AgentMessage.assistant(output)
            else:
                return AgentMessage.assistant(f"Request failed: {result.error}")

        return AgentMessage.assistant("No valid URL found in request")

    async def run(self) -> None:
        """Main agent loop."""
        self._is_running = True
        self.logger.info("BrowserAgent started")
        while self._is_running:
            await asyncio.sleep(1)

    def __del__(self):
        """Cleanup on deletion."""
        if self._session:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass
