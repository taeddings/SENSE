"""
SENSE Bridge Termux Driver
System driver implementation for Android/Termux environment.

Part of Phase 1: Milestone 1.3 - Bridge & OS Control

Termux-specific features:
- pkg package manager
- Android-specific paths
- Termux API integration (optional)
- Storage permission handling
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..interface import (
    SystemDriverABC,
    CommandResult,
    ExecutionStatus,
    FileInfo,
    EmergencyStop,
)
from ..safety import check_command_safety, check_path_safety


class TermuxDriver(SystemDriverABC):
    """
    System driver for Android/Termux environment.

    Provides Termux-specific implementations for:
    - Command execution via subprocess
    - pkg package management
    - Termux-specific path handling
    - Optional Termux:API integration
    """

    def __init__(
        self,
        home_dir: Optional[str] = None,
        storage_dir: Optional[str] = None,
        use_termux_api: bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Termux-specific paths
        self.home_dir = home_dir or os.environ.get(
            "HOME",
            "/data/data/com.termux/files/home"
        )
        self.storage_dir = storage_dir or os.path.join(self.home_dir, "storage")
        self.prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")

        # Check for Termux:API
        self.termux_api_available = use_termux_api and self._check_termux_api()

    @property
    def platform_name(self) -> str:
        """Return platform name."""
        return "termux"

    @property
    def is_available(self) -> bool:
        """Check if running in Termux environment."""
        # Check for Termux-specific environment variables and paths
        if os.environ.get("TERMUX_VERSION"):
            return True
        if os.path.exists("/data/data/com.termux/files/usr/bin"):
            return True
        if "com.termux" in os.environ.get("PREFIX", ""):
            return True
        return False

    def _check_termux_api(self) -> bool:
        """Check if Termux:API is installed."""
        api_path = os.path.join(self.prefix, "bin", "termux-toast")
        return os.path.exists(api_path)

    async def execute(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        """
        Execute a shell command in Termux.

        Args:
            command: The command to execute
            timeout: Maximum execution time in seconds
            cwd: Working directory (defaults to home)
            env: Additional environment variables

        Returns:
            CommandResult with execution details
        """
        # Check emergency stop
        if EmergencyStop.check():
            return CommandResult(
                stdout="",
                stderr=f"Emergency stop active: {EmergencyStop.get_reason()}",
                exit_code=-1,
                status=ExecutionStatus.BLOCKED,
                command=command,
            )

        # Safety check
        safety = check_command_safety(command)
        if not safety.safe:
            self.logger.warning(f"Blocked unsafe command: {safety.reason}")
            return CommandResult(
                stdout="",
                stderr=f"Command blocked: {safety.reason}",
                exit_code=-1,
                status=ExecutionStatus.BLOCKED,
                command=command,
                metadata={"violation": safety.violation.value if safety.violation else None},
            )

        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        # Ensure Termux paths are in PATH
        termux_bin = os.path.join(self.prefix, "bin")
        if termux_bin not in exec_env.get("PATH", ""):
            exec_env["PATH"] = f"{termux_bin}:{exec_env.get('PATH', '')}"

        # Execute command
        start_time = datetime.now()
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or self.home_dir,
                env=exec_env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = process.returncode or 0
                status = ExecutionStatus.SUCCESS if exit_code == 0 else ExecutionStatus.FAILED

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CommandResult(
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    exit_code=-1,
                    status=ExecutionStatus.TIMEOUT,
                    command=command,
                    duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                )

        except PermissionError as e:
            return CommandResult(
                stdout="",
                stderr=f"Permission denied: {e}",
                exit_code=-1,
                status=ExecutionStatus.PERMISSION_DENIED,
                command=command,
            )
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=f"Execution error: {e}",
                exit_code=-1,
                status=ExecutionStatus.FAILED,
                command=command,
            )

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            status=status,
            command=command,
            duration_ms=duration_ms,
        )

    async def file_exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        # Safety check
        safety = check_path_safety(path, allowed_roots=[self.home_dir, self.storage_dir, "/tmp"])
        if not safety.safe:
            self.logger.warning(f"Path check blocked: {safety.reason}")
            return False

        return os.path.exists(path)

    async def read_file(self, path: str) -> str:
        """Read contents of a file."""
        safety = check_path_safety(path, allowed_roots=[self.home_dir, self.storage_dir, "/tmp"])
        if not safety.safe:
            raise PermissionError(f"Path blocked: {safety.reason}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read file: {e}")

    async def write_file(self, path: str, content: str) -> bool:
        """Write content to a file."""
        safety = check_path_safety(path, allowed_roots=[self.home_dir, self.storage_dir, "/tmp"])
        if not safety.safe:
            raise PermissionError(f"Path blocked: {safety.reason}")

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write file: {e}")
            return False

    async def list_directory(self, path: str) -> List[FileInfo]:
        """List contents of a directory."""
        safety = check_path_safety(path, allowed_roots=[self.home_dir, self.storage_dir, "/tmp"])
        if not safety.safe:
            raise PermissionError(f"Path blocked: {safety.reason}")

        results = []
        try:
            for entry in os.scandir(path):
                try:
                    stat_info = entry.stat()
                    results.append(FileInfo(
                        path=entry.path,
                        exists=True,
                        is_file=entry.is_file(),
                        is_dir=entry.is_dir(),
                        size_bytes=stat_info.st_size,
                        modified_at=datetime.fromtimestamp(stat_info.st_mtime),
                        permissions=oct(stat_info.st_mode)[-3:],
                    ))
                except (OSError, PermissionError):
                    results.append(FileInfo(
                        path=entry.path,
                        exists=True,
                        is_file=entry.is_file(),
                        is_dir=entry.is_dir(),
                    ))
        except Exception as e:
            self.logger.error(f"Failed to list directory: {e}")

        return results

    async def get_system_info(self) -> Dict[str, Any]:
        """Get Termux system information."""
        info = {
            "platform": "termux",
            "home_dir": self.home_dir,
            "prefix": self.prefix,
            "termux_api_available": self.termux_api_available,
        }

        # Get Android version if available
        try:
            result = await self.execute("getprop ro.build.version.release", timeout=5)
            if result.success:
                info["android_version"] = result.stdout.strip()
        except Exception:
            pass

        # Get Termux version
        termux_version = os.environ.get("TERMUX_VERSION")
        if termux_version:
            info["termux_version"] = termux_version

        # Get memory info
        try:
            result = await self.execute("free -h", timeout=5)
            if result.success:
                info["memory_info"] = result.stdout.strip()
        except Exception:
            pass

        # Get disk info
        try:
            result = await self.execute(f"df -h {self.home_dir}", timeout=5)
            if result.success:
                info["disk_info"] = result.stdout.strip()
        except Exception:
            pass

        return info

    async def install_package(self, package: str) -> CommandResult:
        """Install a package using pkg."""
        # Use pkg for Termux package management
        return await self.execute(f"pkg install -y {package}", timeout=300)

    async def check_package_installed(self, package: str) -> bool:
        """Check if a package is installed."""
        result = await self.execute(f"pkg list-installed | grep -q '^{package}'", timeout=10)
        return result.success

    # Termux-specific methods

    async def termux_toast(self, message: str) -> bool:
        """Show a toast notification (requires Termux:API)."""
        if not self.termux_api_available:
            return False

        result = await self.execute(f'termux-toast "{message}"', timeout=5)
        return result.success

    async def termux_vibrate(self, duration_ms: int = 500) -> bool:
        """Vibrate the device (requires Termux:API)."""
        if not self.termux_api_available:
            return False

        result = await self.execute(f"termux-vibrate -d {duration_ms}", timeout=5)
        return result.success

    async def termux_battery_status(self) -> Optional[Dict[str, Any]]:
        """Get battery status (requires Termux:API)."""
        if not self.termux_api_available:
            return None

        result = await self.execute("termux-battery-status", timeout=5)
        if result.success:
            try:
                import json
                return json.loads(result.stdout)
            except Exception:
                pass
        return None

    async def setup_storage(self) -> bool:
        """Setup Termux storage access."""
        result = await self.execute("termux-setup-storage", timeout=30)
        return result.success
