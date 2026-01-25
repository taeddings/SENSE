"""
SENSE Bridge Linux Driver
System driver implementation for standard Linux environments.

Part of Phase 1: Milestone 1.3 - Bridge & OS Control

Linux-specific features:
- apt/apt-get/dnf/pacman package managers
- systemd integration
- Standard FHS paths
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


class LinuxDriver(SystemDriverABC):
    """
    System driver for standard Linux environments.

    Provides Linux-specific implementations for:
    - Command execution via subprocess
    - Multiple package manager support (apt, dnf, pacman)
    - Standard FHS path handling
    """

    def __init__(
        self,
        home_dir: Optional[str] = None,
        allowed_roots: Optional[List[str]] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Linux paths
        self.home_dir = home_dir or os.environ.get("HOME", os.path.expanduser("~"))
        self.allowed_roots = allowed_roots or [
            self.home_dir,
            "/tmp",
            "/var/tmp",
        ]

        # Detect package manager
        self.package_manager = self._detect_package_manager()

    @property
    def platform_name(self) -> str:
        """Return platform name."""
        return "linux"

    @property
    def is_available(self) -> bool:
        """Check if running on Linux."""
        import platform
        return platform.system().lower() == "linux"

    def _detect_package_manager(self) -> str:
        """Detect the system's package manager."""
        if os.path.exists("/usr/bin/apt"):
            return "apt"
        elif os.path.exists("/usr/bin/apt-get"):
            return "apt-get"
        elif os.path.exists("/usr/bin/dnf"):
            return "dnf"
        elif os.path.exists("/usr/bin/yum"):
            return "yum"
        elif os.path.exists("/usr/bin/pacman"):
            return "pacman"
        elif os.path.exists("/usr/bin/zypper"):
            return "zypper"
        elif os.path.exists("/usr/bin/apk"):
            return "apk"
        return "unknown"

    async def execute(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        """
        Execute a shell command on Linux.

        Args:
            command: The command to execute
            timeout: Maximum execution time in seconds
            cwd: Working directory
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
        safety = check_path_safety(path, allowed_roots=self.allowed_roots)
        if not safety.safe:
            self.logger.warning(f"Path check blocked: {safety.reason}")
            return False

        return os.path.exists(path)

    async def read_file(self, path: str) -> str:
        """Read contents of a file."""
        safety = check_path_safety(path, allowed_roots=self.allowed_roots)
        if not safety.safe:
            raise PermissionError(f"Path blocked: {safety.reason}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read file: {e}")

    async def write_file(self, path: str, content: str) -> bool:
        """Write content to a file."""
        safety = check_path_safety(path, allowed_roots=self.allowed_roots)
        if not safety.safe:
            raise PermissionError(f"Path blocked: {safety.reason}")

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write file: {e}")
            return False

    async def list_directory(self, path: str) -> List[FileInfo]:
        """List contents of a directory."""
        safety = check_path_safety(path, allowed_roots=self.allowed_roots)
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
                        owner=str(stat_info.st_uid),
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
        """Get Linux system information."""
        import platform

        info = {
            "platform": "linux",
            "home_dir": self.home_dir,
            "package_manager": self.package_manager,
            "kernel": platform.release(),
            "architecture": platform.machine(),
        }

        # Get distribution info
        try:
            result = await self.execute("cat /etc/os-release | grep PRETTY_NAME", timeout=5)
            if result.success:
                line = result.stdout.strip()
                if "=" in line:
                    info["distribution"] = line.split("=", 1)[1].strip('"')
        except Exception:
            pass

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

        # Get CPU info
        try:
            result = await self.execute("nproc", timeout=5)
            if result.success:
                info["cpu_cores"] = int(result.stdout.strip())
        except Exception:
            pass

        return info

    async def install_package(self, package: str) -> CommandResult:
        """Install a package using the detected package manager."""
        install_commands = {
            "apt": f"apt install -y {package}",
            "apt-get": f"apt-get install -y {package}",
            "dnf": f"dnf install -y {package}",
            "yum": f"yum install -y {package}",
            "pacman": f"pacman -S --noconfirm {package}",
            "zypper": f"zypper install -y {package}",
            "apk": f"apk add {package}",
        }

        if self.package_manager not in install_commands:
            return CommandResult(
                stdout="",
                stderr=f"Unknown package manager: {self.package_manager}",
                exit_code=-1,
                status=ExecutionStatus.FAILED,
                command="",
            )

        command = install_commands[self.package_manager]
        return await self.execute(command, timeout=300)

    async def check_package_installed(self, package: str) -> bool:
        """Check if a package is installed."""
        check_commands = {
            "apt": f"dpkg -l | grep -q '^ii.*{package}'",
            "apt-get": f"dpkg -l | grep -q '^ii.*{package}'",
            "dnf": f"rpm -q {package}",
            "yum": f"rpm -q {package}",
            "pacman": f"pacman -Q {package}",
            "zypper": f"rpm -q {package}",
            "apk": f"apk info -e {package}",
        }

        if self.package_manager not in check_commands:
            return False

        result = await self.execute(check_commands[self.package_manager], timeout=10)
        return result.success

    # Linux-specific methods

    async def get_service_status(self, service: str) -> Optional[str]:
        """Get systemd service status."""
        result = await self.execute(f"systemctl status {service}", timeout=10)
        if result.success or result.exit_code == 3:  # 3 = service not running
            return result.stdout
        return None

    async def get_process_list(self, filter_pattern: Optional[str] = None) -> str:
        """Get running processes."""
        command = "ps aux"
        if filter_pattern:
            command += f" | grep '{filter_pattern}'"
        result = await self.execute(command, timeout=10)
        return result.stdout if result.success else ""
