"""
SENSE Bridge Interface
Abstract contracts for system driver and grounding source implementations.

Part of Phase 1: Milestone 1.3 - Bridge & OS Control

The Bridge layer abstracts OS-specific operations, providing:
- Unified command execution interface
- Safety validation before execution
- Platform-specific driver implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime


class ExecutionStatus(Enum):
    """Status of command execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class CommandResult:
    """
    Result of a command execution.

    Attributes:
        stdout: Standard output from the command
        stderr: Standard error from the command
        exit_code: Process exit code (0 = success)
        status: Execution status
        command: The command that was executed
        duration_ms: Execution time in milliseconds
        metadata: Additional execution metadata
    """
    stdout: str
    stderr: str
    exit_code: int
    status: ExecutionStatus
    command: str
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0 and self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "status": self.status.value,
            "command": self.command,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    exists: bool
    is_file: bool = False
    is_dir: bool = False
    size_bytes: int = 0
    modified_at: Optional[datetime] = None
    permissions: str = ""
    owner: str = ""


class SystemDriverABC(ABC):
    """
    Abstract base class for system-level operations.

    Implementations provide platform-specific functionality for:
    - Command execution
    - File operations
    - System information queries
    - Package management
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform name (e.g., 'termux', 'linux', 'darwin')."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this driver is available on the current system."""
        pass

    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        """
        Execute a shell command.

        Args:
            command: The command to execute
            timeout: Maximum execution time in seconds
            cwd: Working directory for the command
            env: Environment variables to set

        Returns:
            CommandResult with execution details
        """
        pass

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        pass

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read contents of a file."""
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str) -> bool:
        """Write content to a file."""
        pass

    @abstractmethod
    async def list_directory(self, path: str) -> List[FileInfo]:
        """List contents of a directory."""
        pass

    @abstractmethod
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information (OS, memory, disk, etc.)."""
        pass

    @abstractmethod
    async def install_package(self, package: str) -> CommandResult:
        """Install a system package using the platform's package manager."""
        pass

    @abstractmethod
    async def check_package_installed(self, package: str) -> bool:
        """Check if a package is installed."""
        pass


class GroundingSourceABC(ABC):
    """
    Abstract base class for grounding sources.

    Grounding sources provide verification data for claims:
    - Synthetic: Math, code execution, logic verification
    - Real-world: Web search, API queries, fact checking
    - Experiential: Action outcomes, state verification
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source name (e.g., 'synthetic', 'web_search')."""
        pass

    @property
    @abstractmethod
    def source_tier(self) -> str:
        """Return the grounding tier ('synthetic', 'realworld', 'experiential')."""
        pass

    @abstractmethod
    async def verify(
        self,
        claim: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> "GroundingResult":
        """
        Verify a claim using this grounding source.

        Args:
            claim: The claim to verify
            context: Additional context for verification

        Returns:
            GroundingResult with verification details
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this grounding source is currently available."""
        pass


@dataclass
class GroundingResult:
    """
    Result of a grounding verification.

    Attributes:
        verified: Whether the claim was verified
        confidence: Confidence score (0.0 to 1.0)
        source: Name of the grounding source
        tier: Grounding tier (synthetic/realworld/experiential)
        evidence: Supporting evidence for the verification
        error: Error message if verification failed
    """
    verified: bool
    confidence: float
    source: str
    tier: str
    evidence: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "verified": self.verified,
            "confidence": self.confidence,
            "source": self.source,
            "tier": self.tier,
            "evidence": self.evidence,
            "error": self.error,
            "metadata": self.metadata,
        }


class EmergencyStop:
    """
    Global emergency stop mechanism.

    When triggered, all Bridge operations should check and halt.
    Available at all layers per CLAUDE.md specification.
    """
    _triggered: bool = False
    _reason: str = ""
    _triggered_at: Optional[datetime] = None

    @classmethod
    def trigger(cls, reason: str) -> None:
        """Trigger emergency stop with reason."""
        cls._triggered = True
        cls._reason = reason
        cls._triggered_at = datetime.now()

    @classmethod
    def check(cls) -> bool:
        """Check if emergency stop has been triggered."""
        return cls._triggered

    @classmethod
    def get_reason(cls) -> str:
        """Get the emergency stop reason."""
        return cls._reason

    @classmethod
    def reset(cls) -> None:
        """Reset emergency stop (use with caution)."""
        cls._triggered = False
        cls._reason = ""
        cls._triggered_at = None

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Get full emergency stop status."""
        return {
            "triggered": cls._triggered,
            "reason": cls._reason,
            "triggered_at": cls._triggered_at.isoformat() if cls._triggered_at else None,
        }
