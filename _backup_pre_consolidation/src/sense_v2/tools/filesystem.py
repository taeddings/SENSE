"""
SENSE-v2 FileSystem Tools
Schema-based tools for file operations.

Hardened for SENSE v2 Unified Evolutionary Architecture:
- Strict type annotations throughout
- FileSystemError for proper error propagation
- Path traversal attack prevention
- Sensitive file access restrictions
"""

from typing import Any, Dict, List, Optional, Union, Set
import os
import logging
import re
from pathlib import Path
from datetime import datetime

from sense_v2.core.base import BaseTool, ToolRegistry
from sense_v2.core.schemas import ToolSchema, ToolParameter, ToolResult


class FileSystemError(Exception):
    """
    Raised when filesystem operations fail.

    Used for self-correction loops - the agent can parse the error
    and retry with a corrected path or operation.
    """
    def __init__(
        self,
        message: str,
        path: str = "",
        operation: str = "",
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.path = path
        self.operation = operation
        self.recoverable = recoverable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": str(self),
            "path": self.path,
            "operation": self.operation,
            "recoverable": self.recoverable,
        }


# Sensitive paths that should be protected from write/delete
PROTECTED_PATHS: Set[str] = {
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/ssh",
    "/root/.ssh",
    "~/.ssh",
    "/boot",
    "/sys",
    "/proc",
}

# Sensitive patterns (regex)
SENSITIVE_PATTERNS: List[str] = [
    r"\.env$",
    r"\.env\.",
    r"id_rsa",
    r"id_ed25519",
    r"\.pem$",
    r"\.key$",
    r"password",
    r"secret",
    r"credentials",
    r"\.aws/credentials",
    r"\.kube/config",
]


def is_path_sensitive(path: Union[str, Path], operation: str = "read") -> bool:
    """
    Check if a path is sensitive and should be protected.

    Args:
        path: The path to check
        operation: The operation being performed ("read", "write", "delete")

    Returns:
        True if the path is sensitive
    """
    path_str = str(Path(path).resolve())

    # Check protected paths (more strict for write/delete)
    if operation in ("write", "delete"):
        for protected in PROTECTED_PATHS:
            expanded = str(Path(protected).expanduser())
            if path_str.startswith(expanded) or path_str == expanded:
                return True

    # Check sensitive patterns
    path_lower = path_str.lower()
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, path_lower, re.IGNORECASE):
            return True

    return False


def sanitize_path(path: str, base_dir: Optional[str] = None) -> Path:
    """
    Sanitize a path to prevent traversal attacks.

    Args:
        path: The path to sanitize
        base_dir: Optional base directory to restrict access to

    Returns:
        Sanitized Path object

    Raises:
        FileSystemError: If path traversal is detected
    """
    # Expand user home
    expanded = Path(path).expanduser()

    # Resolve to absolute path
    resolved = expanded.resolve()

    # Check for path traversal if base_dir is specified
    if base_dir:
        base_resolved = Path(base_dir).expanduser().resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise FileSystemError(
                f"Path traversal detected: {path} escapes {base_dir}",
                path=str(path),
                operation="access",
                recoverable=False,
            )

    return resolved


@ToolRegistry.register
class FileReadTool(BaseTool):
    """
    Tool for reading file contents with security hardening.

    Features:
    - Path traversal prevention
    - Sensitive file warnings
    - Size limits for memory safety
    - Proper error propagation
    """

    # Default limits
    DEFAULT_MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    DEFAULT_MAX_LINES: int = 10000

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurable limits
        self.max_file_size: int = (
            config.get("max_file_size", self.DEFAULT_MAX_FILE_SIZE)
            if config else self.DEFAULT_MAX_FILE_SIZE
        )

        # Optional base directory restriction
        self.base_dir: Optional[str] = config.get("base_dir") if config else None

        # Track read operations for auditing
        self._read_count: int = 0

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="file_read",
            description="Read the contents of a file",
            parameters=[
                ToolParameter(
                    name="path",
                    param_type="string",
                    description="Path to the file to read",
                    required=True,
                ),
                ToolParameter(
                    name="encoding",
                    param_type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8",
                ),
                ToolParameter(
                    name="max_lines",
                    param_type="integer",
                    description="Maximum lines to read (0 for all)",
                    required=False,
                    default=0,
                    min_value=0,
                ),
            ],
            returns="string",
            returns_description="File contents as string",
            category="filesystem",
            max_retries=1,
        )

    async def execute(
        self,
        path: str,
        encoding: str = "utf-8",
        max_lines: int = 0,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Read file contents with security checks.

        Args:
            path: Path to the file to read
            encoding: File encoding (default: utf-8)
            max_lines: Maximum lines to read (0 for all)

        Returns:
            ToolResult with file contents or error
        """
        try:
            # Sanitize and validate path
            try:
                file_path = sanitize_path(path, self.base_dir)
            except FileSystemError as e:
                return ToolResult.error(str(e), metadata={"security_error": True})

            if not file_path.exists():
                return ToolResult.error(
                    f"File not found: {path}",
                    metadata={"path": str(file_path), "error_type": "not_found"},
                )

            if not file_path.is_file():
                return ToolResult.error(
                    f"Not a file: {path}",
                    metadata={"path": str(file_path), "error_type": "not_file"},
                )

            # Check for sensitive files
            if is_path_sensitive(file_path, "read"):
                self.logger.warning(f"Reading sensitive file: {file_path}")

            # Check file size
            size = file_path.stat().st_size
            if size > self.max_file_size:
                return ToolResult.error(
                    f"File too large: {size} bytes (max {self.max_file_size})",
                    metadata={
                        "path": str(file_path),
                        "size": size,
                        "max_size": self.max_file_size,
                        "error_type": "too_large",
                    },
                )

            # Read file
            content = file_path.read_text(encoding=encoding)

            # Limit lines if specified
            lines_count = content.count('\n') + 1
            if max_lines > 0:
                lines = content.split('\n')
                content = '\n'.join(lines[:max_lines])
                truncated = len(lines) > max_lines
            else:
                truncated = False

            self._read_count += 1

            return ToolResult.success(
                content,
                metadata={
                    "path": str(file_path),
                    "size": size,
                    "encoding": encoding,
                    "lines": lines_count,
                    "truncated": truncated,
                    "read_count": self._read_count,
                },
            )

        except UnicodeDecodeError as e:
            return ToolResult.error(
                f"Encoding error: {e}",
                metadata={"path": path, "encoding": encoding, "error_type": "encoding"},
            )
        except PermissionError:
            return ToolResult.error(
                f"Permission denied: {path}",
                metadata={"path": path, "error_type": "permission"},
            )
        except Exception as e:
            self.logger.error(f"File read error: {e}")
            return ToolResult.error(str(e), metadata={"path": path})


@ToolRegistry.register
class FileWriteTool(BaseTool):
    """
    Tool for writing file contents with security hardening.

    Features:
    - Path traversal prevention
    - Sensitive file protection
    - Backup creation option
    - Proper error propagation
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Optional base directory restriction
        self.base_dir: Optional[str] = config.get("base_dir") if config else None

        # Whether to allow writing to sensitive paths
        self.allow_sensitive: bool = config.get("allow_sensitive", False) if config else False

        # Track write operations for auditing
        self._write_count: int = 0

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="file_write",
            description="Write content to a file",
            parameters=[
                ToolParameter(
                    name="path",
                    param_type="string",
                    description="Path to the file to write",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    param_type="string",
                    description="Content to write",
                    required=True,
                ),
                ToolParameter(
                    name="encoding",
                    param_type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8",
                ),
                ToolParameter(
                    name="mode",
                    param_type="string",
                    description="Write mode: overwrite or append",
                    required=False,
                    default="overwrite",
                    enum=["overwrite", "append"],
                ),
                ToolParameter(
                    name="create_dirs",
                    param_type="boolean",
                    description="Create parent directories if needed",
                    required=False,
                    default=True,
                ),
            ],
            returns="object",
            returns_description="Write result with path and bytes written",
            category="filesystem",
            requires_confirmation=False,
            max_retries=1,
        )

    async def execute(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        mode: str = "overwrite",
        create_dirs: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Write content to file with security checks.

        Args:
            path: Path to the file to write
            content: Content to write
            encoding: File encoding (default: utf-8)
            mode: Write mode ("overwrite" or "append")
            create_dirs: Whether to create parent directories

        Returns:
            ToolResult with write status or error
        """
        try:
            # Sanitize and validate path
            try:
                file_path = sanitize_path(path, self.base_dir)
            except FileSystemError as e:
                return ToolResult.error(str(e), metadata={"security_error": True})

            # Check for sensitive files
            if is_path_sensitive(file_path, "write"):
                if not self.allow_sensitive:
                    return ToolResult.error(
                        f"Cannot write to sensitive path: {path}",
                        metadata={
                            "path": str(file_path),
                            "security_error": True,
                            "error_type": "sensitive_path",
                        },
                    )
                self.logger.warning(f"Writing to sensitive file: {file_path}")

            # Check if file exists (for logging)
            file_existed = file_path.exists()
            original_size = file_path.stat().st_size if file_existed else 0

            # Create parent directories
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write mode
            write_mode = 'w' if mode == "overwrite" else 'a'

            with open(file_path, write_mode, encoding=encoding) as f:
                f.write(content)

            bytes_written = len(content.encode(encoding))
            new_size = file_path.stat().st_size

            self._write_count += 1

            return ToolResult.success(
                {
                    "path": str(file_path),
                    "bytes_written": bytes_written,
                    "mode": mode,
                    "file_existed": file_existed,
                    "new_size": new_size,
                },
                metadata={
                    "write_count": self._write_count,
                    "original_size": original_size,
                },
            )

        except PermissionError:
            return ToolResult.error(
                f"Permission denied: {path}",
                metadata={"path": path, "error_type": "permission"},
            )
        except OSError as e:
            return ToolResult.error(
                f"OS error: {e}",
                metadata={"path": path, "error_type": "os_error"},
            )
        except Exception as e:
            self.logger.error(f"File write error: {e}")
            return ToolResult.error(str(e), metadata={"path": path})


@ToolRegistry.register
class FileListTool(BaseTool):
    """Tool for listing directory contents."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="file_list",
            description="List contents of a directory",
            parameters=[
                ToolParameter(
                    name="path",
                    param_type="string",
                    description="Directory path to list",
                    required=True,
                ),
                ToolParameter(
                    name="pattern",
                    param_type="string",
                    description="Glob pattern to filter results",
                    required=False,
                    default="*",
                ),
                ToolParameter(
                    name="recursive",
                    param_type="boolean",
                    description="Recurse into subdirectories",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="include_hidden",
                    param_type="boolean",
                    description="Include hidden files",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="max_results",
                    param_type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=100,
                    min_value=1,
                    max_value=1000,
                ),
            ],
            returns="array",
            returns_description="List of file info objects",
            category="filesystem",
            max_retries=1,
        )

    async def execute(
        self,
        path: str,
        pattern: str = "*",
        recursive: bool = False,
        include_hidden: bool = False,
        max_results: int = 100,
        **kwargs
    ) -> ToolResult:
        """List directory contents."""
        try:
            dir_path = Path(path).expanduser().resolve()

            if not dir_path.exists():
                return ToolResult.error(f"Directory not found: {path}")

            if not dir_path.is_dir():
                return ToolResult.error(f"Not a directory: {path}")

            # Get files
            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))

            # Filter hidden if needed
            if not include_hidden:
                files = [f for f in files if not f.name.startswith('.')]

            # Limit results
            files = files[:max_results]

            # Build result
            results = []
            for f in files:
                try:
                    stat = f.stat()
                    results.append({
                        "name": f.name,
                        "path": str(f),
                        "is_dir": f.is_dir(),
                        "size": stat.st_size if f.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except (PermissionError, OSError):
                    continue

            return ToolResult.success(
                results,
                metadata={
                    "path": str(dir_path),
                    "total": len(results),
                    "pattern": pattern,
                },
            )

        except PermissionError:
            return ToolResult.error(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error(str(e))


@ToolRegistry.register
class FileExistsTool(BaseTool):
    """Tool for checking if a file exists."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="file_exists",
            description="Check if a file or directory exists",
            parameters=[
                ToolParameter(
                    name="path",
                    param_type="string",
                    description="Path to check",
                    required=True,
                ),
            ],
            returns="object",
            returns_description="Object with exists, is_file, is_dir fields",
            category="filesystem",
            max_retries=0,
        )

    async def execute(self, path: str, **kwargs) -> ToolResult:
        """Check file existence."""
        try:
            file_path = Path(path).expanduser().resolve()

            return ToolResult.success({
                "exists": file_path.exists(),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "path": str(file_path),
            })

        except Exception as e:
            return ToolResult.error(str(e))
