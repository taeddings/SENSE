"""
SENSE-v2 FileSystem Tools
Schema-based tools for file operations.
"""

from typing import Any, Dict, List, Optional
import os
import logging
from pathlib import Path
from datetime import datetime

from sense_v2.core.base import BaseTool, ToolRegistry
from sense_v2.core.schemas import ToolSchema, ToolParameter, ToolResult


@ToolRegistry.register
class FileReadTool(BaseTool):
    """Tool for reading file contents."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Max file size to read (10MB)
        self.max_file_size = 10 * 1024 * 1024

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
        **kwargs
    ) -> ToolResult:
        """Read file contents."""
        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return ToolResult.error(f"File not found: {path}")

            if not file_path.is_file():
                return ToolResult.error(f"Not a file: {path}")

            # Check file size
            size = file_path.stat().st_size
            if size > self.max_file_size:
                return ToolResult.error(
                    f"File too large: {size} bytes (max {self.max_file_size})"
                )

            # Read file
            content = file_path.read_text(encoding=encoding)

            # Limit lines if specified
            if max_lines > 0:
                lines = content.split('\n')
                content = '\n'.join(lines[:max_lines])

            return ToolResult.success(
                content,
                metadata={
                    "path": str(file_path),
                    "size": size,
                    "encoding": encoding,
                },
            )

        except UnicodeDecodeError as e:
            return ToolResult.error(f"Encoding error: {e}")
        except PermissionError:
            return ToolResult.error(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error(str(e))


@ToolRegistry.register
class FileWriteTool(BaseTool):
    """Tool for writing file contents."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

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
        **kwargs
    ) -> ToolResult:
        """Write content to file."""
        try:
            file_path = Path(path).expanduser().resolve()

            # Create parent directories
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write mode
            write_mode = 'w' if mode == "overwrite" else 'a'

            with open(file_path, write_mode, encoding=encoding) as f:
                f.write(content)

            return ToolResult.success(
                {
                    "path": str(file_path),
                    "bytes_written": len(content.encode(encoding)),
                    "mode": mode,
                },
            )

        except PermissionError:
            return ToolResult.error(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error(str(e))


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
