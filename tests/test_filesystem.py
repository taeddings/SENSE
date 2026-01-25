"""
Tests for SENSE-v2 FileSystem Tools
Per SYSTEM_PROMPT: Every tool must include test_[toolname].py
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from sense.tools.filesystem import (
    FileReadTool,
    FileWriteTool,
    FileListTool,
    FileExistsTool,
)
from sense.core.schemas import ToolResultStatus


class TestFileReadSuccess:
    """Test successful file read operations."""

    @pytest.mark.asyncio
    async def test_file_read_success(self):
        """Read existing file."""
        tool = FileReadTool()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content here")
            temp_path = f.name

        try:
            result = await tool.execute(path=temp_path)

            assert result.is_success is True
            assert result.status == ToolResultStatus.SUCCESS
            assert "test content here" in result.output
            assert "path" in result.metadata
            assert "size" in result.metadata
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_file_read_with_encoding(self):
        """Read file with specific encoding."""
        tool = FileReadTool()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("unicode content: \u00e9\u00e0\u00fc")
            temp_path = f.name

        try:
            result = await tool.execute(path=temp_path, encoding="utf-8")

            assert result.is_success is True
            assert "\u00e9\u00e0\u00fc" in result.output
            assert result.metadata["encoding"] == "utf-8"
        finally:
            os.unlink(temp_path)


class TestFileReadNotFound:
    """Test file not found handling."""

    @pytest.mark.asyncio
    async def test_file_read_not_found(self):
        """Handle missing file."""
        tool = FileReadTool()
        result = await tool.execute(path="/nonexistent/path/file.txt")

        assert result.is_success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_file_read_directory_error(self):
        """Handle directory instead of file."""
        tool = FileReadTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await tool.execute(path=tmpdir)

            assert result.is_success is False
            assert "not a file" in result.error.lower()


class TestFileReadMaxLines:
    """Test line limiting functionality."""

    @pytest.mark.asyncio
    async def test_file_read_max_lines(self):
        """Verify line limiting."""
        tool = FileReadTool()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for i in range(100):
                f.write(f"line {i}\n")
            temp_path = f.name

        try:
            result = await tool.execute(path=temp_path, max_lines=5)

            assert result.is_success is True
            lines = result.output.strip().split('\n')
            assert len(lines) == 5
            assert "line 0" in lines[0]
            assert "line 4" in lines[4]
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_file_read_all_lines(self):
        """Verify max_lines=0 reads all lines."""
        tool = FileReadTool()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for i in range(10):
                f.write(f"line {i}\n")
            temp_path = f.name

        try:
            result = await tool.execute(path=temp_path, max_lines=0)

            assert result.is_success is True
            lines = result.output.strip().split('\n')
            assert len(lines) == 10
        finally:
            os.unlink(temp_path)


class TestFileWriteSuccess:
    """Test successful file write operations."""

    @pytest.mark.asyncio
    async def test_file_write_success(self):
        """Write new file."""
        tool = FileWriteTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_write.txt")
            result = await tool.execute(
                path=file_path,
                content="new file content"
            )

            assert result.is_success is True
            assert "path" in result.output
            assert "bytes_written" in result.output
            assert result.output["bytes_written"] > 0

            # Verify file was written
            with open(file_path, 'r') as f:
                assert f.read() == "new file content"

    @pytest.mark.asyncio
    async def test_file_write_overwrite(self):
        """Write overwrites existing file."""
        tool = FileWriteTool()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("original content")
            temp_path = f.name

        try:
            result = await tool.execute(
                path=temp_path,
                content="new content",
                mode="overwrite"
            )

            assert result.is_success is True

            with open(temp_path, 'r') as f:
                assert f.read() == "new content"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_file_write_creates_directories(self):
        """Write creates parent directories."""
        tool = FileWriteTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "a", "b", "c", "test.txt")
            result = await tool.execute(
                path=nested_path,
                content="nested content",
                create_dirs=True
            )

            assert result.is_success is True
            assert os.path.exists(nested_path)


class TestFileWriteAppend:
    """Test file append mode."""

    @pytest.mark.asyncio
    async def test_file_write_append(self):
        """Append mode adds to existing file."""
        tool = FileWriteTool()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("original ")
            temp_path = f.name

        try:
            result = await tool.execute(
                path=temp_path,
                content="appended",
                mode="append"
            )

            assert result.is_success is True
            assert result.output["mode"] == "append"

            with open(temp_path, 'r') as f:
                assert f.read() == "original appended"
        finally:
            os.unlink(temp_path)


class TestFileListDirectory:
    """Test directory listing."""

    @pytest.mark.asyncio
    async def test_file_list_directory(self):
        """List directory contents."""
        tool = FileListTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                Path(tmpdir, f"file{i}.txt").touch()

            result = await tool.execute(path=tmpdir)

            assert result.is_success is True
            assert len(result.output) == 3
            assert all("name" in item for item in result.output)
            assert all("path" in item for item in result.output)
            assert all("is_dir" in item for item in result.output)

    @pytest.mark.asyncio
    async def test_file_list_not_found(self):
        """Handle nonexistent directory."""
        tool = FileListTool()
        result = await tool.execute(path="/nonexistent/directory")

        assert result.is_success is False
        assert "not found" in result.error.lower()


class TestFileListPattern:
    """Test glob pattern filtering."""

    @pytest.mark.asyncio
    async def test_file_list_pattern(self):
        """Glob pattern filtering."""
        tool = FileListTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mixed files
            Path(tmpdir, "file1.txt").touch()
            Path(tmpdir, "file2.txt").touch()
            Path(tmpdir, "file3.py").touch()

            result = await tool.execute(path=tmpdir, pattern="*.txt")

            assert result.is_success is True
            assert len(result.output) == 2
            assert all(item["name"].endswith(".txt") for item in result.output)

    @pytest.mark.asyncio
    async def test_file_list_recursive(self):
        """Recursive directory listing."""
        tool = FileListTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            Path(tmpdir, "root.txt").touch()
            Path(subdir, "nested.txt").touch()

            result = await tool.execute(path=tmpdir, pattern="*.txt", recursive=True)

            assert result.is_success is True
            assert len(result.output) == 2

    @pytest.mark.asyncio
    async def test_file_list_hidden_files(self):
        """Include/exclude hidden files."""
        tool = FileListTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "visible.txt").touch()
            Path(tmpdir, ".hidden.txt").touch()

            # Without hidden
            result = await tool.execute(path=tmpdir, include_hidden=False)
            assert len(result.output) == 1
            assert result.output[0]["name"] == "visible.txt"

            # With hidden
            result = await tool.execute(path=tmpdir, include_hidden=True)
            assert len(result.output) == 2

    @pytest.mark.asyncio
    async def test_file_list_max_results(self):
        """Max results limiting."""
        tool = FileListTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(20):
                Path(tmpdir, f"file{i}.txt").touch()

            result = await tool.execute(path=tmpdir, max_results=5)

            assert result.is_success is True
            assert len(result.output) == 5


class TestFileExists:
    """Test file existence checking."""

    @pytest.mark.asyncio
    async def test_file_exists(self):
        """Check existence of file."""
        tool = FileExistsTool()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            result = await tool.execute(path=temp_path)

            assert result.is_success is True
            assert result.output["exists"] is True
            assert result.output["is_file"] is True
            assert result.output["is_dir"] is False
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_file_exists_directory(self):
        """Check existence of directory."""
        tool = FileExistsTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await tool.execute(path=tmpdir)

            assert result.is_success is True
            assert result.output["exists"] is True
            assert result.output["is_file"] is False
            assert result.output["is_dir"] is True

    @pytest.mark.asyncio
    async def test_file_exists_not_found(self):
        """Check nonexistent path."""
        tool = FileExistsTool()
        result = await tool.execute(path="/nonexistent/path/12345")

        assert result.is_success is True  # Tool succeeds, just reports not found
        assert result.output["exists"] is False
        assert result.output["is_file"] is False
        assert result.output["is_dir"] is False


class TestFileToolSchemas:
    """Test tool schema definitions."""

    def test_file_read_schema(self):
        """Verify FileReadTool schema."""
        tool = FileReadTool()
        schema = tool.schema

        assert schema.name == "file_read"
        assert schema.category == "filesystem"
        assert any(p.name == "path" and p.required for p in schema.parameters)

    def test_file_write_schema(self):
        """Verify FileWriteTool schema."""
        tool = FileWriteTool()
        schema = tool.schema

        assert schema.name == "file_write"
        assert schema.category == "filesystem"
        assert any(p.name == "path" and p.required for p in schema.parameters)
        assert any(p.name == "content" and p.required for p in schema.parameters)
        assert any(p.name == "mode" and p.enum == ["overwrite", "append"] for p in schema.parameters)

    def test_file_list_schema(self):
        """Verify FileListTool schema."""
        tool = FileListTool()
        schema = tool.schema

        assert schema.name == "file_list"
        assert schema.category == "filesystem"
        assert any(p.name == "pattern" for p in schema.parameters)
        assert any(p.name == "recursive" for p in schema.parameters)

    def test_file_exists_schema(self):
        """Verify FileExistsTool schema."""
        tool = FileExistsTool()
        schema = tool.schema

        assert schema.name == "file_exists"
        assert schema.category == "filesystem"
        assert len(schema.parameters) == 1
        assert schema.parameters[0].name == "path"


class TestFileToolValidation:
    """Test input validation."""

    def test_file_read_validates_path(self):
        """FileReadTool validates path parameter."""
        tool = FileReadTool()

        errors = tool.validate_input(path="/some/path")
        assert len(errors) == 0

        errors = tool.validate_input()  # Missing required path
        assert len(errors) > 0

    def test_file_write_validates_content(self):
        """FileWriteTool validates content parameter."""
        tool = FileWriteTool()

        errors = tool.validate_input(path="/some/path", content="data")
        assert len(errors) == 0

        errors = tool.validate_input(path="/some/path")  # Missing content
        assert len(errors) > 0
