"""
Tests for SENSE-v2 Terminal Tools
Per SYSTEM_PROMPT: Every tool must include test_[toolname].py
"""

import pytest
import asyncio
import tempfile
import os

from sense_v2.tools.terminal import TerminalTool, TerminalInteractiveTool
from sense_v2.core.schemas import ToolResultStatus


class TestTerminalToolSchema:
    """Test terminal tool schema properties."""

    def test_terminal_tool_schema(self):
        """Verify schema properties for TerminalTool."""
        tool = TerminalTool()
        schema = tool.schema

        assert schema.name == "terminal_exec"
        assert schema.category == "terminal"
        assert len(schema.parameters) == 6

        # Check command parameter
        cmd_param = next(p for p in schema.parameters if p.name == "command")
        assert cmd_param.required is True
        assert cmd_param.param_type == "string"

        # Check timeout parameter
        timeout_param = next(p for p in schema.parameters if p.name == "timeout")
        assert timeout_param.required is False
        assert timeout_param.default == 60
        assert timeout_param.min_value == 1
        assert timeout_param.max_value == 300

        # Check cwd parameter
        cwd_param = next(p for p in schema.parameters if p.name == "cwd")
        assert cwd_param.required is False

        # Check session parameter
        session_param = next(p for p in schema.parameters if p.name == "session")
        assert session_param.required is False
        assert session_param.param_type == "string"
        assert session_param.default == "default"

        # Check ssh_host parameter
        ssh_host_param = next(p for p in schema.parameters if p.name == "ssh_host")
        assert ssh_host_param.required is False
        assert ssh_host_param.param_type == "string"

        # Check ssh_user parameter
        ssh_user_param = next(p for p in schema.parameters if p.name == "ssh_user")
        assert ssh_user_param.required is False
        assert ssh_user_param.param_type == "string"
        assert ssh_user_param.default == "root"

    def test_terminal_interactive_schema(self):
        """Verify schema properties for TerminalInteractiveTool."""
        tool = TerminalInteractiveTool()
        schema = tool.schema

        assert schema.name == "terminal_interactive"
        assert schema.category == "terminal"
        assert schema.requires_confirmation is True

        # Check action parameter
        action_param = next(p for p in schema.parameters if p.name == "action")
        assert action_param.required is True
        assert action_param.enum == ["start", "send", "read", "close"]


class TestTerminalExecuteSuccess:
    """Test successful terminal command execution."""

    @pytest.mark.asyncio
    async def test_terminal_execute_success(self):
        """Execute simple command (echo)."""
        tool = TerminalTool()
        result = await tool.execute(command="echo 'hello world'")

        assert result.is_success is True
        assert result.status == ToolResultStatus.SUCCESS
        assert "hello world" in result.output
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_terminal_execute_with_exit_code(self):
        """Verify exit code is captured correctly."""
        tool = TerminalTool()

        # Successful command
        result = await tool.execute(command="true")
        assert result.exit_code == 0
        assert result.is_success is True

        # Failed command
        result = await tool.execute(command="false")
        assert result.exit_code == 1
        assert result.is_success is False

    @pytest.mark.asyncio
    async def test_terminal_execute_captures_stderr(self):
        """Verify stderr is captured."""
        tool = TerminalTool()
        result = await tool.execute(command="ls /nonexistent_path_12345")

        assert result.is_success is False
        assert result.stderr is not None
        assert len(result.stderr) > 0


class TestTerminalExecuteTimeout:
    """Test terminal timeout handling."""

    @pytest.mark.asyncio
    async def test_terminal_execute_timeout(self):
        """Verify timeout handling."""
        tool = TerminalTool()
        result = await tool.execute(command="sleep 5", timeout=1)

        assert result.status == ToolResultStatus.TIMEOUT
        assert result.is_success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_terminal_respects_custom_timeout(self):
        """Verify custom timeout is respected."""
        tool = TerminalTool()

        # Should complete within timeout
        result = await tool.execute(command="sleep 0.1", timeout=5)
        assert result.is_success is True


class TestTerminalBlockedCommand:
    """Test dangerous command blocking."""

    @pytest.mark.asyncio
    async def test_terminal_blocked_command_rm_rf(self):
        """Verify rm -rf / is blocked."""
        tool = TerminalTool()
        result = await tool.execute(command="rm -rf /")

        assert result.is_success is False
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_terminal_blocked_command_fork_bomb(self):
        """Verify fork bomb is blocked."""
        tool = TerminalTool()
        result = await tool.execute(command=":(){:|:&};:")

        assert result.is_success is False
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_terminal_blocked_command_mkfs(self):
        """Verify mkfs is blocked."""
        tool = TerminalTool()
        result = await tool.execute(command="mkfs.ext4 /dev/sda1")

        assert result.is_success is False
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_terminal_blocked_command_dd(self):
        """Verify dd if=/dev/zero of=/dev/sda is blocked."""
        tool = TerminalTool()
        result = await tool.execute(command="dd if=/dev/zero of=/dev/sda")

        assert result.is_success is False
        assert "blocked" in result.error.lower()


class TestTerminalCwdParameter:
    """Test working directory parameter."""

    @pytest.mark.asyncio
    async def test_terminal_cwd_parameter(self):
        """Verify working directory param works."""
        tool = TerminalTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file in temp directory
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            # Execute ls in temp directory
            result = await tool.execute(command="ls", cwd=tmpdir)

            assert result.is_success is True
            assert "test.txt" in result.output

    @pytest.mark.asyncio
    async def test_terminal_cwd_invalid_directory(self):
        """Verify error handling for invalid cwd."""
        tool = TerminalTool()
        result = await tool.execute(
            command="ls",
            cwd="/nonexistent_directory_12345"
        )

        assert result.is_success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_terminal_cwd_in_metadata(self):
        """Verify cwd is recorded in metadata."""
        tool = TerminalTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await tool.execute(command="pwd", cwd=tmpdir)

            assert result.is_success is True
            assert "cwd" in result.metadata
            assert tmpdir in result.metadata["cwd"]


class TestTerminalInteractiveStartClose:
    """Test interactive terminal session lifecycle."""

    @pytest.mark.asyncio
    async def test_terminal_interactive_start_close(self):
        """Test session start and close lifecycle."""
        tool = TerminalInteractiveTool()

        # Start session
        start_result = await tool.execute(action="start")
        assert start_result.is_success is True
        assert "session_id" in start_result.output
        assert start_result.output["status"] == "started"

        session_id = start_result.output["session_id"]

        # Close session
        close_result = await tool.execute(action="close", session_id=session_id)
        assert close_result.is_success is True
        assert close_result.output["status"] == "closed"

    @pytest.mark.asyncio
    async def test_terminal_interactive_close_nonexistent(self):
        """Test closing nonexistent session."""
        tool = TerminalInteractiveTool()

        result = await tool.execute(action="close", session_id="nonexistent_123")
        assert result.is_success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_terminal_interactive_invalid_action(self):
        """Test invalid action handling."""
        tool = TerminalInteractiveTool()

        result = await tool.execute(action="send")  # Missing session_id and input
        assert result.is_success is False


class TestTerminalToolMetadata:
    """Test tool metadata and tracking."""

    @pytest.mark.asyncio
    async def test_terminal_command_in_metadata(self):
        """Verify command is recorded in metadata."""
        tool = TerminalTool()
        result = await tool.execute(command="echo test")

        assert "command" in result.metadata
        assert "echo test" in result.metadata["command"]

    @pytest.mark.asyncio
    async def test_terminal_execution_time_tracked(self):
        """Verify execution time is tracked."""
        tool = TerminalTool()
        result = await tool.execute(command="sleep 0.1")

        assert result.execution_time_ms > 0
        assert result.execution_time_ms >= 100  # At least 100ms for sleep 0.1

    def test_terminal_tool_success_rate(self):
        """Test success rate tracking."""
        tool = TerminalTool()
        assert tool.success_rate == 0.0  # No executions yet

    @pytest.mark.asyncio
    async def test_terminal_tool_validates_input(self):
        """Test input validation."""
        tool = TerminalTool()

        # Valid input
        errors = tool.validate_input(command="echo test")
        assert len(errors) == 0

        # Missing required parameter
        errors = tool.validate_input()
        assert len(errors) > 0
        assert any("command" in e.lower() for e in errors)
