"""
SENSE-v2 Schema Definitions
All high-level functions exposed as Schema-based Python Tools.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json
from datetime import datetime


class ToolResultStatus(Enum):
    """Status codes for tool execution results."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RETRY = "retry"


class MessageRole(Enum):
    """Roles for agent messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolParameter:
    """Definition of a single tool parameter."""
    name: str
    param_type: str  # "string", "integer", "float", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.param_type,
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolSchema:
    """
    Schema definition for a SENSE-v2 tool.
    All high-level functions must be exposed through this schema.
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: str = "object"
    returns_description: str = ""
    category: str = "general"
    requires_confirmation: bool = False
    timeout_seconds: int = 60
    max_retries: int = 3

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use format."""
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param in self.parameters:
            input_schema["properties"][param.name] = param.to_schema()
            if param.required:
                input_schema["required"].append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema,
        }

    def validate_input(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate input against schema, return list of errors."""
        errors = []

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in inputs:
                errors.append(f"Missing required parameter: {param.name}")
                continue

            if param.name in inputs:
                value = inputs[param.name]

                # Type validation
                type_map = {
                    "string": str,
                    "integer": int,
                    "float": (int, float),
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }
                expected_type = type_map.get(param.param_type)
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Parameter {param.name} must be {param.param_type}")

                # Enum validation
                if param.enum and value not in param.enum:
                    errors.append(f"Parameter {param.name} must be one of {param.enum}")

                # Range validation
                if param.min_value is not None and isinstance(value, (int, float)):
                    if value < param.min_value:
                        errors.append(f"Parameter {param.name} must be >= {param.min_value}")
                if param.max_value is not None and isinstance(value, (int, float)):
                    if value > param.max_value:
                        errors.append(f"Parameter {param.name} must be <= {param.max_value}")

        return errors


@dataclass
class ToolResult:
    """
    Result from a tool execution.
    Includes feedback mechanism for self-correction loops.
    """
    status: ToolResultStatus
    output: Any = None
    error: Optional[str] = None
    stderr: Optional[str] = None
    stdout: Optional[str] = None
    exit_code: Optional[int] = None
    execution_time_ms: int = 0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if the tool execution was successful."""
        return self.status == ToolResultStatus.SUCCESS

    @property
    def should_retry(self) -> bool:
        """Determine if the tool should be retried based on error analysis."""
        if self.status == ToolResultStatus.RETRY:
            return True
        if self.stderr and self._is_recoverable_error():
            return True
        return False

    def _is_recoverable_error(self) -> bool:
        """Analyze stderr to determine if error is recoverable."""
        if not self.stderr:
            return False

        recoverable_patterns = [
            "timeout",
            "connection refused",
            "resource temporarily unavailable",
            "too many open files",
            "memory allocation failed",
        ]

        stderr_lower = self.stderr.lower()
        return any(pattern in stderr_lower for pattern in recoverable_patterns)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "exit_code": self.exit_code,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }

    @classmethod
    def success(cls, output: Any, **kwargs) -> "ToolResult":
        """Create a successful result."""
        return cls(status=ToolResultStatus.SUCCESS, output=output, **kwargs)

    @classmethod
    def error(cls, error_msg: str, stderr: Optional[str] = None, **kwargs) -> "ToolResult":
        """Create an error result."""
        return cls(status=ToolResultStatus.ERROR, error=error_msg, stderr=stderr, **kwargs)

    @classmethod
    def from_process(cls, returncode: int, stdout: str, stderr: str, **kwargs) -> "ToolResult":
        """Create result from subprocess execution."""
        status = ToolResultStatus.SUCCESS if returncode == 0 else ToolResultStatus.ERROR
        return cls(
            status=status,
            output=stdout if returncode == 0 else None,
            error=stderr if returncode != 0 else None,
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
            **kwargs
        )


@dataclass
class AgentMessage:
    """
    Message structure for agent communication.
    Supports multi-turn conversations with tool use.
    """
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format compatible with LLM APIs."""
        msg = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    @classmethod
    def system(cls, content: str, **kwargs) -> "AgentMessage":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs) -> "AgentMessage":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List] = None, **kwargs) -> "AgentMessage":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls, **kwargs)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str, **kwargs) -> "AgentMessage":
        """Create a tool result message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            **kwargs
        )

    def token_estimate(self) -> int:
        """Estimate token count for this message."""
        # Rough estimate: ~4 characters per token
        return len(self.content) // 4 + 10  # +10 for role/formatting overhead


@dataclass
class RewardSignal:
    """
    Reward signal for evolutionary learning.
    Based on Unit Test Success or Terminal Exit Codes.
    """
    value: float  # 0.0 to 1.0 for normalized rewards
    binary: bool  # Whether this is a binary (pass/fail) or scalar reward
    source: str   # "unit_test", "exit_code", "combined"

    # Detailed breakdown
    unit_test_passed: Optional[int] = None
    unit_test_total: Optional[int] = None
    exit_code: Optional[int] = None

    # Metadata for learning
    task_id: Optional[str] = None
    difficulty_level: float = 1.0

    @classmethod
    def from_unit_tests(cls, passed: int, total: int, **kwargs) -> "RewardSignal":
        """Create reward from unit test results."""
        if total == 0:
            value = 0.0
        else:
            value = passed / total
        return cls(
            value=value,
            binary=passed == total,
            source="unit_test",
            unit_test_passed=passed,
            unit_test_total=total,
            **kwargs
        )

    @classmethod
    def from_exit_code(cls, exit_code: int, **kwargs) -> "RewardSignal":
        """Create reward from process exit code."""
        return cls(
            value=1.0 if exit_code == 0 else 0.0,
            binary=True,
            source="exit_code",
            exit_code=exit_code,
            **kwargs
        )

    @classmethod
    def combined(
        cls,
        unit_test_passed: int,
        unit_test_total: int,
        exit_code: int,
        unit_test_weight: float = 0.6,
        exit_code_weight: float = 0.4,
        **kwargs
    ) -> "RewardSignal":
        """Create combined reward from multiple sources."""
        test_reward = unit_test_passed / unit_test_total if unit_test_total > 0 else 0.0
        exit_reward = 1.0 if exit_code == 0 else 0.0

        value = (test_reward * unit_test_weight) + (exit_reward * exit_code_weight)

        return cls(
            value=value,
            binary=False,
            source="combined",
            unit_test_passed=unit_test_passed,
            unit_test_total=unit_test_total,
            exit_code=exit_code,
            **kwargs
        )
