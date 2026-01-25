"""
SENSE-v2 LLM Base Classes
Abstract base for LLM providers with tool calling support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from enum import Enum
import logging


class LLMRole(Enum):
    """Message roles for LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""
    role: LLMRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        msg = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        return cls(role=LLMRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "LLMMessage":
        return cls(role=LLMRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List] = None) -> "LLMMessage":
        return cls(role=LLMRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, content: str, tool_call_id: str, name: str) -> "LLMMessage":
        return cls(role=LLMRole.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            }
        }


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw_response: Optional[Any] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def to_message(self) -> LLMMessage:
        """Convert response to an assistant message."""
        tool_calls_dict = [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None
        return LLMMessage.assistant(self.content, tool_calls=tool_calls_dict)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

    # Provider-specific
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3

    # vLLM specific
    vllm_gpu_memory_utilization: float = 0.85
    vllm_tensor_parallel_size: int = 1


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Supports:
    - Chat completions
    - Tool/function calling
    - Streaming responses
    - Multiple backends (vLLM, OpenAI, Anthropic)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load models, connect to API, etc.)."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools in OpenAI function format
            tool_choice: "auto", "none", or specific tool name
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content and optional tool calls
        """
        pass

    @abstractmethod
    async def complete_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream a completion from the LLM.

        Yields:
            String chunks of the response
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def format_tools_for_api(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for the specific API.
        Override in subclasses for provider-specific formatting.
        """
        return tools

    async def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Convenience method for single-turn chat.

        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            tools: Optional tools to make available

        Returns:
            LLMResponse
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage.system(system_prompt))
        messages.append(LLMMessage.user(user_message))

        return await self.complete(messages, tools=tools)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return self.__class__.__name__

    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized
