"""
Tests for SENSE-v2 LLM Providers
Tests mock provider and provider interfaces.
"""

import pytest
import asyncio
from typing import List, Dict, Any, Optional

from sense.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMRole,
    ToolCall,
)
from sense.llm.providers import MockProvider, OpenAIProvider, VLLMProvider


class TestLLMMessage:
    """Test LLMMessage class."""

    def test_message_creation(self):
        """Test creating messages."""
        system_msg = LLMMessage.system("You are a helpful assistant")
        assert system_msg.role == LLMRole.SYSTEM
        assert system_msg.content == "You are a helpful assistant"

        user_msg = LLMMessage.user("Hello")
        assert user_msg.role == LLMRole.USER

        assistant_msg = LLMMessage.assistant("Hi there!")
        assert assistant_msg.role == LLMRole.ASSISTANT

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = LLMMessage.user("Test message")
        d = msg.to_dict()

        assert d["role"] == "user"
        assert d["content"] == "Test message"

    def test_tool_result_message(self):
        """Test tool result message creation."""
        msg = LLMMessage.tool_result(
            content='{"result": "success"}',
            tool_call_id="call_123",
            name="my_tool"
        )

        assert msg.role == LLMRole.TOOL
        assert msg.tool_call_id == "call_123"
        assert msg.name == "my_tool"


class TestToolCall:
    """Test ToolCall class."""

    def test_tool_call_creation(self):
        """Test creating tool calls."""
        tc = ToolCall(
            id="call_abc",
            name="get_weather",
            arguments={"city": "Seattle"}
        )

        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments["city"] == "Seattle"

    def test_tool_call_to_dict(self):
        """Test tool call serialization."""
        tc = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"}
        )
        d = tc.to_dict()

        assert d["id"] == "call_123"
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert d["function"]["arguments"]["query"] == "test"


class TestLLMResponse:
    """Test LLMResponse class."""

    def test_response_creation(self):
        """Test creating responses."""
        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
            finish_reason="stop"
        )

        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.has_tool_calls is False

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        tc = ToolCall(id="1", name="test", arguments={})
        response = LLMResponse(
            content="",
            tool_calls=[tc],
            model="gpt-4"
        )

        assert response.has_tool_calls is True
        assert len(response.tool_calls) == 1

    def test_response_to_message(self):
        """Test converting response to message."""
        response = LLMResponse(content="Test response", model="gpt-4")
        msg = response.to_message()

        assert msg.role == LLMRole.ASSISTANT
        assert msg.content == "Test response"


class TestLLMConfig:
    """Test LLMConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 60

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            model="llama-2-7b",
            temperature=0.5,
            max_tokens=2048,
            api_key="test-key"
        )

        assert config.model == "llama-2-7b"
        assert config.temperature == 0.5
        assert config.api_key == "test-key"


class TestMockProvider:
    """Test MockProvider implementation."""

    @pytest.mark.asyncio
    async def test_mock_provider_initialization(self):
        """Test initializing mock provider."""
        provider = MockProvider()
        await provider.initialize()

        assert provider.is_initialized()
        assert provider.get_provider_name() == "MockProvider"

        await provider.close()
        assert not provider.is_initialized()

    @pytest.mark.asyncio
    async def test_mock_provider_default_response(self):
        """Test default mock response."""
        async with MockProvider() as provider:
            messages = [LLMMessage.user("Hello")]
            response = await provider.complete(messages)

            assert response.content == "Mock response"
            assert response.model == "mock-model"

    @pytest.mark.asyncio
    async def test_mock_provider_custom_response(self):
        """Test setting custom response."""
        provider = MockProvider()
        await provider.initialize()

        provider.set_default_response("Custom response")

        messages = [LLMMessage.user("Test")]
        response = await provider.complete(messages)

        assert response.content == "Custom response"
        await provider.close()

    @pytest.mark.asyncio
    async def test_mock_provider_queued_responses(self):
        """Test queued responses in order."""
        provider = MockProvider()
        await provider.initialize()

        provider.set_responses([
            LLMResponse(content="First", model="mock"),
            LLMResponse(content="Second", model="mock"),
            LLMResponse(content="Third", model="mock"),
        ])

        messages = [LLMMessage.user("Test")]

        r1 = await provider.complete(messages)
        assert r1.content == "First"

        r2 = await provider.complete(messages)
        assert r2.content == "Second"

        r3 = await provider.complete(messages)
        assert r3.content == "Third"

        await provider.close()

    @pytest.mark.asyncio
    async def test_mock_provider_tool_call(self):
        """Test mock tool call response."""
        provider = MockProvider()
        await provider.initialize()

        provider.set_tool_call_response(
            tool_name="get_weather",
            arguments={"city": "Seattle"}
        )

        messages = [LLMMessage.user("What's the weather?")]
        tools = [{"name": "get_weather", "parameters": {}}]

        response = await provider.complete(messages, tools=tools)

        assert response.has_tool_calls
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments["city"] == "Seattle"

        await provider.close()

    @pytest.mark.asyncio
    async def test_mock_provider_call_history(self):
        """Test call history tracking."""
        provider = MockProvider()
        await provider.initialize()

        messages1 = [LLMMessage.user("First call")]
        messages2 = [LLMMessage.user("Second call")]

        await provider.complete(messages1)
        await provider.complete(messages2)

        history = provider.get_call_history()
        assert len(history) == 2

        last_call = provider.get_last_call()
        assert last_call["messages"][0]["content"] == "Second call"

        await provider.close()

    @pytest.mark.asyncio
    async def test_mock_provider_response_callback(self):
        """Test dynamic response callback."""
        provider = MockProvider()
        await provider.initialize()

        def callback(messages, tools, tool_choice):
            user_msg = messages[-1].content if messages else ""
            return LLMResponse(
                content=f"Echo: {user_msg}",
                model="callback-mock"
            )

        provider.set_response_callback(callback)

        messages = [LLMMessage.user("Hello world")]
        response = await provider.complete(messages)

        assert response.content == "Echo: Hello world"
        assert response.model == "callback-mock"

        await provider.close()

    @pytest.mark.asyncio
    async def test_mock_provider_streaming(self):
        """Test streaming responses."""
        provider = MockProvider()
        await provider.initialize()

        provider.set_default_response("This is a streaming test")

        messages = [LLMMessage.user("Stream test")]
        chunks = []

        async for chunk in provider.complete_stream(messages):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert "streaming" in full_response.lower()

        await provider.close()

    @pytest.mark.asyncio
    async def test_mock_provider_reset(self):
        """Test resetting provider state."""
        provider = MockProvider()
        await provider.initialize()

        provider.set_default_response("Custom")
        provider.add_response(LLMResponse(content="Queued", model="mock"))
        await provider.complete([LLMMessage.user("Test")])

        assert len(provider.get_call_history()) == 1

        provider.reset()

        assert len(provider.get_call_history()) == 0
        assert len(provider._responses) == 0

        await provider.close()

    @pytest.mark.asyncio
    async def test_mock_provider_chat_convenience(self):
        """Test chat convenience method."""
        provider = MockProvider()
        await provider.initialize()

        provider.set_default_response("Hello!")

        response = await provider.chat(
            user_message="Hi there",
            system_prompt="Be helpful"
        )

        assert response.content == "Hello!"

        last_call = provider.get_last_call()
        assert len(last_call["messages"]) == 2
        assert last_call["messages"][0]["role"] == "system"
        assert last_call["messages"][1]["role"] == "user"

        await provider.close()


class TestProviderInterfaces:
    """Test that providers implement the interface correctly."""

    def test_openai_provider_exists(self):
        """Test OpenAIProvider can be instantiated."""
        config = LLMConfig(model="gpt-4", api_key="test")
        provider = OpenAIProvider(config)

        assert provider.config.model == "gpt-4"
        assert isinstance(provider, BaseLLMProvider)

    def test_vllm_provider_exists(self):
        """Test VLLMProvider can be instantiated."""
        config = LLMConfig(
            model="meta-llama/Llama-2-7b-hf",
            vllm_gpu_memory_utilization=0.85
        )
        provider = VLLMProvider(config)

        assert provider.config.model == "meta-llama/Llama-2-7b-hf"
        assert isinstance(provider, BaseLLMProvider)

    def test_all_providers_have_required_methods(self):
        """Test all providers have required methods."""
        providers = [
            MockProvider(),
            OpenAIProvider(),
            VLLMProvider(),
        ]

        for provider in providers:
            assert hasattr(provider, "initialize")
            assert hasattr(provider, "complete")
            assert hasattr(provider, "complete_stream")
            assert hasattr(provider, "close")
            assert hasattr(provider, "chat")
            assert callable(provider.initialize)
            assert callable(provider.complete)


class TestToolFormatting:
    """Test tool formatting for different providers."""

    def test_openai_tool_formatting(self):
        """Test OpenAI tool format conversion."""
        provider = OpenAIProvider()

        tools = [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            }
        ]

        formatted = provider.format_tools_for_api(tools)

        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "get_weather"

    def test_already_formatted_tools(self):
        """Test tools that are already in correct format."""
        provider = OpenAIProvider()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {}
                }
            }
        ]

        formatted = provider.format_tools_for_api(tools)

        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "search"


class TestProviderContextManager:
    """Test async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using provider as context manager."""
        async with MockProvider() as provider:
            assert provider.is_initialized()
            response = await provider.chat("Hello")
            assert response.content is not None

        # After exiting, should be closed
        assert not provider.is_initialized()

    @pytest.mark.asyncio
    async def test_context_manager_with_config(self):
        """Test context manager with custom config."""
        config = LLMConfig(model="test-model", temperature=0.5)

        async with MockProvider(config) as provider:
            assert provider.config.model == "test-model"
            assert provider.config.temperature == 0.5
