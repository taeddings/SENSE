"""
SENSE-v2 LLM Provider Implementations
Concrete implementations for various LLM backends.
"""

from typing import Any, Dict, List, Optional, AsyncIterator, Callable
import asyncio
import json
import logging
import os
import uuid

from sense.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    ToolCall,
)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI-compatible API provider.

    Works with:
    - OpenAI API
    - Azure OpenAI
    - Local servers with OpenAI-compatible API (LM Studio, Ollama, etc.)
    - vLLM with OpenAI-compatible server
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._client = None

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            base_url = self.config.api_base or os.environ.get("OPENAI_API_BASE")

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            self._initialized = True
            self.logger.info(f"OpenAI provider initialized (base: {base_url or 'default'})")

        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    async def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        if not self._initialized:
            await self.initialize()

        # Format messages
        formatted_messages = [msg.to_dict() for msg in messages]

        # Build request params
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
        }

        if self.config.stop:
            params["stop"] = self.config.stop

        # Add tools if provided
        if tools:
            params["tools"] = self.format_tools_for_api(tools)
            if tool_choice:
                params["tool_choice"] = tool_choice

        try:
            response = await self._client.chat.completions.create(**params)

            # Parse response
            choice = response.choices[0]
            content = choice.message.content or ""

            # Parse tool calls
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else {},
                model=response.model,
                raw_response=response,
            )

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    async def complete_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion using OpenAI API."""
        if not self._initialized:
            await self.initialize()

        formatted_messages = [msg.to_dict() for msg in messages]

        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        if tools:
            params["tools"] = self.format_tools_for_api(tools)

        try:
            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {e}")
            raise

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False

    def format_tools_for_api(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI API."""
        formatted = []
        for tool in tools:
            if "type" not in tool:
                # Convert from our schema format to OpenAI format
                formatted.append({
                    "type": "function",
                    "function": tool,
                })
            else:
                formatted.append(tool)
        return formatted


class VLLMProvider(BaseLLMProvider):
    """
    vLLM provider for local model inference.

    Optimized for:
    - AMD ROCm (RDNA 3.5)
    - 128GB UMA configurations
    - High throughput batch inference
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._llm = None
        self._tokenizer = None

    async def initialize(self) -> None:
        """Initialize vLLM engine."""
        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs

            engine_args = AsyncEngineArgs(
                model=self.config.model,
                dtype=self.config.api_base or "float16",  # Reuse api_base for dtype
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                max_model_len=self.config.max_tokens,
                trust_remote_code=True,
            )

            self._llm = AsyncLLMEngine.from_engine_args(engine_args)
            self._initialized = True
            self.logger.info(f"vLLM engine initialized: {self.config.model}")

        except ImportError:
            raise ImportError("vllm package required. Install with: pip install vllm")

    async def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using vLLM."""
        if not self._initialized:
            await self.initialize()

        from vllm import SamplingParams

        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )

        # Generate
        request_id = str(uuid.uuid4())
        results = []

        async for output in self._llm.generate(prompt, sampling_params, request_id):
            results.append(output)

        if not results:
            return LLMResponse(content="", finish_reason="error")

        final_output = results[-1]
        generated_text = final_output.outputs[0].text

        # Parse for tool calls if tools provided
        tool_calls = []
        if tools:
            tool_calls = self._parse_tool_calls(generated_text)

        return LLMResponse(
            content=generated_text,
            tool_calls=tool_calls,
            finish_reason="stop",
            model=self.config.model,
            raw_response=final_output,
        )

    async def complete_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion using vLLM."""
        if not self._initialized:
            await self.initialize()

        from vllm import SamplingParams

        prompt = self._format_chat_prompt(messages)

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )

        request_id = str(uuid.uuid4())
        prev_text = ""

        async for output in self._llm.generate(prompt, sampling_params, request_id):
            new_text = output.outputs[0].text
            delta = new_text[len(prev_text):]
            prev_text = new_text
            if delta:
                yield delta

    async def close(self) -> None:
        """Shutdown vLLM engine."""
        if self._llm:
            # vLLM cleanup
            self._llm = None
            self._initialized = False

    def _format_chat_prompt(self, messages: List[LLMMessage]) -> str:
        """Format messages into a chat prompt string."""
        # Simple ChatML format - override for model-specific formats
        prompt_parts = []
        for msg in messages:
            if msg.role.value == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role.value == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role.value == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls from generated text."""
        tool_calls = []
        # Simple JSON extraction - enhance for production
        try:
            if "```json" in text:
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                if json_end > json_start:
                    json_str = text[json_start:json_end].strip()
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "name" in data:
                        tool_calls.append(ToolCall(
                            id=str(uuid.uuid4()),
                            name=data["name"],
                            arguments=data.get("arguments", {}),
                        ))
        except json.JSONDecodeError:
            pass
        return tool_calls


class MockProvider(BaseLLMProvider):
    """
    Mock LLM provider for testing.

    Allows configuring responses for predictable test behavior.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._responses: List[LLMResponse] = []
        self._response_index = 0
        self._call_history: List[Dict[str, Any]] = []
        self._default_response = "Mock response"
        self._response_callback: Optional[Callable] = None

    async def initialize(self) -> None:
        """Initialize mock provider."""
        self._initialized = True
        self.logger.info("Mock provider initialized")

    async def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Return configured mock response."""
        # Record the call
        self._call_history.append({
            "messages": [m.to_dict() for m in messages],
            "tools": tools,
            "tool_choice": tool_choice,
            "kwargs": kwargs,
        })

        # Use callback if set
        if self._response_callback:
            return self._response_callback(messages, tools, tool_choice)

        # Use queued responses
        if self._responses:
            if self._response_index < len(self._responses):
                response = self._responses[self._response_index]
                self._response_index += 1
                return response

        # Default response
        return LLMResponse(
            content=self._default_response,
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    async def complete_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream mock response word by word."""
        response = await self.complete(messages, tools, **kwargs)
        words = response.content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)  # Simulate streaming delay

    async def close(self) -> None:
        """Close mock provider."""
        self._initialized = False

    # Test helper methods

    def set_responses(self, responses: List[LLMResponse]) -> None:
        """Queue responses to return in order."""
        self._responses = responses
        self._response_index = 0

    def add_response(self, response: LLMResponse) -> None:
        """Add a response to the queue."""
        self._responses.append(response)

    def set_default_response(self, content: str) -> None:
        """Set default response content."""
        self._default_response = content

    def set_response_callback(
        self,
        callback: Callable[[List[LLMMessage], Optional[List], Optional[str]], LLMResponse]
    ) -> None:
        """Set a callback to generate responses dynamically."""
        self._response_callback = callback

    def set_tool_call_response(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Configure a response that includes a tool call."""
        self._responses.append(LLMResponse(
            content="",
            tool_calls=[ToolCall(
                id=str(uuid.uuid4()),
                name=tool_name,
                arguments=arguments,
            )],
            model="mock-model",
        ))

    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of all calls made to this provider."""
        return self._call_history

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the most recent call."""
        return self._call_history[-1] if self._call_history else None

    def reset(self) -> None:
        """Reset all state."""
        self._responses = []
        self._response_index = 0
        self._call_history = []
        self._response_callback = None
