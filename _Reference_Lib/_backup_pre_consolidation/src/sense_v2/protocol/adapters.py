"""
SENSE Protocol Adapters

Provides adapters for converting between high-level domain objects
and binary protocol messages.

ADAPTER PATTERN:
================
Adapters bridge the gap between:
- Domain objects (AgentMessage, ToolResult, etc.)
- Wire format (SENSEMessage with binary payloads)

This separation allows:
1. Domain logic to remain clean and protocol-agnostic
2. Protocol format to evolve without breaking domain code
3. Easy testing of domain logic without protocol concerns

USAGE:
======
    # Convert AgentMessage to protocol
    agent_msg = AgentMessage.user("Hello!")
    sense_msg = AgentMessageAdapter.to_protocol(agent_msg)
    wire_bytes = sense_msg.to_bytes()

    # Convert from protocol
    sense_msg = SENSEMessage.parse(wire_bytes)
    agent_msg = AgentMessageAdapter.from_protocol(sense_msg)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from sense_v2.core.schemas import (
    AgentMessage,
    MessageRole,
    ToolResult,
    ToolResultStatus,
)
from .constants import (
    METHOD_ID_AGENT_SYSTEM,
    METHOD_ID_AGENT_USER,
    METHOD_ID_AGENT_ASSISTANT,
    METHOD_ID_AGENT_TOOL,
    METHOD_ID_TOOL_CALL,
    METHOD_ID_TOOL_RESULT,
    FLAG_PAYLOAD_MSGPACK,
)
from .message import SENSEMessage
from .exceptions import MalformedMessageError


class AgentMessageAdapter:
    """
    Adapter for converting between AgentMessage and SENSEMessage.

    AgentMessage is the high-level domain object used throughout
    the SENSE system. This adapter handles serialization to/from
    the binary protocol format.

    Method ID Mapping:
        MessageRole.SYSTEM    → 0x0101
        MessageRole.USER      → 0x0102
        MessageRole.ASSISTANT → 0x0103
        MessageRole.TOOL      → 0x0104
    """

    # Method ID mappings for each message role
    ROLE_TO_METHOD_ID: Dict[MessageRole, int] = {
        MessageRole.SYSTEM: METHOD_ID_AGENT_SYSTEM,
        MessageRole.USER: METHOD_ID_AGENT_USER,
        MessageRole.ASSISTANT: METHOD_ID_AGENT_ASSISTANT,
        MessageRole.TOOL: METHOD_ID_AGENT_TOOL,
    }

    METHOD_ID_TO_ROLE: Dict[int, MessageRole] = {
        v: k for k, v in ROLE_TO_METHOD_ID.items()
    }

    @classmethod
    def to_protocol(
        cls,
        msg: AgentMessage,
        message_id: Optional[int] = None,
    ) -> SENSEMessage:
        """
        Convert AgentMessage to binary SENSEMessage.

        Args:
            msg: The AgentMessage to convert
            message_id: Optional message ID (auto-generated if None)

        Returns:
            SENSEMessage ready for wire transmission
        """
        method_id = cls.ROLE_TO_METHOD_ID.get(msg.role, METHOD_ID_AGENT_USER)

        # Build payload dict
        payload = {
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
        }

        if msg.tool_calls:
            payload["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            payload["tool_call_id"] = msg.tool_call_id
        if msg.name:
            payload["name"] = msg.name
        if msg.metadata:
            payload["metadata"] = msg.metadata

        return SENSEMessage.create_request(
            method_id=method_id,
            payload=payload,
            message_id=message_id,
            flags=FLAG_PAYLOAD_MSGPACK,
        )

    @classmethod
    def from_protocol(cls, msg: SENSEMessage) -> AgentMessage:
        """
        Convert binary SENSEMessage to AgentMessage.

        Args:
            msg: The SENSEMessage to convert

        Returns:
            AgentMessage instance

        Raises:
            MalformedMessageError: If message format is invalid
        """
        role = cls.METHOD_ID_TO_ROLE.get(msg.method_id)
        if role is None:
            raise MalformedMessageError(
                f"Unknown agent method ID: 0x{msg.method_id:04X}",
                field="method_id"
            )

        payload = msg.get_payload()
        if not isinstance(payload, dict):
            raise MalformedMessageError(
                f"Expected dict payload, got {type(payload)}",
                field="payload"
            )

        # Parse timestamp
        timestamp_str = payload.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        return AgentMessage(
            role=role,
            content=payload.get("content", ""),
            tool_calls=payload.get("tool_calls"),
            tool_call_id=payload.get("tool_call_id"),
            name=payload.get("name"),
            timestamp=timestamp,
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def to_bytes(cls, msg: AgentMessage) -> bytes:
        """
        Convenience method to serialize AgentMessage directly to bytes.

        Args:
            msg: AgentMessage to serialize

        Returns:
            Wire format bytes
        """
        return cls.to_protocol(msg).to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> AgentMessage:
        """
        Convenience method to deserialize AgentMessage from bytes.

        Args:
            data: Wire format bytes

        Returns:
            AgentMessage instance
        """
        sense_msg = SENSEMessage.parse(data)
        return cls.from_protocol(sense_msg)


class ToolResultAdapter:
    """
    Adapter for converting between ToolResult and SENSEMessage.

    ToolResult represents the outcome of a tool execution.
    This adapter serializes it for transmission over the protocol.
    """

    @classmethod
    def to_protocol(
        cls,
        result: ToolResult,
        request_message_id: int,
    ) -> SENSEMessage:
        """
        Convert ToolResult to binary SENSEMessage.

        The result is sent as a response to the original tool call request.

        Args:
            result: The ToolResult to convert
            request_message_id: ID of the original request

        Returns:
            SENSEMessage response
        """
        payload = {
            "status": result.status.value,
            "output": result.output,
            "error": result.error,
            "stderr": result.stderr,
            "stdout": result.stdout,
            "exit_code": result.exit_code,
            "execution_time_ms": result.execution_time_ms,
            "retry_count": result.retry_count,
            "metadata": result.metadata,
        }

        # Create a minimal request to respond to
        class _Request:
            message_id = request_message_id
            method_id = METHOD_ID_TOOL_CALL
            flags = FLAG_PAYLOAD_MSGPACK

            @property
            def header(self):
                return self

        return SENSEMessage.create_response(
            request=_Request(),
            payload=payload,
            is_error=not result.is_success,
        )

    @classmethod
    def from_protocol(cls, msg: SENSEMessage) -> ToolResult:
        """
        Convert binary SENSEMessage to ToolResult.

        Args:
            msg: The SENSEMessage to convert

        Returns:
            ToolResult instance

        Raises:
            MalformedMessageError: If message format is invalid
        """
        payload = msg.get_payload()
        if not isinstance(payload, dict):
            raise MalformedMessageError(
                f"Expected dict payload, got {type(payload)}",
                field="payload"
            )

        status_str = payload.get("status", "error")
        try:
            status = ToolResultStatus(status_str)
        except ValueError:
            status = ToolResultStatus.ERROR

        return ToolResult(
            status=status,
            output=payload.get("output"),
            error=payload.get("error"),
            stderr=payload.get("stderr"),
            stdout=payload.get("stdout"),
            exit_code=payload.get("exit_code"),
            execution_time_ms=payload.get("execution_time_ms", 0),
            retry_count=payload.get("retry_count", 0),
            metadata=payload.get("metadata", {}),
        )


class ConversationAdapter:
    """
    Adapter for serializing/deserializing entire conversations.

    A conversation is a list of AgentMessages that can be
    serialized as a single batch for efficiency.
    """

    @classmethod
    def to_payload(cls, messages: List[AgentMessage]) -> Dict[str, Any]:
        """
        Convert a list of messages to a payload dict.

        Args:
            messages: List of AgentMessages

        Returns:
            Dict suitable for protocol payload
        """
        return {
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                }
                for msg in messages
            ]
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> List[AgentMessage]:
        """
        Convert a payload dict to a list of messages.

        Args:
            payload: Dict from protocol payload

        Returns:
            List of AgentMessages
        """
        messages = []
        for msg_data in payload.get("messages", []):
            role = MessageRole(msg_data.get("role", "user"))

            timestamp_str = msg_data.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            messages.append(AgentMessage(
                role=role,
                content=msg_data.get("content", ""),
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
                name=msg_data.get("name"),
                timestamp=timestamp,
                metadata=msg_data.get("metadata", {}),
            ))

        return messages


def extend_agent_message():
    """
    Monkey-patch AgentMessage with binary serialization methods.

    This adds to_binary() and from_binary() methods to AgentMessage
    for convenient protocol integration.

    Call this once at module initialization.
    """

    def to_binary(self: AgentMessage) -> bytes:
        """Serialize AgentMessage to DRGN binary format."""
        return AgentMessageAdapter.to_bytes(self)

    def from_binary(cls, data: bytes) -> AgentMessage:
        """Deserialize AgentMessage from DRGN binary format."""
        return AgentMessageAdapter.from_bytes(data)

    AgentMessage.to_binary = to_binary
    AgentMessage.from_binary = classmethod(from_binary)
