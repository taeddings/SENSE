"""
SENSE Protocol High-Level Message API

Provides the SENSEMessage class - the primary interface for creating,
parsing, and manipulating protocol messages.

DESIGN PHILOSOPHY:
==================
SENSEMessage abstracts the low-level binary details:
- Header construction/parsing
- CRC32 computation/verification
- Serialization format selection
- Compression handling

Users work with Python objects, not bytes.

USAGE EXAMPLES:
===============
    # Creating a request
    msg = SENSEMessage.create_request(
        method_id=METHOD_ID_AGENT_USER,
        payload={"content": "Hello, world!"},
        message_id=12345,
    )
    wire_bytes = msg.to_bytes()

    # Parsing a response
    msg = SENSEMessage.parse(wire_bytes)
    print(msg.payload)  # {'content': 'Hello, world!'}

    # Streaming (partial) parsing
    header = SENSEMessage.parse_header(partial_data)
    if len(data) >= header.total_bytes + 4:
        msg = SENSEMessage.parse(data)
"""

import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Tuple

from .constants import (
    MAGIC_SIGNATURE_INT,
    PROTOCOL_VERSION,
    HEADER_SIZE,
    HEADER_FORMAT,
    MAX_MESSAGE_SIZE,
    FLAG_PAYLOAD_MSGPACK,
    FLAG_PAYLOAD_JSON,
    FLAG_PAYLOAD_STRING,
    FLAG_COMPRESSED,
    FLAG_RESPONSE,
    FLAG_ERROR,
    is_response,
    is_error,
    make_response_flags,
)
from .header import DRGNHeader, compute_crc32
from .parser import BinaryParser
from .serializers import (
    serialize_payload,
    deserialize_payload,
    get_default_serializer,
    MSGPACK_AVAILABLE,
)
from .exceptions import (
    IncompleteMessageError,
    MessageTooLargeError,
    CRCMismatchError,
    MalformedMessageError,
    IntegrityError,
)


# Thread-safe message ID counter
_message_id_lock = threading.Lock()
_message_id_counter = 0


def generate_message_id() -> int:
    """
    Generate a unique message ID.

    Thread-safe counter that wraps at 64-bit limit.
    In practice, wrapping is unlikely (would take billions of years
    at 1 million messages per second).

    Returns:
        Unique 64-bit message ID
    """
    global _message_id_counter
    with _message_id_lock:
        _message_id_counter = (_message_id_counter + 1) & 0xFFFFFFFFFFFFFFFF
        return _message_id_counter


@dataclass
class SENSEMessage:
    """
    High-level SENSE protocol message.

    This class combines a header with a payload and provides
    convenient methods for creating, serializing, and parsing
    protocol messages.

    Attributes:
        header: The DRGN header containing metadata
        payload: The deserialized payload data
        raw_payload: The raw payload bytes (optional, for lazy parsing)
    """

    header: DRGNHeader
    payload: Any = None
    raw_payload: Optional[bytes] = None

    @property
    def method_id(self) -> int:
        """Get the method ID from header."""
        return self.header.method_id

    @property
    def message_id(self) -> int:
        """Get the message ID from header."""
        return self.header.message_id

    @property
    def flags(self) -> int:
        """Get the flags from header."""
        return self.header.flags

    @property
    def is_response(self) -> bool:
        """Check if this is a response message."""
        return is_response(self.header.flags)

    @property
    def is_error(self) -> bool:
        """Check if this is an error response."""
        return is_error(self.header.flags)

    @property
    def is_request(self) -> bool:
        """Check if this is a request message."""
        return not is_response(self.header.flags)

    def get_payload(self) -> Any:
        """
        Get the payload, deserializing if necessary.

        Supports lazy parsing - if raw_payload is set but payload
        is None, deserializes on first access.

        Returns:
            Deserialized payload
        """
        if self.payload is None and self.raw_payload is not None:
            # Lazy deserialization
            self.payload = deserialize_payload(self.raw_payload, self.header.flags)
        return self.payload

    def to_bytes(self) -> bytes:
        """
        Serialize the complete message to wire format.

        Returns:
            Complete message as bytes (header + payload)
        """
        # Serialize payload if we have structured data
        if self.raw_payload is not None:
            payload_bytes = self.raw_payload
        else:
            payload_bytes = serialize_payload(self.payload, self.header.flags)

        # Create header with correct CRC and size
        header = DRGNHeader.create(
            method_id=self.header.method_id,
            payload_size=len(payload_bytes),
            flags=self.header.flags,
            message_id=self.header.message_id,
            crc32=compute_crc32(payload_bytes),
        )

        return header.pack() + payload_bytes

    @classmethod
    def create_request(
        cls,
        method_id: int,
        payload: Any = None,
        message_id: Optional[int] = None,
        flags: Optional[int] = None,
        compress: bool = False,
    ) -> 'SENSEMessage':
        """
        Factory method to create a request message.

        Args:
            method_id: The RPC method identifier
            payload: Request payload (dict, list, string, etc.)
            message_id: Optional message ID (auto-generated if None)
            flags: Optional flags (default based on payload type)
            compress: Whether to compress the payload

        Returns:
            New SENSEMessage configured as a request
        """
        if message_id is None:
            message_id = generate_message_id()

        if flags is None:
            flags = _infer_flags(payload)

        if compress:
            flags |= FLAG_COMPRESSED

        # Serialize to get size for header
        payload_bytes = serialize_payload(payload, flags)

        header = DRGNHeader.create(
            method_id=method_id,
            payload_size=len(payload_bytes),
            flags=flags,
            message_id=message_id,
            crc32=compute_crc32(payload_bytes),
        )

        return cls(
            header=header,
            payload=payload,
            raw_payload=payload_bytes,
        )

    @classmethod
    def create_response(
        cls,
        request: 'SENSEMessage',
        payload: Any = None,
        is_error: bool = False,
        flags: Optional[int] = None,
    ) -> 'SENSEMessage':
        """
        Factory method to create a response to a request.

        The response inherits the message_id from the request
        for correlation, and sets the RESPONSE flag.

        Args:
            request: The original request message
            payload: Response payload
            is_error: Whether this is an error response
            flags: Optional flags (defaults to request flags with RESPONSE set)

        Returns:
            New SENSEMessage configured as a response
        """
        if flags is None:
            flags = make_response_flags(request.header.flags, is_error)
        else:
            flags |= FLAG_RESPONSE
            if is_error:
                flags |= FLAG_ERROR

        # Serialize to get size for header
        payload_bytes = serialize_payload(payload, flags)

        header = DRGNHeader.create(
            method_id=request.header.method_id,
            payload_size=len(payload_bytes),
            flags=flags,
            message_id=request.header.message_id,  # Same ID for correlation
            crc32=compute_crc32(payload_bytes),
        )

        return cls(
            header=header,
            payload=payload,
            raw_payload=payload_bytes,
        )

    @classmethod
    def create_error_response(
        cls,
        request: 'SENSEMessage',
        error_message: str,
        error_code: Optional[int] = None,
    ) -> 'SENSEMessage':
        """
        Factory method to create an error response.

        Args:
            request: The original request
            error_message: Human-readable error description
            error_code: Optional numeric error code

        Returns:
            Error response message
        """
        payload = {
            "error": error_message,
        }
        if error_code is not None:
            payload["error_code"] = error_code

        return cls.create_response(request, payload, is_error=True)

    @classmethod
    def parse(
        cls,
        data: Union[bytes, bytearray, memoryview],
        verify_crc: bool = True,
        lazy_payload: bool = False,
    ) -> 'SENSEMessage':
        """
        Parse a complete message from wire format.

        Args:
            data: Complete message bytes (header + payload)
            verify_crc: Whether to verify CRC32 checksum
            lazy_payload: If True, defer payload deserialization

        Returns:
            Parsed SENSEMessage

        Raises:
            IncompleteMessageError: If data is too short
            CRCMismatchError: If CRC verification fails
            MalformedMessageError: If message structure is invalid
        """
        if len(data) < HEADER_SIZE:
            raise IncompleteMessageError(
                "Data too short for header",
                needed=HEADER_SIZE,
                available=len(data),
            )

        # Parse header
        header = DRGNHeader.unpack(data)

        # Calculate expected total size
        total_size = header.total_bytes + 4  # +4 for signature field

        if len(data) < total_size:
            raise IncompleteMessageError(
                "Incomplete message",
                needed=total_size,
                available=len(data),
            )

        # Extract payload
        payload_bytes = bytes(data[HEADER_SIZE:total_size])

        # Verify CRC if requested
        if verify_crc:
            computed_crc = compute_crc32(payload_bytes)
            if computed_crc != header.crc32:
                raise CRCMismatchError(header.crc32, computed_crc)

        # Parse or defer payload
        if lazy_payload:
            return cls(
                header=header,
                payload=None,
                raw_payload=payload_bytes,
            )
        else:
            payload = deserialize_payload(payload_bytes, header.flags)
            return cls(
                header=header,
                payload=payload,
                raw_payload=payload_bytes,
            )

    @classmethod
    def parse_header(
        cls,
        data: Union[bytes, bytearray, memoryview],
    ) -> DRGNHeader:
        """
        Parse only the header (for streaming/partial reads).

        Useful for determining message size before reading the
        complete payload.

        Args:
            data: At least 29 bytes of header data

        Returns:
            Parsed DRGNHeader
        """
        return DRGNHeader.unpack(data)

    @classmethod
    def try_parse(
        cls,
        data: Union[bytes, bytearray, memoryview],
        verify_crc: bool = True,
    ) -> Tuple[Optional['SENSEMessage'], int]:
        """
        Try to parse a message, returning None if incomplete.

        Useful for streaming where you accumulate bytes until
        a complete message is available.

        Args:
            data: Buffer that may contain a complete message
            verify_crc: Whether to verify CRC32

        Returns:
            Tuple of (message or None, bytes_consumed)
        """
        if len(data) < HEADER_SIZE:
            return None, 0

        try:
            header = DRGNHeader.unpack(data)
        except Exception:
            return None, 0

        total_size = header.total_bytes + 4

        if len(data) < total_size:
            return None, 0

        try:
            msg = cls.parse(data[:total_size], verify_crc=verify_crc)
            return msg, total_size
        except Exception:
            return None, 0

    def __repr__(self) -> str:
        payload_preview = str(self.payload)[:50] if self.payload else "<raw>"
        return (
            f"SENSEMessage("
            f"method=0x{self.method_id:04X}, "
            f"id={self.message_id}, "
            f"{'response' if self.is_response else 'request'}, "
            f"payload={payload_preview}...)"
        )


def _infer_flags(payload: Any) -> int:
    """
    Infer appropriate flags from payload type.

    Args:
        payload: The payload to be serialized

    Returns:
        Appropriate flag value
    """
    if isinstance(payload, str):
        return FLAG_PAYLOAD_STRING
    elif MSGPACK_AVAILABLE:
        return FLAG_PAYLOAD_MSGPACK
    else:
        return FLAG_PAYLOAD_JSON


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ping(message_id: Optional[int] = None) -> SENSEMessage:
    """
    Create a ping message for health checks.

    Args:
        message_id: Optional message ID

    Returns:
        Ping request message
    """
    from .constants import METHOD_ID_PING
    return SENSEMessage.create_request(
        method_id=METHOD_ID_PING,
        payload={"timestamp": __import__('time').time()},
        message_id=message_id,
    )


def create_pong(ping: SENSEMessage) -> SENSEMessage:
    """
    Create a pong response to a ping.

    Args:
        ping: The ping request

    Returns:
        Pong response message
    """
    from .constants import METHOD_ID_PONG
    return SENSEMessage.create_response(
        request=ping,
        payload={
            "timestamp": __import__('time').time(),
            "request_timestamp": ping.payload.get("timestamp") if ping.payload else None,
        },
    )
