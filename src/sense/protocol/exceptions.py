"""
SENSE Protocol Exceptions

Defines all protocol-specific exception classes with detailed error information.

Exception Hierarchy:
    ProtocolError (base)
    ├── HeaderError
    │   ├── InvalidSignatureError
    │   └── UnsupportedVersionError
    ├── MessageError
    │   ├── IncompleteMessageError
    │   ├── MessageTooLargeError
    │   └── MalformedMessageError
    ├── IntegrityError
    │   └── CRCMismatchError
    ├── SerializationError
    │   ├── EncodeError
    │   └── DecodeError
    └── AsyncIOError
        ├── ReadTimeoutError
        └── WriteTimeoutError

WHY DETAILED EXCEPTIONS?
------------------------
Specific exception types allow callers to:
1. Handle different error conditions differently (retry vs. reject)
2. Log meaningful error information for debugging
3. Provide user-friendly error messages
4. Avoid catching broad exceptions that mask bugs
"""

from typing import Optional, Any


class ProtocolError(Exception):
    """
    Base exception for all SENSE protocol errors.

    All protocol exceptions inherit from this class, allowing callers
    to catch all protocol errors with a single except clause if desired.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ', '.join(f'{k}={v!r}' for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# HEADER ERRORS
# =============================================================================

class HeaderError(ProtocolError):
    """Base class for header-related errors."""
    pass


class InvalidSignatureError(HeaderError):
    """
    Raised when the message doesn't start with 'DRGN' magic signature.

    This typically indicates:
    1. Data is not a SENSE protocol message
    2. Message was corrupted in transit
    3. Stream is out of sync (partial message received)
    """

    def __init__(self, received: bytes, expected: bytes = b'DRGN'):
        super().__init__(
            f"Invalid magic signature",
            details={
                'received': received.hex() if isinstance(received, bytes) else str(received),
                'expected': expected.hex(),
            }
        )
        self.received = received
        self.expected = expected


class UnsupportedVersionError(HeaderError):
    """
    Raised when protocol version is not supported.

    This allows for forward compatibility - newer clients can detect
    when they're communicating with older servers that don't support
    newer protocol features.
    """

    def __init__(self, version: int, supported_versions: list = None):
        supported = supported_versions or [0]
        super().__init__(
            f"Unsupported protocol version",
            details={
                'received_version': version,
                'supported_versions': supported,
            }
        )
        self.version = version
        self.supported_versions = supported


# =============================================================================
# MESSAGE ERRORS
# =============================================================================

class MessageError(ProtocolError):
    """Base class for message-level errors."""
    pass


class IncompleteMessageError(MessageError):
    """
    Raised when buffer doesn't contain enough data.

    This is a RECOVERABLE error - it typically means we need to
    wait for more data to arrive over the network. Callers should:
    1. Buffer the partial data
    2. Wait for more bytes
    3. Retry parsing when more data is available
    """

    def __init__(
        self,
        message: str = "Incomplete message",
        needed: Optional[int] = None,
        available: Optional[int] = None,
        offset: Optional[int] = None
    ):
        details = {}
        if needed is not None:
            details['bytes_needed'] = needed
        if available is not None:
            details['bytes_available'] = available
        if offset is not None:
            details['offset'] = offset

        super().__init__(message, details)
        self.needed = needed
        self.available = available
        self.offset = offset

    @property
    def is_recoverable(self) -> bool:
        """Incomplete messages are typically recoverable with more data."""
        return True


class MessageTooLargeError(MessageError):
    """
    Raised when message exceeds size limits.

    This is a SECURITY measure to prevent:
    1. Memory exhaustion attacks (allocating huge buffers)
    2. Denial of service (processing extremely large payloads)
    3. Resource starvation in Termux's limited environment

    This error is NOT recoverable - the message should be rejected.
    """

    def __init__(self, size: int, limit: int, field: str = "message"):
        super().__init__(
            f"{field.capitalize()} size exceeds limit",
            details={
                'size': size,
                'limit': limit,
                'field': field,
                'exceeds_by': size - limit,
            }
        )
        self.size = size
        self.limit = limit
        self.field = field

    @property
    def is_recoverable(self) -> bool:
        """Size limit violations are not recoverable."""
        return False


class MalformedMessageError(MessageError):
    """
    Raised when message structure is invalid.

    This indicates the message data is corrupted or doesn't follow
    the protocol specification. Unlike IncompleteMessageError,
    this is NOT recoverable with more data.
    """

    def __init__(
        self,
        message: str = "Malformed message",
        offset: Optional[int] = None,
        field: Optional[str] = None,
        expected: Any = None,
        received: Any = None
    ):
        details = {}
        if offset is not None:
            details['offset'] = offset
        if field is not None:
            details['field'] = field
        if expected is not None:
            details['expected'] = expected
        if received is not None:
            details['received'] = received

        super().__init__(message, details)
        self.offset = offset
        self.field = field

    @property
    def is_recoverable(self) -> bool:
        """Malformed messages are not recoverable."""
        return False


# =============================================================================
# INTEGRITY ERRORS
# =============================================================================

class IntegrityError(ProtocolError):
    """Base class for data integrity errors."""
    pass


class CRCMismatchError(IntegrityError):
    """
    Raised when CRC32 checksum doesn't match.

    This indicates payload corruption during:
    1. Network transmission (bit flips, packet corruption)
    2. Storage (disk errors)
    3. Memory corruption (rare but possible)

    The message should be rejected and potentially re-requested.
    """

    def __init__(self, expected: int, computed: int):
        super().__init__(
            "CRC32 checksum mismatch",
            details={
                'expected_crc': f"0x{expected:08X}",
                'computed_crc': f"0x{computed:08X}",
            }
        )
        self.expected = expected
        self.computed = computed


# =============================================================================
# SERIALIZATION ERRORS
# =============================================================================

class SerializationError(ProtocolError):
    """Base class for serialization/deserialization errors."""
    pass


class EncodeError(SerializationError):
    """
    Raised when payload cannot be encoded.

    This typically indicates:
    1. Unsupported data type in payload
    2. Circular references in data structure
    3. Non-serializable objects (functions, classes, etc.)
    """

    def __init__(
        self,
        message: str = "Failed to encode payload",
        format_type: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if format_type:
            details['format'] = format_type
        if original_error:
            details['original_error'] = str(original_error)

        super().__init__(message, details)
        self.format_type = format_type
        self.original_error = original_error


class DecodeError(SerializationError):
    """
    Raised when payload cannot be decoded.

    This typically indicates:
    1. Corrupted payload data
    2. Wrong serialization format specified in flags
    3. Incompatible MessagePack/JSON format
    """

    def __init__(
        self,
        message: str = "Failed to decode payload",
        format_type: Optional[str] = None,
        offset: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if format_type:
            details['format'] = format_type
        if offset is not None:
            details['offset'] = offset
        if original_error:
            details['original_error'] = str(original_error)

        super().__init__(message, details)
        self.format_type = format_type
        self.offset = offset
        self.original_error = original_error


# =============================================================================
# ASYNC I/O ERRORS
# =============================================================================

class AsyncIOError(ProtocolError):
    """Base class for async I/O errors."""
    pass


class ReadTimeoutError(AsyncIOError):
    """
    Raised when async read operation times out.

    This may indicate:
    1. Network congestion or packet loss
    2. Remote peer is slow or unresponsive
    3. Deadlock on the sender side
    """

    def __init__(self, timeout: float, bytes_expected: Optional[int] = None):
        details = {'timeout_seconds': timeout}
        if bytes_expected is not None:
            details['bytes_expected'] = bytes_expected

        super().__init__("Read operation timed out", details)
        self.timeout = timeout
        self.bytes_expected = bytes_expected


class WriteTimeoutError(AsyncIOError):
    """
    Raised when async write operation times out.

    This may indicate:
    1. Network congestion (send buffer full)
    2. Remote peer not reading data
    3. Connection issues
    """

    def __init__(self, timeout: float, bytes_to_write: Optional[int] = None):
        details = {'timeout_seconds': timeout}
        if bytes_to_write is not None:
            details['bytes_to_write'] = bytes_to_write

        super().__init__("Write operation timed out", details)
        self.timeout = timeout
        self.bytes_to_write = bytes_to_write


class ConnectionClosedError(AsyncIOError):
    """
    Raised when connection is closed unexpectedly.

    This indicates the remote peer closed the connection before
    the message was fully transmitted/received.
    """

    def __init__(self, message: str = "Connection closed unexpectedly"):
        super().__init__(message)


# =============================================================================
# BUFFER MANAGEMENT ERRORS
# =============================================================================

class BufferError(ProtocolError):
    """Base class for buffer management errors."""
    pass


class BufferOverflowError(BufferError):
    """
    Raised when write operation would exceed buffer capacity.
    """

    def __init__(self, requested: int, available: int):
        super().__init__(
            "Buffer overflow",
            details={
                'bytes_requested': requested,
                'bytes_available': available,
            }
        )
        self.requested = requested
        self.available = available


class BufferUnderflowError(BufferError):
    """
    Raised when read operation requests more data than available.
    """

    def __init__(self, requested: int, available: int, offset: Optional[int] = None):
        details = {
            'bytes_requested': requested,
            'bytes_available': available,
        }
        if offset is not None:
            details['offset'] = offset

        super().__init__("Buffer underflow", details)
        self.requested = requested
        self.available = available
        self.offset = offset
