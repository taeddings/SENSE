"""
SENSE Protocol Binary Parser

Zero-copy in-place parser using memoryview for efficient buffer parsing.

WHY ZERO-COPY MATTERS:
======================
Traditional JSON parsing copies data multiple times:
  Network buffer → Python string → JSON decoder → Python dict

Each copy means:
1. MORE RAM USAGE: 2-3x message size in temporary allocations
2. MORE CACHE MISSES: CPU fetches from different memory locations
3. MORE GC PRESSURE: Temporary objects need garbage collection

Zero-copy parsing keeps data in the original buffer and returns
"views" (pointers) into that memory. No copies = faster parsing.

WHY ZERO-COPY REDUCES CACHE MISSES:
===================================
Your CPU has multiple cache levels (L1, L2, L3) that store recently
accessed memory. When you copy data, the copy goes to a NEW memory
location that isn't in cache yet. The CPU must fetch it from RAM,
which is 100x slower than cache.

Zero-copy keeps data in the SAME location. When you parse the header,
that memory is now in L1 cache. When you read the payload (which is
right next to the header), it's likely ALREADY in cache from the
prefetcher. Result: fewer cache misses, faster parsing.

MEMORYVIEW EXPLAINED:
=====================
A memoryview is like a "window" into a buffer. It doesn't copy data;
it just remembers where the data is and how to read it. When you
slice a memoryview, you get another memoryview pointing to the same
underlying buffer, just at a different offset.

Example:
    buffer = b'DRGN...(header)...(payload)...'
    view = memoryview(buffer)
    header_view = view[0:29]     # Points to same memory, no copy!
    payload_view = view[29:]     # Also no copy!
"""

import struct
from typing import Union, Tuple, Optional, List, Any

from .constants import (
    HEADER_SIZE,
    HEADER_FORMAT,
    MAX_STRING_LENGTH,
    MAX_ARRAY_ELEMENTS,
    MAX_NESTING_DEPTH,
)
from .exceptions import (
    IncompleteMessageError,
    MessageTooLargeError,
    MalformedMessageError,
    BufferUnderflowError,
)


class BinaryParser:
    """
    In-place binary buffer parser with bounds checking.

    This parser operates on a memoryview, enabling zero-copy access
    to the underlying buffer. All read operations advance an internal
    cursor, making it easy to parse sequential binary data.

    Thread Safety:
        NOT thread-safe. Each thread should have its own parser instance.

    Example Usage:
        >>> buffer = message_bytes
        >>> parser = BinaryParser(buffer)
        >>> header_fields = parser.read_fixed('!IIBIIQI')
        >>> payload = parser.read_bytes(payload_size)
        >>> remaining = parser.remaining()
    """

    __slots__ = ('_buffer', '_position', '_length', '_nesting_depth')

    def __del__(self):
        """Release buffer reference on deletion."""
        self._buffer = None

    def __init__(self, buffer: Union[bytes, bytearray, memoryview]):
        """
        Initialize parser with a buffer.

        Args:
            buffer: The buffer to parse. Will be wrapped in memoryview
                   if not already one.
        """
        if isinstance(buffer, memoryview):
            self._buffer = buffer
        else:
            self._buffer = memoryview(buffer)

        self._position = 0
        self._length = len(self._buffer)
        self._nesting_depth = 0

    @property
    def position(self) -> int:
        """Current read position in buffer."""
        return self._position

    @property
    def remaining(self) -> int:
        """Number of bytes remaining to be read."""
        return self._length - self._position

    @property
    def is_exhausted(self) -> bool:
        """Check if all bytes have been consumed."""
        return self._position >= self._length

    def seek(self, position: int) -> None:
        """
        Move read cursor to specified position.

        Args:
            position: Absolute position in buffer

        Raises:
            ValueError: If position is out of bounds
        """
        if position < 0 or position > self._length:
            raise ValueError(
                f"Seek position {position} out of bounds [0, {self._length}]"
            )
        self._position = position

    def skip(self, count: int) -> None:
        """
        Skip over bytes without reading.

        Args:
            count: Number of bytes to skip

        Raises:
            IncompleteMessageError: If not enough bytes available
        """
        if self._position + count > self._length:
            raise IncompleteMessageError(
                f"Cannot skip {count} bytes",
                needed=count,
                available=self.remaining,
                offset=self._position
            )
        self._position += count

    def peek(self, count: int) -> memoryview:
        """
        Read bytes without advancing cursor.

        Args:
            count: Number of bytes to peek

        Returns:
            memoryview slice of requested bytes

        Raises:
            IncompleteMessageError: If not enough bytes available
        """
        if self._position + count > self._length:
            raise IncompleteMessageError(
                f"Cannot peek {count} bytes",
                needed=count,
                available=self.remaining,
                offset=self._position
            )
        return self._buffer[self._position:self._position + count]

    def read_bytes(self, count: int) -> memoryview:
        """
        Read raw bytes and advance cursor.

        This returns a memoryview slice - NO COPY is made.
        The returned view is valid as long as the original buffer exists.

        Args:
            count: Number of bytes to read

        Returns:
            memoryview slice of requested bytes

        Raises:
            IncompleteMessageError: If not enough bytes available
        """
        if self._position + count > self._length:
            raise IncompleteMessageError(
                f"Cannot read {count} bytes",
                needed=count,
                available=self.remaining,
                offset=self._position
            )

        result = self._buffer[self._position:self._position + count]
        self._position += count
        return result

    def read_bytes_copy(self, count: int) -> bytes:
        """
        Read raw bytes as a copy.

        Use this when you need independent bytes that outlive the buffer,
        or when passing to APIs that require bytes (not memoryview).

        Args:
            count: Number of bytes to read

        Returns:
            bytes object (copy of buffer data)
        """
        view = self.read_bytes(count)
        return bytes(view)

    def read_fixed(self, fmt: str) -> Tuple:
        """
        Read fixed-size values using struct format.

        This is the core parsing method for binary protocols. It uses
        struct.unpack_from() which is optimized in C and handles byte
        order conversion automatically.

        Args:
            fmt: struct format string (e.g., '!I' for network-order uint32)

        Returns:
            Tuple of unpacked values

        Raises:
            IncompleteMessageError: If not enough bytes for format

        Example:
            >>> parser = BinaryParser(data)
            >>> (signature, size, version) = parser.read_fixed('!IIB')
        """
        size = struct.calcsize(fmt)
        if self._position + size > self._length:
            raise IncompleteMessageError(
                f"Need {size} bytes for format '{fmt}'",
                needed=size,
                available=self.remaining,
                offset=self._position
            )

        result = struct.unpack_from(fmt, self._buffer, self._position)
        self._position += size
        return result

    def read_uint8(self) -> int:
        """Read unsigned 8-bit integer."""
        return self.read_fixed('!B')[0]

    def read_uint16(self) -> int:
        """Read unsigned 16-bit integer (network byte order)."""
        return self.read_fixed('!H')[0]

    def read_uint32(self) -> int:
        """Read unsigned 32-bit integer (network byte order)."""
        return self.read_fixed('!I')[0]

    def read_uint64(self) -> int:
        """Read unsigned 64-bit integer (network byte order)."""
        return self.read_fixed('!Q')[0]

    def read_int8(self) -> int:
        """Read signed 8-bit integer."""
        return self.read_fixed('!b')[0]

    def read_int16(self) -> int:
        """Read signed 16-bit integer (network byte order)."""
        return self.read_fixed('!h')[0]

    def read_int32(self) -> int:
        """Read signed 32-bit integer (network byte order)."""
        return self.read_fixed('!i')[0]

    def read_int64(self) -> int:
        """Read signed 64-bit integer (network byte order)."""
        return self.read_fixed('!q')[0]

    def read_float32(self) -> float:
        """Read 32-bit float (network byte order)."""
        return self.read_fixed('!f')[0]

    def read_float64(self) -> float:
        """Read 64-bit float (network byte order)."""
        return self.read_fixed('!d')[0]

    def read_string_view(self, length: int) -> memoryview:
        """
        Return a memoryview slice for string data.

        LAZY DECODING: This does NOT decode to UTF-8 - it returns
        raw bytes. Decode only when the string is actually needed.
        This is useful when you might discard the data or when
        the receiver needs raw bytes.

        Args:
            length: Number of bytes to read

        Returns:
            memoryview pointing to string bytes

        Raises:
            MessageTooLargeError: If length exceeds MAX_STRING_LENGTH
            IncompleteMessageError: If not enough bytes available
        """
        if length > MAX_STRING_LENGTH:
            raise MessageTooLargeError(
                length,
                MAX_STRING_LENGTH,
                field="string"
            )
        return self.read_bytes(length)

    def read_string(self, length: int, encoding: str = 'utf-8') -> str:
        """
        Read and decode a string.

        This DOES decode to the specified encoding. Use read_string_view()
        for lazy decoding.

        Args:
            length: Number of bytes to read
            encoding: String encoding (default UTF-8)

        Returns:
            Decoded string

        Raises:
            MessageTooLargeError: If length exceeds MAX_STRING_LENGTH
            MalformedMessageError: If decoding fails
        """
        view = self.read_string_view(length)
        try:
            return bytes(view).decode(encoding)
        except UnicodeDecodeError as e:
            raise MalformedMessageError(
                f"Failed to decode string as {encoding}",
                offset=self._position - length,
                field="string",
            )

    def read_length_prefixed_bytes(self, length_format: str = '!I') -> memoryview:
        """
        Read bytes with a length prefix.

        Common pattern in binary protocols: first read the length,
        then read that many bytes.

        Args:
            length_format: struct format for length field (default: uint32)

        Returns:
            memoryview of the data bytes
        """
        (length,) = self.read_fixed(length_format)
        return self.read_bytes(length)

    def read_length_prefixed_string(
        self,
        length_format: str = '!I',
        encoding: str = 'utf-8'
    ) -> str:
        """
        Read a length-prefixed string.

        Args:
            length_format: struct format for length field
            encoding: String encoding

        Returns:
            Decoded string
        """
        (length,) = self.read_fixed(length_format)
        return self.read_string(length, encoding)

    def read_remaining(self) -> memoryview:
        """
        Read all remaining bytes.

        Returns:
            memoryview of all remaining data
        """
        result = self._buffer[self._position:]
        self._position = self._length
        return result

    def enter_nesting(self) -> None:
        """
        Track nesting depth for recursive structures.

        Raises:
            MalformedMessageError: If max nesting depth exceeded
        """
        self._nesting_depth += 1
        if self._nesting_depth > MAX_NESTING_DEPTH:
            raise MalformedMessageError(
                f"Maximum nesting depth ({MAX_NESTING_DEPTH}) exceeded",
                offset=self._position
            )

    def exit_nesting(self) -> None:
        """Exit one level of nesting."""
        self._nesting_depth = max(0, self._nesting_depth - 1)

    def get_slice(self, start: int, end: int) -> memoryview:
        """
        Get a slice of the buffer (zero-copy).

        Args:
            start: Start offset (absolute)
            end: End offset (absolute)

        Returns:
            memoryview slice
        """
        return self._buffer[start:end]

    def get_buffer(self) -> memoryview:
        """Get the underlying buffer."""
        return self._buffer

    def create_sub_parser(self, length: int) -> 'BinaryParser':
        """
        Create a sub-parser for a section of the buffer.

        Useful for parsing nested structures where you want to
        limit the scope of parsing.

        Args:
            length: Number of bytes for sub-parser

        Returns:
            New BinaryParser for the slice
        """
        sub_buffer = self.read_bytes(length)
        return BinaryParser(sub_buffer)

    def __repr__(self) -> str:
        return (
            f"BinaryParser(pos={self._position}, "
            f"remaining={self.remaining}, "
            f"total={self._length})"
        )


class BufferBuilder:
    """
    Helper for building binary buffers incrementally.

    Complements BinaryParser by providing the inverse operation:
    constructing binary data from structured values.

    Example:
        >>> builder = BufferBuilder()
        >>> builder.write_uint32(0x4452474E)  # 'DRGN'
        >>> builder.write_string("Hello")
        >>> data = builder.to_bytes()
    """

    __slots__ = ('_parts', '_total_size')

    def __init__(self, initial_capacity: int = 1024):
        """
        Initialize buffer builder.

        Args:
            initial_capacity: Hint for initial capacity (not enforced)
        """
        self._parts: List[bytes] = []
        self._total_size = 0

    @property
    def size(self) -> int:
        """Current size of accumulated data."""
        return self._total_size

    def write_bytes(self, data: Union[bytes, bytearray, memoryview]) -> 'BufferBuilder':
        """
        Write raw bytes.

        Args:
            data: Bytes to write

        Returns:
            self for chaining
        """
        if isinstance(data, memoryview):
            data = bytes(data)
        elif isinstance(data, bytearray):
            data = bytes(data)
        self._parts.append(data)
        self._total_size += len(data)
        return self

    def write_fixed(self, fmt: str, *values) -> 'BufferBuilder':
        """
        Write values using struct format.

        Args:
            fmt: struct format string
            *values: Values to pack

        Returns:
            self for chaining
        """
        data = struct.pack(fmt, *values)
        self._parts.append(data)
        self._total_size += len(data)
        return self

    def write_uint8(self, value: int) -> 'BufferBuilder':
        """Write unsigned 8-bit integer."""
        return self.write_fixed('!B', value)

    def write_uint16(self, value: int) -> 'BufferBuilder':
        """Write unsigned 16-bit integer (network byte order)."""
        return self.write_fixed('!H', value)

    def write_uint32(self, value: int) -> 'BufferBuilder':
        """Write unsigned 32-bit integer (network byte order)."""
        return self.write_fixed('!I', value)

    def write_uint64(self, value: int) -> 'BufferBuilder':
        """Write unsigned 64-bit integer (network byte order)."""
        return self.write_fixed('!Q', value)

    def write_int8(self, value: int) -> 'BufferBuilder':
        """Write signed 8-bit integer."""
        return self.write_fixed('!b', value)

    def write_int16(self, value: int) -> 'BufferBuilder':
        """Write signed 16-bit integer (network byte order)."""
        return self.write_fixed('!h', value)

    def write_int32(self, value: int) -> 'BufferBuilder':
        """Write signed 32-bit integer (network byte order)."""
        return self.write_fixed('!i', value)

    def write_int64(self, value: int) -> 'BufferBuilder':
        """Write signed 64-bit integer (network byte order)."""
        return self.write_fixed('!q', value)

    def write_float32(self, value: float) -> 'BufferBuilder':
        """Write 32-bit float (network byte order)."""
        return self.write_fixed('!f', value)

    def write_float64(self, value: float) -> 'BufferBuilder':
        """Write 64-bit float (network byte order)."""
        return self.write_fixed('!d', value)

    def write_string(
        self,
        value: str,
        encoding: str = 'utf-8',
        length_prefix_format: Optional[str] = '!I'
    ) -> 'BufferBuilder':
        """
        Write a string with optional length prefix.

        Args:
            value: String to write
            encoding: String encoding
            length_prefix_format: Format for length prefix (None to skip)

        Returns:
            self for chaining
        """
        encoded = value.encode(encoding)
        if length_prefix_format:
            self.write_fixed(length_prefix_format, len(encoded))
        self.write_bytes(encoded)
        return self

    def write_length_prefixed_bytes(
        self,
        data: bytes,
        length_format: str = '!I'
    ) -> 'BufferBuilder':
        """
        Write bytes with a length prefix.

        Args:
            data: Bytes to write
            length_format: struct format for length field

        Returns:
            self for chaining
        """
        self.write_fixed(length_format, len(data))
        self.write_bytes(data)
        return self

    def to_bytes(self) -> bytes:
        """
        Finalize and return the complete buffer.

        Returns:
            Complete buffer as bytes
        """
        return b''.join(self._parts)

    def clear(self) -> None:
        """Clear the buffer for reuse."""
        self._parts.clear()
        self._total_size = 0

    def __repr__(self) -> str:
        return f"BufferBuilder(size={self._total_size}, parts={len(self._parts)})"
