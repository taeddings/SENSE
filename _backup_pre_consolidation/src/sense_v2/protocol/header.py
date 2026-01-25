"""
SENSE Protocol Header (DRGN Format)

Implements the fixed 29-byte binary header for SENSE protocol messages.

DRGN HEADER LAYOUT (29 bytes total):
=====================================
Offset  Size   Field            Type      Description
──────────────────────────────────────────────────────────────
0       4      Signature        uint32    Magic 'DRGN' (0x4452474E)
4       4      TotalBytes       uint32    Message size after this field
8       1      ProtocolVersion  uint8     Version (0x00 for v1)
9       4      MethodID         uint32    Remote function identifier
13      4      Flags            uint32    Payload type bitmask
17      8      MessageID        uint64    Async request/response tracking
25      4      CRC32            uint32    Checksum of payload

WHY FIXED-SIZE HEADER?
----------------------
1. PREDICTABLE PARSING: We know exactly where each field is located.
   No need to parse variable-length fields to find the payload.

2. ZERO-COPY FRIENDLY: Can use struct.unpack_from() directly on buffer
   without allocating intermediate data structures.

3. NETWORK EFFICIENCY: Single read for header, then payload size is known.
   Enables optimal TCP windowing and read buffering.

4. ALIGNMENT: 29 bytes might seem odd, but all multi-byte fields are
   naturally aligned within the struct (uint32 at 4-byte boundaries,
   uint64 at 8-byte boundaries) when using packed format.

BYTE ORDER (Network / Big-Endian):
----------------------------------
Using '!' (network byte order) in struct format for:
- Cross-platform compatibility (ARM64, x86-64, etc.)
- Standard wire format matching TCP/IP conventions
- Avoids unaligned access faults on ARM processors
"""

import struct
import zlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional

from .constants import (
    MAGIC_SIGNATURE,
    MAGIC_SIGNATURE_INT,
    PROTOCOL_VERSION,
    HEADER_SIZE,
    HEADER_FORMAT,
    MAX_MESSAGE_SIZE,
    FLAG_PAYLOAD_MSGPACK,
)
from .exceptions import (
    InvalidSignatureError,
    UnsupportedVersionError,
    IncompleteMessageError,
    MessageTooLargeError,
    MalformedMessageError,
)


@dataclass(frozen=True, slots=True)
class DRGNHeader:
    """
    Immutable DRGN protocol header.

    Using frozen=True and slots=True for:
    - Immutability: Headers shouldn't be modified after creation
    - Memory efficiency: slots avoid __dict__ overhead
    - Hash support: Frozen dataclasses are hashable

    Attributes:
        signature: Magic bytes 'DRGN' (0x4452474E)
        total_bytes: Size of message after the signature field
        protocol_version: Protocol version (0 for v1.0)
        method_id: Identifier for the remote procedure/message type
        flags: Bitmask for payload type and processing options
        message_id: Unique ID for request/response correlation
        crc32: CRC32 checksum of the payload
    """

    signature: int
    total_bytes: int
    protocol_version: int
    method_id: int
    flags: int
    message_id: int
    crc32: int

    def __post_init__(self):
        """Validate header fields after initialization."""
        # Validate signature
        if self.signature != MAGIC_SIGNATURE_INT:
            raise InvalidSignatureError(
                self.signature.to_bytes(4, 'big'),
                MAGIC_SIGNATURE
            )

        # Validate protocol version
        if self.protocol_version > 0:
            raise UnsupportedVersionError(
                self.protocol_version,
                supported_versions=[0]
            )

        # Validate message size
        if self.total_bytes > MAX_MESSAGE_SIZE:
            raise MessageTooLargeError(
                self.total_bytes,
                MAX_MESSAGE_SIZE,
                field="total_bytes"
            )

    @property
    def payload_size(self) -> int:
        """
        Calculate the size of the payload in bytes.

        total_bytes includes everything after the signature field,
        so payload = total_bytes - (header_size - 4 bytes for signature)
        """
        return self.total_bytes - (HEADER_SIZE - 4)

    def pack(self) -> bytes:
        """
        Pack header into wire format bytes.

        Returns:
            29 bytes representing the header in network byte order.

        WHY struct.pack vs manual bytes:
        struct.pack handles byte order conversion and is optimized
        in C - it's faster than manual byte manipulation in Python.
        """
        return struct.pack(
            HEADER_FORMAT,
            self.signature,
            self.total_bytes,
            self.protocol_version,
            self.method_id,
            self.flags,
            self.message_id,
            self.crc32,
        )

    @classmethod
    def unpack(cls, data: Union[bytes, bytearray, memoryview]) -> 'DRGNHeader':
        """
        Unpack header from wire format bytes.

        Args:
            data: At least 29 bytes of header data

        Returns:
            DRGNHeader instance with validated fields

        Raises:
            IncompleteMessageError: If data is less than 29 bytes
            InvalidSignatureError: If magic signature doesn't match
            UnsupportedVersionError: If protocol version is unsupported
            MessageTooLargeError: If total_bytes exceeds limits
        """
        if len(data) < HEADER_SIZE:
            raise IncompleteMessageError(
                "Incomplete header",
                needed=HEADER_SIZE,
                available=len(data),
                offset=0
            )

        try:
            fields = struct.unpack_from(HEADER_FORMAT, data, 0)
        except struct.error as e:
            raise MalformedMessageError(
                f"Failed to unpack header: {e}",
                offset=0
            )

        return cls(
            signature=fields[0],
            total_bytes=fields[1],
            protocol_version=fields[2],
            method_id=fields[3],
            flags=fields[4],
            message_id=fields[5],
            crc32=fields[6],
        )

    @classmethod
    def unpack_from(
        cls,
        data: Union[bytes, bytearray, memoryview],
        offset: int = 0
    ) -> Tuple['DRGNHeader', int]:
        """
        Unpack header from buffer at specified offset.

        This is useful for parsing multiple messages from a single buffer
        or for zero-copy parsing where we don't want to slice the buffer.

        Args:
            data: Buffer containing header data
            offset: Byte offset to start parsing from

        Returns:
            Tuple of (DRGNHeader, bytes_consumed)

        Raises:
            IncompleteMessageError: If not enough data available
        """
        available = len(data) - offset
        if available < HEADER_SIZE:
            raise IncompleteMessageError(
                "Incomplete header at offset",
                needed=HEADER_SIZE,
                available=available,
                offset=offset
            )

        try:
            fields = struct.unpack_from(HEADER_FORMAT, data, offset)
        except struct.error as e:
            raise MalformedMessageError(
                f"Failed to unpack header: {e}",
                offset=offset
            )

        header = cls(
            signature=fields[0],
            total_bytes=fields[1],
            protocol_version=fields[2],
            method_id=fields[3],
            flags=fields[4],
            message_id=fields[5],
            crc32=fields[6],
        )

        return header, HEADER_SIZE

    @classmethod
    def create(
        cls,
        method_id: int,
        payload_size: int,
        flags: int = FLAG_PAYLOAD_MSGPACK,
        message_id: int = 0,
        crc32: int = 0,
    ) -> 'DRGNHeader':
        """
        Factory method to create a new header.

        Automatically sets signature and protocol version, and calculates
        total_bytes from payload_size.

        Args:
            method_id: Remote procedure identifier
            payload_size: Size of payload in bytes
            flags: Payload type and processing flags
            message_id: Request/response correlation ID
            crc32: Checksum of payload (typically computed separately)

        Returns:
            New DRGNHeader instance
        """
        # total_bytes = header_after_signature + payload
        # header_after_signature = HEADER_SIZE - 4 (signature size)
        total_bytes = (HEADER_SIZE - 4) + payload_size

        return cls(
            signature=MAGIC_SIGNATURE_INT,
            total_bytes=total_bytes,
            protocol_version=PROTOCOL_VERSION,
            method_id=method_id,
            flags=flags,
            message_id=message_id,
            crc32=crc32,
        )

    def with_crc32(self, payload: bytes) -> 'DRGNHeader':
        """
        Create a new header with CRC32 computed from payload.

        Since DRGNHeader is frozen (immutable), this creates a new instance
        with the updated CRC32 value.

        Args:
            payload: The payload bytes to compute CRC32 for

        Returns:
            New DRGNHeader with computed CRC32
        """
        computed_crc = compute_crc32(payload)
        return DRGNHeader(
            signature=self.signature,
            total_bytes=self.total_bytes,
            protocol_version=self.protocol_version,
            method_id=self.method_id,
            flags=self.flags,
            message_id=self.message_id,
            crc32=computed_crc,
        )

    def verify_crc32(self, payload: bytes) -> bool:
        """
        Verify that payload matches stored CRC32.

        Args:
            payload: The payload bytes to verify

        Returns:
            True if CRC32 matches, False otherwise
        """
        computed = compute_crc32(payload)
        return computed == self.crc32

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return (
            f"DRGNHeader("
            f"method=0x{self.method_id:04X}, "
            f"flags=0x{self.flags:08X}, "
            f"msg_id={self.message_id}, "
            f"payload={self.payload_size}B, "
            f"crc=0x{self.crc32:08X})"
        )


def compute_crc32(data: Union[bytes, bytearray, memoryview]) -> int:
    """
    Compute CRC32 checksum for data.

    Uses zlib.crc32 which implements the standard IEEE 802.3 polynomial.
    Returns unsigned 32-bit integer (masked with 0xFFFFFFFF for Python 2
    compatibility, though we only support Python 3).

    WHY CRC32?
    ----------
    1. FAST: Hardware-accelerated on modern CPUs (CRC32 instruction)
    2. SIMPLE: Easy to implement and verify
    3. ADEQUATE: Good enough for detecting corruption (not security!)

    Note: CRC32 is NOT cryptographically secure. It detects accidental
    corruption, not malicious tampering. For security, use HMAC or similar.

    Args:
        data: Bytes to compute checksum for

    Returns:
        Unsigned 32-bit CRC32 value
    """
    return zlib.crc32(data) & 0xFFFFFFFF


def verify_crc32(data: bytes, expected: int) -> bool:
    """
    Verify data against expected CRC32.

    Args:
        data: Bytes to verify
        expected: Expected CRC32 value

    Returns:
        True if CRC32 matches, False otherwise
    """
    return compute_crc32(data) == expected
