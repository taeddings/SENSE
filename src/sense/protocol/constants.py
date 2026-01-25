"""
SENSE Protocol Constants

Defines all protocol-level constants, magic numbers, and configuration limits.

WHY NETWORK BYTE ORDER (!)?
---------------------------
Network byte order (big-endian) is the standard for cross-platform protocols.
Using '!' in struct format strings ensures:
1. Consistent byte order on ARM64 (Android/Termux), x86-64, and other architectures
2. No unaligned access faults on ARM processors
3. Standard wire format matching TCP/IP conventions

The 'DRGN' magic signature (0x4452474E) allows quick message validation
without parsing the entire header - if the first 4 bytes don't match,
we can reject the message immediately.
"""

import struct

# =============================================================================
# PROTOCOL SIGNATURE
# =============================================================================

# Magic signature bytes: ASCII "DRGN" = 0x44 0x52 0x47 0x4E
MAGIC_SIGNATURE = b'DRGN'
MAGIC_SIGNATURE_INT = 0x4452474E

# Protocol version (uint8)
PROTOCOL_VERSION = 0x00  # v1.0

# =============================================================================
# HEADER STRUCTURE
# =============================================================================

# Header size in bytes (fixed)
# Layout: !I I B I I Q I = 4 + 4 + 1 + 4 + 4 + 8 + 4 = 29 bytes
HEADER_SIZE = 29

# Struct format for header (network byte order)
# !  = network (big-endian)
# I  = uint32 (4 bytes) - signature
# I  = uint32 (4 bytes) - total_bytes
# B  = uint8  (1 byte)  - protocol_version
# I  = uint32 (4 bytes) - method_id
# I  = uint32 (4 bytes) - flags
# Q  = uint64 (8 bytes) - message_id
# I  = uint32 (4 bytes) - crc32
HEADER_FORMAT = '!IIBIIQI'

# Verify header size at module load time
assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE, \
    f"Header format size mismatch: expected {HEADER_SIZE}, got {struct.calcsize(HEADER_FORMAT)}"

# =============================================================================
# PAYLOAD FLAGS (Bitmask)
# =============================================================================

# Payload type flags (lower 16 bits)
FLAG_PAYLOAD_STRING = 0x00000000   # UTF-8 string payload
FLAG_PAYLOAD_TENSOR = 0x00000001   # Binary tensor data
FLAG_PAYLOAD_MSGPACK = 0x00000002  # MessagePack serialized
FLAG_PAYLOAD_JSON = 0x00000003     # JSON serialized (fallback)

# Processing flags (upper 16 bits)
FLAG_COMPRESSED = 0x00000010       # zlib compressed payload
FLAG_ENCRYPTED = 0x00000020        # Encrypted payload (future)
FLAG_STREAMING = 0x00000040        # Streaming response expected
FLAG_PRIORITY = 0x00000080         # High-priority message

# Response flags
FLAG_RESPONSE = 0x00000100         # This is a response message
FLAG_ERROR = 0x00000200            # Response indicates error
FLAG_PARTIAL = 0x00000400          # Partial response (more coming)

# =============================================================================
# METHOD IDs
# =============================================================================

# Reserved method IDs (0x0000 - 0x00FF)
METHOD_ID_PING = 0x00000001        # Health check
METHOD_ID_PONG = 0x00000002        # Health check response
METHOD_ID_ERROR = 0x00000003       # Error response
METHOD_ID_CLOSE = 0x00000004       # Connection close

# Agent message methods (0x0100 - 0x01FF)
METHOD_ID_AGENT_SYSTEM = 0x00000101    # System message
METHOD_ID_AGENT_USER = 0x00000102      # User message
METHOD_ID_AGENT_ASSISTANT = 0x00000103 # Assistant message
METHOD_ID_AGENT_TOOL = 0x00000104      # Tool message

# Tool invocation methods (0x0200 - 0x02FF)
METHOD_ID_TOOL_CALL = 0x00000201       # Tool invocation request
METHOD_ID_TOOL_RESULT = 0x00000202     # Tool result response

# Memory operations (0x0300 - 0x03FF)
METHOD_ID_MEMORY_STORE = 0x00000301    # Store to memory
METHOD_ID_MEMORY_SEARCH = 0x00000302   # Search memory
METHOD_ID_MEMORY_DELETE = 0x00000303   # Delete from memory

# Engram operations (0x0400 - 0x04FF)
METHOD_ID_ENGRAM_LOOKUP = 0x00000401   # Engram table lookup
METHOD_ID_ENGRAM_INSERT = 0x00000402   # Insert engram
METHOD_ID_ENGRAM_BATCH = 0x00000403    # Batch operations

# =============================================================================
# SIZE LIMITS (DoS Protection)
# =============================================================================

# Maximum message size (64 MB)
# WHY 64 MB? Large enough for embedding batches, small enough to prevent
# memory exhaustion on mobile/Termux with limited RAM.
MAX_MESSAGE_SIZE = 64 * 1024 * 1024

# Maximum string field length (16 MB)
# Prevents single string fields from consuming all memory.
MAX_STRING_LENGTH = 16 * 1024 * 1024

# Maximum array elements (1 million)
# Prevents unbounded array allocation attacks.
MAX_ARRAY_ELEMENTS = 1_000_000

# Maximum nested depth (64 levels)
# Prevents stack overflow from deeply nested structures.
MAX_NESTING_DEPTH = 64

# =============================================================================
# ASYNC I/O SETTINGS
# =============================================================================

# Default read buffer size for async streams (64 KB)
# WHY 64 KB? Matches typical TCP window size and provides good balance
# between memory usage and read efficiency.
ASYNC_BUFFER_SIZE = 65536

# Read timeout for async operations (30 seconds)
ASYNC_READ_TIMEOUT = 30.0

# Write timeout for async operations (30 seconds)
ASYNC_WRITE_TIMEOUT = 30.0

# Maximum concurrent pending messages (256)
MAX_PENDING_MESSAGES = 256

# =============================================================================
# CRC32 SETTINGS
# =============================================================================

# CRC32 polynomial (standard IEEE 802.3)
# We use zlib.crc32 which implements this polynomial.
CRC32_POLYNOMIAL = 0xEDB88320

# Initial CRC value
CRC32_INITIAL = 0x00000000

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_payload_type(flags: int) -> int:
    """Extract payload type from flags (lower 4 bits)."""
    return flags & 0x0000000F


def is_compressed(flags: int) -> bool:
    """Check if payload is compressed."""
    return bool(flags & FLAG_COMPRESSED)


def is_encrypted(flags: int) -> bool:
    """Check if payload is encrypted."""
    return bool(flags & FLAG_ENCRYPTED)


def is_response(flags: int) -> bool:
    """Check if message is a response."""
    return bool(flags & FLAG_RESPONSE)


def is_error(flags: int) -> bool:
    """Check if message indicates an error."""
    return bool(flags & FLAG_ERROR)


def make_response_flags(request_flags: int, is_error: bool = False) -> int:
    """Create response flags from request flags."""
    flags = request_flags | FLAG_RESPONSE
    if is_error:
        flags |= FLAG_ERROR
    return flags
