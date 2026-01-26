"""
SENSE Binary Protocol Module

Implements the DRGN binary message protocol for efficient inter-agent
communication with zero-copy parsing and async I/O support.

ARCHITECTURE OVERVIEW:
======================

    ┌─────────────────────────────────────────────────────────────────┐
    │                    SENSE Binary Protocol Stack                   │
    ├─────────────────────────────────────────────────────────────────┤
    │  SENSEMessage (High-Level API)                                  │
    │    ├── create_request() / create_response()                     │
    │    └── Uses DRGNHeader + BinaryParser                           │
    ├─────────────────────────────────────────────────────────────────┤
    │  DRGNHeader (Fixed 29-byte wire format)                         │
    │    └── Network byte order (!), CRC32 integrity                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  BinaryParser (Zero-Copy In-Place Parsing)                      │
    │    └── memoryview slices, lazy UTF-8 decoding                   │
    ├─────────────────────────────────────────────────────────────────┤
    │  AsyncMessageReader/Writer (Non-Blocking I/O)                   │
    │    └── asyncio streams, message framing                         │
    └─────────────────────────────────────────────────────────────────┘

QUICK START:
============

    # Creating a request
    from sense_v2.protocol import SENSEMessage, METHOD_ID_AGENT_USER

    msg = SENSEMessage.create_request(
        method_id=METHOD_ID_AGENT_USER,
        payload={"content": "Hello, agent!"},
    )
    wire_bytes = msg.to_bytes()

    # Parsing a message
    msg = SENSEMessage.parse(wire_bytes)
    print(msg.payload)  # {'content': 'Hello, agent!'}

    # Async I/O
    async with AsyncMessageChannel.connect('localhost', 8080) as channel:
        response = await channel.request(METHOD_ID_PING, {})

    # AgentMessage integration
    from sense_v2.protocol import AgentMessageAdapter
    agent_msg = AgentMessage.user("Hello!")
    binary = AgentMessageAdapter.to_bytes(agent_msg)
"""

# Protocol version
__version__ = "1.0.0"

# =============================================================================
# CONSTANTS
# =============================================================================
from .constants import (
    # Magic and version
    MAGIC_SIGNATURE,
    MAGIC_SIGNATURE_INT,
    PROTOCOL_VERSION,
    HEADER_SIZE,
    HEADER_FORMAT,

    # Payload flags
    FLAG_PAYLOAD_STRING,
    FLAG_PAYLOAD_TENSOR,
    FLAG_PAYLOAD_MSGPACK,
    FLAG_PAYLOAD_JSON,
    FLAG_COMPRESSED,
    FLAG_ENCRYPTED,
    FLAG_STREAMING,
    FLAG_PRIORITY,
    FLAG_RESPONSE,
    FLAG_ERROR,
    FLAG_PARTIAL,

    # Method IDs
    METHOD_ID_PING,
    METHOD_ID_PONG,
    METHOD_ID_ERROR,
    METHOD_ID_CLOSE,
    METHOD_ID_AGENT_SYSTEM,
    METHOD_ID_AGENT_USER,
    METHOD_ID_AGENT_ASSISTANT,
    METHOD_ID_AGENT_TOOL,
    METHOD_ID_TOOL_CALL,
    METHOD_ID_TOOL_RESULT,
    METHOD_ID_MEMORY_STORE,
    METHOD_ID_MEMORY_SEARCH,
    METHOD_ID_MEMORY_DELETE,
    METHOD_ID_ENGRAM_LOOKUP,
    METHOD_ID_ENGRAM_INSERT,
    METHOD_ID_ENGRAM_BATCH,

    # Limits
    MAX_MESSAGE_SIZE,
    MAX_STRING_LENGTH,
    MAX_ARRAY_ELEMENTS,
    MAX_NESTING_DEPTH,
    ASYNC_BUFFER_SIZE,
    ASYNC_READ_TIMEOUT,
    ASYNC_WRITE_TIMEOUT,

    # Helper functions
    get_payload_type,
    is_compressed,
    is_encrypted,
    is_response,
    is_error,
    make_response_flags,
)

# =============================================================================
# EXCEPTIONS
# =============================================================================
from .exceptions import (
    ProtocolError,
    HeaderError,
    InvalidSignatureError,
    UnsupportedVersionError,
    MessageError,
    IncompleteMessageError,
    MessageTooLargeError,
    MalformedMessageError,
    IntegrityError,
    CRCMismatchError,
    SerializationError,
    EncodeError,
    DecodeError,
    AsyncIOError,
    ReadTimeoutError,
    WriteTimeoutError,
    ConnectionClosedError,
    BufferError,
    BufferOverflowError,
    BufferUnderflowError,
)

# =============================================================================
# HEADER
# =============================================================================
from .header import (
    DRGNHeader,
    compute_crc32,
    verify_crc32,
)

# =============================================================================
# PARSER
# =============================================================================
from .parser import (
    BinaryParser,
    BufferBuilder,
)

# =============================================================================
# SERIALIZERS
# =============================================================================
from .serializers import (
    Serializer,
    MessagePackSerializer,
    JSONSerializer,
    StringSerializer,
    MSGPACK_AVAILABLE,
    compress,
    decompress,
    get_serializer,
    get_default_serializer,
    serialize_payload,
    deserialize_payload,
)

# =============================================================================
# MESSAGE
# =============================================================================
from .message import (
    SENSEMessage,
    generate_message_id,
    create_ping,
    create_pong,
)

# =============================================================================
# ASYNC I/O
# =============================================================================
from .async_io import (
    AsyncMessageReader,
    AsyncMessageWriter,
    AsyncMessageChannel,
    read_message,
    write_message,
)

# =============================================================================
# ADAPTERS
# =============================================================================
from .adapters import (
    AgentMessageAdapter,
    ToolResultAdapter,
    ConversationAdapter,
    extend_agent_message,
)


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_json_vs_binary(iterations: int = 1000) -> dict:
    """
    Compare binary protocol vs json.loads performance.

    WHY THIS MATTERS:
    JSON parsing is CPU-bound due to character-by-character scanning.
    Binary protocols skip this entirely - we know exact byte offsets.

    Args:
        iterations: Number of iterations for each benchmark

    Returns:
        Dict with timing results and speedup factor
    """
    import timeit
    import json

    # Test data
    test_data = {"method_id": 42, "payload": "x" * 1000, "nested": {"a": 1, "b": 2}}
    json_bytes = json.dumps(test_data).encode()

    # Create binary message
    binary_msg = SENSEMessage.create_request(
        method_id=42,
        payload=test_data,
    )
    binary_bytes = binary_msg.to_bytes()

    # Benchmark JSON
    json_time = timeit.timeit(
        lambda: json.loads(json_bytes),
        number=iterations
    )

    # Benchmark binary
    binary_time = timeit.timeit(
        lambda: SENSEMessage.parse(binary_bytes),
        number=iterations
    )

    speedup = json_time / binary_time if binary_time > 0 else float('inf')

    return {
        "json_ms": json_time * 1000,
        "binary_ms": binary_time * 1000,
        "json_per_op_us": (json_time / iterations) * 1_000_000,
        "binary_per_op_us": (binary_time / iterations) * 1_000_000,
        "speedup": round(speedup, 2),
        "iterations": iterations,
        "json_size": len(json_bytes),
        "binary_size": len(binary_bytes),
        "size_ratio": round(len(binary_bytes) / len(json_bytes), 2),
    }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Extend AgentMessage with binary serialization methods
extend_agent_message()


__all__ = [
    # Version
    "__version__",

    # Constants
    "MAGIC_SIGNATURE",
    "MAGIC_SIGNATURE_INT",
    "PROTOCOL_VERSION",
    "HEADER_SIZE",
    "HEADER_FORMAT",
    "FLAG_PAYLOAD_STRING",
    "FLAG_PAYLOAD_TENSOR",
    "FLAG_PAYLOAD_MSGPACK",
    "FLAG_PAYLOAD_JSON",
    "FLAG_COMPRESSED",
    "FLAG_ENCRYPTED",
    "FLAG_STREAMING",
    "FLAG_PRIORITY",
    "FLAG_RESPONSE",
    "FLAG_ERROR",
    "FLAG_PARTIAL",
    "METHOD_ID_PING",
    "METHOD_ID_PONG",
    "METHOD_ID_ERROR",
    "METHOD_ID_CLOSE",
    "METHOD_ID_AGENT_SYSTEM",
    "METHOD_ID_AGENT_USER",
    "METHOD_ID_AGENT_ASSISTANT",
    "METHOD_ID_AGENT_TOOL",
    "METHOD_ID_TOOL_CALL",
    "METHOD_ID_TOOL_RESULT",
    "METHOD_ID_MEMORY_STORE",
    "METHOD_ID_MEMORY_SEARCH",
    "METHOD_ID_MEMORY_DELETE",
    "METHOD_ID_ENGRAM_LOOKUP",
    "METHOD_ID_ENGRAM_INSERT",
    "METHOD_ID_ENGRAM_BATCH",
    "MAX_MESSAGE_SIZE",
    "MAX_STRING_LENGTH",
    "MAX_ARRAY_ELEMENTS",
    "MAX_NESTING_DEPTH",
    "ASYNC_BUFFER_SIZE",
    "ASYNC_READ_TIMEOUT",
    "ASYNC_WRITE_TIMEOUT",
    "get_payload_type",
    "is_compressed",
    "is_encrypted",
    "is_response",
    "is_error",
    "make_response_flags",

    # Exceptions
    "ProtocolError",
    "HeaderError",
    "InvalidSignatureError",
    "UnsupportedVersionError",
    "MessageError",
    "IncompleteMessageError",
    "MessageTooLargeError",
    "MalformedMessageError",
    "IntegrityError",
    "CRCMismatchError",
    "SerializationError",
    "EncodeError",
    "DecodeError",
    "AsyncIOError",
    "ReadTimeoutError",
    "WriteTimeoutError",
    "ConnectionClosedError",
    "BufferError",
    "BufferOverflowError",
    "BufferUnderflowError",

    # Header
    "DRGNHeader",
    "compute_crc32",
    "verify_crc32",

    # Parser
    "BinaryParser",
    "BufferBuilder",

    # Serializers
    "Serializer",
    "MessagePackSerializer",
    "JSONSerializer",
    "StringSerializer",
    "MSGPACK_AVAILABLE",
    "compress",
    "decompress",
    "get_serializer",
    "get_default_serializer",
    "serialize_payload",
    "deserialize_payload",

    # Message
    "SENSEMessage",
    "generate_message_id",
    "create_ping",
    "create_pong",

    # Async I/O
    "AsyncMessageReader",
    "AsyncMessageWriter",
    "AsyncMessageChannel",
    "read_message",
    "write_message",

    # Adapters
    "AgentMessageAdapter",
    "ToolResultAdapter",
    "ConversationAdapter",
    "extend_agent_message",

    # Benchmarking
    "benchmark_json_vs_binary",
]
