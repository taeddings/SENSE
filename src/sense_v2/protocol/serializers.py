"""
SENSE Protocol Serializers

Provides MessagePack (primary) and JSON (fallback) serialization.

WHY MESSAGEPACK AS PRIMARY?
===========================
MessagePack is a binary serialization format that's:
1. COMPACT: 50-80% smaller than JSON for typical payloads
2. FAST: Binary parsing is faster than text parsing
3. TYPE-RICH: Supports binary data, unlike JSON
4. COMPATIBLE: Easy conversion to/from Python dicts

WHY JSON FALLBACK?
==================
JSON is used when:
1. MessagePack library is not installed
2. Debug mode is enabled (human-readable)
3. Interoperating with systems that don't support MessagePack

SERIALIZATION FLOW:
===================
    Python dict/list → Serializer → bytes → network/storage
    bytes → Deserializer → Python dict/list

The serializers handle:
- Encoding/decoding to appropriate format
- Type conversion (e.g., bytes to base64 for JSON)
- Error handling with meaningful messages
"""

import json
import base64
import zlib
from typing import Any, Dict, Union, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

from .constants import (
    FLAG_PAYLOAD_MSGPACK,
    FLAG_PAYLOAD_JSON,
    FLAG_PAYLOAD_STRING,
    FLAG_COMPRESSED,
    get_payload_type,
    is_compressed,
)
from .exceptions import (
    EncodeError,
    DecodeError,
)


# Try to import msgpack, gracefully fall back to JSON
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False


class Serializer(ABC):
    """Abstract base class for serializers."""

    @abstractmethod
    def encode(self, data: Any) -> bytes:
        """Encode data to bytes."""
        pass

    @abstractmethod
    def decode(self, data: bytes) -> Any:
        """Decode bytes to data."""
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Human-readable format name."""
        pass

    @property
    @abstractmethod
    def flag(self) -> int:
        """Protocol flag for this format."""
        pass


class MessagePackSerializer(Serializer):
    """
    MessagePack serializer implementation.

    MessagePack is a binary serialization format that's efficient
    and supports all Python types we need (including bytes).

    Options:
        use_bin_type: True to preserve bytes/str distinction
        strict_map_key: False to allow non-string keys
    """

    def __init__(self, strict_map_key: bool = False):
        """
        Initialize MessagePack serializer.

        Args:
            strict_map_key: If True, only allow string map keys
        """
        if not MSGPACK_AVAILABLE:
            raise ImportError(
                "msgpack is required for MessagePackSerializer. "
                "Install with: pip install msgpack"
            )
        self._strict_map_key = strict_map_key

    def encode(self, data: Any) -> bytes:
        """
        Encode data to MessagePack bytes.

        Args:
            data: Python object to encode

        Returns:
            MessagePack-encoded bytes

        Raises:
            EncodeError: If encoding fails
        """
        try:
            # Custom encoder for special types
            def default_encoder(obj):
                if isinstance(obj, datetime):
                    return {'__datetime__': obj.isoformat()}
                elif isinstance(obj, Enum):
                    return {'__enum__': obj.value}
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                raise TypeError(f"Cannot serialize {type(obj)}")

            return msgpack.packb(
                data,
                use_bin_type=True,
                default=default_encoder,
                strict_map_key=self._strict_map_key,
            )
        except Exception as e:
            raise EncodeError(
                f"MessagePack encoding failed: {e}",
                format_type="msgpack",
                original_error=e
            )

    def decode(self, data: bytes) -> Any:
        """
        Decode MessagePack bytes to Python object.

        Args:
            data: MessagePack-encoded bytes

        Returns:
            Decoded Python object

        Raises:
            DecodeError: If decoding fails
        """
        try:
            # Custom decoder for special types
            def object_hook(obj):
                if isinstance(obj, dict):
                    if '__datetime__' in obj:
                        return datetime.fromisoformat(obj['__datetime__'])
                    # Note: Enum reconstruction requires type info not stored
                return obj

            result = msgpack.unpackb(
                data,
                raw=False,
                strict_map_key=self._strict_map_key,
                object_hook=object_hook,
            )
            return result
        except Exception as e:
            raise DecodeError(
                f"MessagePack decoding failed: {e}",
                format_type="msgpack",
                original_error=e
            )

    @property
    def format_name(self) -> str:
        return "msgpack"

    @property
    def flag(self) -> int:
        return FLAG_PAYLOAD_MSGPACK


class JSONSerializer(Serializer):
    """
    JSON serializer implementation.

    JSON is used as a fallback when MessagePack is unavailable
    or when human-readable output is needed (debugging).

    Note: JSON doesn't support binary data directly, so bytes
    are base64-encoded with a special prefix.
    """

    BINARY_PREFIX = '__base64__:'

    def __init__(self, indent: Optional[int] = None, sort_keys: bool = False):
        """
        Initialize JSON serializer.

        Args:
            indent: Indentation for pretty-printing (None for compact)
            sort_keys: Whether to sort dictionary keys
        """
        self._indent = indent
        self._sort_keys = sort_keys

    def encode(self, data: Any) -> bytes:
        """
        Encode data to JSON bytes.

        Binary data is base64-encoded with a special prefix.

        Args:
            data: Python object to encode

        Returns:
            UTF-8 encoded JSON bytes

        Raises:
            EncodeError: If encoding fails
        """
        try:
            def default_encoder(obj):
                if isinstance(obj, bytes):
                    # Encode bytes as base64 with prefix
                    b64 = base64.b64encode(obj).decode('ascii')
                    return f"{self.BINARY_PREFIX}{b64}"
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, Enum):
                    return obj.value
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                raise TypeError(f"Cannot serialize {type(obj)}")

            json_str = json.dumps(
                data,
                default=default_encoder,
                indent=self._indent,
                sort_keys=self._sort_keys,
                ensure_ascii=False,
            )
            return json_str.encode('utf-8')
        except Exception as e:
            raise EncodeError(
                f"JSON encoding failed: {e}",
                format_type="json",
                original_error=e
            )

    def decode(self, data: bytes) -> Any:
        """
        Decode JSON bytes to Python object.

        Base64-encoded strings with the special prefix are
        decoded back to bytes.

        Args:
            data: UTF-8 encoded JSON bytes

        Returns:
            Decoded Python object

        Raises:
            DecodeError: If decoding fails
        """
        try:
            json_str = data.decode('utf-8')

            def object_hook(obj):
                if isinstance(obj, dict):
                    return {
                        k: self._decode_value(v)
                        for k, v in obj.items()
                    }
                return obj

            result = json.loads(json_str, object_hook=object_hook)
            return self._decode_value(result)
        except Exception as e:
            raise DecodeError(
                f"JSON decoding failed: {e}",
                format_type="json",
                original_error=e
            )

    def _decode_value(self, value: Any) -> Any:
        """Recursively decode special values."""
        if isinstance(value, str) and value.startswith(self.BINARY_PREFIX):
            b64 = value[len(self.BINARY_PREFIX):]
            return base64.b64decode(b64)
        elif isinstance(value, list):
            return [self._decode_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._decode_value(v) for k, v in value.items()}
        return value

    @property
    def format_name(self) -> str:
        return "json"

    @property
    def flag(self) -> int:
        return FLAG_PAYLOAD_JSON


class StringSerializer(Serializer):
    """
    Simple UTF-8 string serializer.

    Used when the payload is a plain string, no structured data.
    """

    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize string serializer.

        Args:
            encoding: Character encoding to use
        """
        self._encoding = encoding

    def encode(self, data: str) -> bytes:
        """
        Encode string to bytes.

        Args:
            data: String to encode

        Returns:
            Encoded bytes

        Raises:
            EncodeError: If not a string or encoding fails
        """
        if not isinstance(data, str):
            raise EncodeError(
                f"StringSerializer requires str, got {type(data)}",
                format_type="string"
            )
        try:
            return data.encode(self._encoding)
        except Exception as e:
            raise EncodeError(
                f"String encoding failed: {e}",
                format_type="string",
                original_error=e
            )

    def decode(self, data: bytes) -> str:
        """
        Decode bytes to string.

        Args:
            data: Bytes to decode

        Returns:
            Decoded string

        Raises:
            DecodeError: If decoding fails
        """
        try:
            return data.decode(self._encoding)
        except Exception as e:
            raise DecodeError(
                f"String decoding failed: {e}",
                format_type="string",
                original_error=e
            )

    @property
    def format_name(self) -> str:
        return "string"

    @property
    def flag(self) -> int:
        return FLAG_PAYLOAD_STRING


# =============================================================================
# COMPRESSION UTILITIES
# =============================================================================

def compress(data: bytes, level: int = 6) -> bytes:
    """
    Compress data using zlib.

    Args:
        data: Data to compress
        level: Compression level (0-9, default 6)

    Returns:
        Compressed data
    """
    return zlib.compress(data, level)


def decompress(data: bytes) -> bytes:
    """
    Decompress zlib-compressed data.

    Args:
        data: Compressed data

    Returns:
        Decompressed data

    Raises:
        DecodeError: If decompression fails
    """
    try:
        return zlib.decompress(data)
    except zlib.error as e:
        raise DecodeError(
            f"Decompression failed: {e}",
            format_type="zlib",
            original_error=e
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_serializer(flags: int) -> Serializer:
    """
    Get appropriate serializer based on payload flags.

    Args:
        flags: Protocol flags from header

    Returns:
        Serializer instance

    Raises:
        ValueError: If payload type is unknown
    """
    payload_type = get_payload_type(flags)

    if payload_type == FLAG_PAYLOAD_MSGPACK:
        if MSGPACK_AVAILABLE:
            return MessagePackSerializer()
        else:
            # Fall back to JSON if msgpack not available
            return JSONSerializer()
    elif payload_type == FLAG_PAYLOAD_JSON:
        return JSONSerializer()
    elif payload_type == FLAG_PAYLOAD_STRING:
        return StringSerializer()
    else:
        raise ValueError(f"Unknown payload type: {payload_type}")


def get_default_serializer() -> Serializer:
    """
    Get the default serializer (MessagePack if available, else JSON).

    Returns:
        Default Serializer instance
    """
    if MSGPACK_AVAILABLE:
        return MessagePackSerializer()
    return JSONSerializer()


def serialize_payload(data: Any, flags: int) -> bytes:
    """
    Serialize data according to flags.

    Handles serialization and optional compression.

    Args:
        data: Data to serialize
        flags: Protocol flags specifying format and compression

    Returns:
        Serialized (and possibly compressed) bytes
    """
    serializer = get_serializer(flags)
    encoded = serializer.encode(data)

    if is_compressed(flags):
        encoded = compress(encoded)

    return encoded


def deserialize_payload(data: bytes, flags: int) -> Any:
    """
    Deserialize data according to flags.

    Handles decompression and deserialization.

    Args:
        data: Serialized bytes
        flags: Protocol flags specifying format and compression

    Returns:
        Deserialized Python object
    """
    if is_compressed(flags):
        data = decompress(data)

    serializer = get_serializer(flags)
    return serializer.decode(data)
