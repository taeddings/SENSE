"""
Tests for SENSE Binary Protocol

Comprehensive test suite covering:
- DRGNHeader packing/unpacking
- BinaryParser zero-copy operations
- SENSEMessage creation and parsing
- Serialization (MessagePack/JSON)
- CRC32 integrity verification
- Error handling and bounds checking
"""

import pytest
import struct
import zlib
from datetime import datetime

from sense_v2.protocol import (
    # Constants
    MAGIC_SIGNATURE,
    MAGIC_SIGNATURE_INT,
    PROTOCOL_VERSION,
    HEADER_SIZE,
    HEADER_FORMAT,
    FLAG_PAYLOAD_MSGPACK,
    FLAG_PAYLOAD_JSON,
    FLAG_PAYLOAD_STRING,
    FLAG_COMPRESSED,
    FLAG_RESPONSE,
    FLAG_ERROR,
    MAX_MESSAGE_SIZE,
    MAX_STRING_LENGTH,
    METHOD_ID_AGENT_USER,
    METHOD_ID_PING,

    # Header
    DRGNHeader,
    compute_crc32,
    verify_crc32,

    # Parser
    BinaryParser,
    BufferBuilder,

    # Serializers
    JSONSerializer,
    StringSerializer,
    MSGPACK_AVAILABLE,
    serialize_payload,
    deserialize_payload,
    compress,
    decompress,

    # Message
    SENSEMessage,
    generate_message_id,
    create_ping,
    create_pong,

    # Exceptions
    InvalidSignatureError,
    UnsupportedVersionError,
    IncompleteMessageError,
    MessageTooLargeError,
    MalformedMessageError,
    CRCMismatchError,
)


class TestConstants:
    """Test protocol constants."""

    def test_header_size(self):
        """Verify header size matches struct format."""
        assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE
        assert HEADER_SIZE == 29

    def test_magic_signature(self):
        """Verify magic signature bytes."""
        assert MAGIC_SIGNATURE == b'DRGN'
        assert MAGIC_SIGNATURE_INT == 0x4452474E

    def test_protocol_version(self):
        """Verify initial protocol version."""
        assert PROTOCOL_VERSION == 0x00


class TestDRGNHeader:
    """Test DRGNHeader class."""

    def test_create_header(self):
        """Test creating a header with factory method."""
        header = DRGNHeader.create(
            method_id=0x1234,
            payload_size=100,
            flags=FLAG_PAYLOAD_MSGPACK,
            message_id=42,
            crc32=0xDEADBEEF,
        )

        assert header.signature == MAGIC_SIGNATURE_INT
        assert header.protocol_version == PROTOCOL_VERSION
        assert header.method_id == 0x1234
        assert header.flags == FLAG_PAYLOAD_MSGPACK
        assert header.message_id == 42
        assert header.crc32 == 0xDEADBEEF
        assert header.payload_size == 100

    def test_pack_unpack_roundtrip(self):
        """Test header serialization roundtrip."""
        original = DRGNHeader.create(
            method_id=0xABCD,
            payload_size=256,
            flags=FLAG_PAYLOAD_JSON | FLAG_COMPRESSED,
            message_id=999999,
            crc32=0x12345678,
        )

        packed = original.pack()
        assert len(packed) == HEADER_SIZE

        unpacked = DRGNHeader.unpack(packed)
        assert unpacked.signature == original.signature
        assert unpacked.total_bytes == original.total_bytes
        assert unpacked.protocol_version == original.protocol_version
        assert unpacked.method_id == original.method_id
        assert unpacked.flags == original.flags
        assert unpacked.message_id == original.message_id
        assert unpacked.crc32 == original.crc32

    def test_invalid_signature_error(self):
        """Test that invalid signature raises error."""
        # Create buffer with wrong signature
        bad_data = b'XXXX' + b'\x00' * (HEADER_SIZE - 4)

        with pytest.raises(InvalidSignatureError) as exc_info:
            DRGNHeader.unpack(bad_data)

        assert 'XXXX' in str(exc_info.value.received.hex()) or 'Invalid' in str(exc_info.value)

    def test_incomplete_header_error(self):
        """Test that incomplete data raises error."""
        partial_data = b'DRGN' + b'\x00' * 10  # Only 14 bytes

        with pytest.raises(IncompleteMessageError) as exc_info:
            DRGNHeader.unpack(partial_data)

        assert exc_info.value.needed == HEADER_SIZE

    def test_message_too_large_error(self):
        """Test that oversized message is rejected."""
        # Create header claiming huge payload
        huge_size = MAX_MESSAGE_SIZE + 1000
        header_bytes = struct.pack(
            HEADER_FORMAT,
            MAGIC_SIGNATURE_INT,
            huge_size,
            PROTOCOL_VERSION,
            0x0001,
            0,
            0,
            0,
        )

        with pytest.raises(MessageTooLargeError) as exc_info:
            DRGNHeader.unpack(header_bytes)

        assert exc_info.value.limit == MAX_MESSAGE_SIZE

    def test_with_crc32(self):
        """Test creating header with CRC computed from payload."""
        header = DRGNHeader.create(
            method_id=1,
            payload_size=5,
            crc32=0,
        )

        payload = b'hello'
        header_with_crc = header.with_crc32(payload)

        expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
        assert header_with_crc.crc32 == expected_crc

    def test_verify_crc32(self):
        """Test CRC verification."""
        payload = b'test payload'
        crc = compute_crc32(payload)

        header = DRGNHeader.create(
            method_id=1,
            payload_size=len(payload),
            crc32=crc,
        )

        assert header.verify_crc32(payload) is True
        assert header.verify_crc32(b'wrong payload') is False


class TestBinaryParser:
    """Test BinaryParser class."""

    def test_read_fixed_values(self):
        """Test reading fixed-size values."""
        data = struct.pack('!IHBQ', 0x12345678, 0xABCD, 0x42, 0xDEADBEEFCAFEBABE)
        parser = BinaryParser(data)

        assert parser.read_uint32() == 0x12345678
        assert parser.read_uint16() == 0xABCD
        assert parser.read_uint8() == 0x42
        assert parser.read_uint64() == 0xDEADBEEFCAFEBABE
        assert parser.is_exhausted

    def test_read_bytes_zero_copy(self):
        """Test that read_bytes returns memoryview (no copy)."""
        data = b'hello world'
        parser = BinaryParser(data)

        view = parser.read_bytes(5)
        assert isinstance(view, memoryview)
        assert bytes(view) == b'hello'

    def test_read_string(self):
        """Test reading and decoding strings."""
        text = "Hello, World!"
        data = text.encode('utf-8')
        parser = BinaryParser(data)

        result = parser.read_string(len(data))
        assert result == text

    def test_bounds_checking(self):
        """Test that out-of-bounds reads raise errors."""
        data = b'short'
        parser = BinaryParser(data)

        with pytest.raises(IncompleteMessageError):
            parser.read_bytes(100)

    def test_peek_doesnt_advance(self):
        """Test that peek doesn't advance cursor."""
        data = b'hello'
        parser = BinaryParser(data)

        peeked = parser.peek(3)
        assert bytes(peeked) == b'hel'
        assert parser.position == 0

        read = parser.read_bytes(3)
        assert bytes(read) == b'hel'
        assert parser.position == 3

    def test_skip(self):
        """Test skipping bytes."""
        data = b'hello world'
        parser = BinaryParser(data)

        parser.skip(6)
        remaining = parser.read_remaining()
        assert bytes(remaining) == b'world'

    def test_seek(self):
        """Test seeking to position."""
        data = b'0123456789'
        parser = BinaryParser(data)

        parser.seek(5)
        assert parser.read_uint8() == ord('5')

    def test_sub_parser(self):
        """Test creating sub-parser for nested structures."""
        data = b'header' + b'payload_data'
        parser = BinaryParser(data)

        parser.skip(6)  # Skip header
        sub_parser = parser.create_sub_parser(12)

        assert sub_parser.read_string(12) == 'payload_data'


class TestBufferBuilder:
    """Test BufferBuilder class."""

    def test_build_simple_buffer(self):
        """Test building a simple buffer."""
        builder = BufferBuilder()
        builder.write_uint32(0x12345678)
        builder.write_uint16(0xABCD)
        builder.write_bytes(b'hello')

        result = builder.to_bytes()
        expected = struct.pack('!IH', 0x12345678, 0xABCD) + b'hello'
        assert result == expected

    def test_write_string_with_length_prefix(self):
        """Test writing length-prefixed string."""
        builder = BufferBuilder()
        builder.write_string("hello", length_prefix_format='!I')

        result = builder.to_bytes()
        assert result == struct.pack('!I', 5) + b'hello'

    def test_builder_size_tracking(self):
        """Test that builder tracks size correctly."""
        builder = BufferBuilder()
        assert builder.size == 0

        builder.write_uint32(1)
        assert builder.size == 4

        builder.write_bytes(b'test')
        assert builder.size == 8


class TestSerializers:
    """Test serialization formats."""

    def test_json_serializer_roundtrip(self):
        """Test JSON serialization roundtrip."""
        serializer = JSONSerializer()
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        encoded = serializer.encode(data)
        decoded = serializer.decode(encoded)

        assert decoded == data

    def test_json_serializer_handles_bytes(self):
        """Test JSON serializer handles binary data."""
        serializer = JSONSerializer()
        data = {"binary": b'\x00\x01\x02\x03'}

        encoded = serializer.encode(data)
        decoded = serializer.decode(encoded)

        assert decoded["binary"] == data["binary"]

    def test_string_serializer(self):
        """Test string serializer."""
        serializer = StringSerializer()
        text = "Hello, World!"

        encoded = serializer.encode(text)
        decoded = serializer.decode(encoded)

        assert decoded == text

    @pytest.mark.skipif(not MSGPACK_AVAILABLE, reason="msgpack not installed")
    def test_msgpack_serializer_roundtrip(self):
        """Test MessagePack serialization roundtrip."""
        from sense_v2.protocol import MessagePackSerializer

        serializer = MessagePackSerializer()
        data = {"key": "value", "bytes": b'\x00\x01', "nested": {"a": 1}}

        encoded = serializer.encode(data)
        decoded = serializer.decode(encoded)

        assert decoded["key"] == data["key"]
        assert decoded["bytes"] == data["bytes"]

    def test_compression(self):
        """Test compression/decompression."""
        data = b'A' * 1000  # Highly compressible
        compressed = compress(data)
        decompressed = decompress(compressed)

        assert decompressed == data
        assert len(compressed) < len(data)


class TestSENSEMessage:
    """Test SENSEMessage class."""

    def test_create_request(self):
        """Test creating a request message."""
        msg = SENSEMessage.create_request(
            method_id=METHOD_ID_AGENT_USER,
            payload={"content": "Hello!"},
        )

        assert msg.method_id == METHOD_ID_AGENT_USER
        assert msg.is_request
        assert not msg.is_response
        assert msg.payload == {"content": "Hello!"}

    def test_create_response(self):
        """Test creating a response message."""
        request = SENSEMessage.create_request(
            method_id=METHOD_ID_PING,
            payload={},
        )

        response = SENSEMessage.create_response(
            request=request,
            payload={"pong": True},
        )

        assert response.is_response
        assert response.message_id == request.message_id
        assert response.payload == {"pong": True}

    def test_message_roundtrip(self):
        """Test message serialization roundtrip."""
        original = SENSEMessage.create_request(
            method_id=0x1234,
            payload={"test": "data", "number": 42},
        )

        wire_bytes = original.to_bytes()
        parsed = SENSEMessage.parse(wire_bytes)

        assert parsed.method_id == original.method_id
        assert parsed.message_id == original.message_id
        assert parsed.payload == original.payload

    def test_crc_verification(self):
        """Test CRC verification on parse."""
        msg = SENSEMessage.create_request(
            method_id=1,
            payload="test",
        )

        wire_bytes = bytearray(msg.to_bytes())

        # Corrupt the payload
        wire_bytes[-1] ^= 0xFF

        with pytest.raises(CRCMismatchError):
            SENSEMessage.parse(bytes(wire_bytes), verify_crc=True)

    def test_lazy_payload_parsing(self):
        """Test lazy payload deserialization."""
        msg = SENSEMessage.create_request(
            method_id=1,
            payload={"key": "value"},
        )

        wire_bytes = msg.to_bytes()
        parsed = SENSEMessage.parse(wire_bytes, lazy_payload=True)

        # Payload should be None until accessed
        assert parsed.payload is None
        assert parsed.raw_payload is not None

        # get_payload should deserialize
        payload = parsed.get_payload()
        assert payload == {"key": "value"}

    def test_error_response(self):
        """Test creating error response."""
        request = SENSEMessage.create_request(method_id=1, payload={})
        error_response = SENSEMessage.create_error_response(
            request=request,
            error_message="Something went wrong",
            error_code=500,
        )

        assert error_response.is_error
        assert error_response.payload["error"] == "Something went wrong"
        assert error_response.payload["error_code"] == 500

    def test_try_parse_incomplete(self):
        """Test try_parse with incomplete data."""
        msg, consumed = SENSEMessage.try_parse(b'DRG')  # Too short
        assert msg is None
        assert consumed == 0

    def test_try_parse_complete(self):
        """Test try_parse with complete message."""
        original = SENSEMessage.create_request(method_id=1, payload="test")
        wire_bytes = original.to_bytes()

        # Add extra garbage at the end
        data = wire_bytes + b'extra garbage'

        msg, consumed = SENSEMessage.try_parse(data)
        assert msg is not None
        assert consumed == len(wire_bytes)
        assert msg.payload == "test"


class TestMessageIdGeneration:
    """Test message ID generation."""

    def test_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = [generate_message_id() for _ in range(1000)]
        assert len(set(ids)) == 1000

    def test_auto_id_in_request(self):
        """Test that requests get auto-generated IDs."""
        msg1 = SENSEMessage.create_request(method_id=1, payload={})
        msg2 = SENSEMessage.create_request(method_id=1, payload={})

        assert msg1.message_id != msg2.message_id


class TestPingPong:
    """Test ping/pong helpers."""

    def test_create_ping(self):
        """Test creating ping message."""
        ping = create_ping()
        assert ping.method_id == METHOD_ID_PING
        assert "timestamp" in ping.payload

    def test_create_pong(self):
        """Test creating pong response."""
        ping = create_ping()
        pong = create_pong(ping)

        assert pong.is_response
        assert pong.message_id == ping.message_id
        assert "timestamp" in pong.payload


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_payload(self):
        """Test message with empty payload."""
        msg = SENSEMessage.create_request(method_id=1, payload={})
        wire_bytes = msg.to_bytes()
        parsed = SENSEMessage.parse(wire_bytes)

        assert parsed.payload == {}

    def test_string_payload(self):
        """Test message with string payload."""
        msg = SENSEMessage.create_request(
            method_id=1,
            payload="plain string",
            flags=FLAG_PAYLOAD_STRING,
        )
        wire_bytes = msg.to_bytes()
        parsed = SENSEMessage.parse(wire_bytes)

        assert parsed.payload == "plain string"

    def test_large_payload(self):
        """Test message with large payload."""
        large_data = {"data": "x" * 100000}
        msg = SENSEMessage.create_request(method_id=1, payload=large_data)

        wire_bytes = msg.to_bytes()
        parsed = SENSEMessage.parse(wire_bytes)

        assert parsed.payload == large_data

    def test_nested_payload(self):
        """Test message with deeply nested payload."""
        nested = {"level": 0}
        current = nested
        for i in range(1, 20):
            current["nested"] = {"level": i}
            current = current["nested"]

        msg = SENSEMessage.create_request(method_id=1, payload=nested)
        wire_bytes = msg.to_bytes()
        parsed = SENSEMessage.parse(wire_bytes)

        assert parsed.payload["level"] == 0


class TestCRC32:
    """Test CRC32 functions."""

    def test_compute_crc32(self):
        """Test CRC32 computation matches zlib."""
        data = b'test data for crc'
        computed = compute_crc32(data)
        expected = zlib.crc32(data) & 0xFFFFFFFF

        assert computed == expected

    def test_verify_crc32(self):
        """Test CRC32 verification."""
        data = b'test data'
        crc = compute_crc32(data)

        assert verify_crc32(data, crc) is True
        assert verify_crc32(data, crc + 1) is False
        assert verify_crc32(b'wrong data', crc) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
