#!/usr/bin/env python3
"""Quick protocol tests."""

import sys
sys.path.insert(0, '.')

print("Testing SENSE Protocol Implementation")
print("=" * 50)

# Test imports
print("\n[1/6] Testing imports...")
try:
    from sense_v2.protocol import (
        SENSEMessage,
        DRGNHeader,
        BinaryParser,
        BufferBuilder,
        HEADER_SIZE,
        MAGIC_SIGNATURE_INT,
        compute_crc32,
        FLAG_PAYLOAD_MSGPACK,
        METHOD_ID_AGENT_USER,
    )
    from sense_v2.engram.manager import EngramManager
    from sense_v2.core.config import ProtocolConfig
    print("   All imports successful")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/6] Testing DRGNHeader...")
try:
    header = DRGNHeader.create(
        method_id=0x1234,
        payload_size=100,
        message_id=42,
        crc32=0xDEADBEEF,
    )
    assert header.method_id == 0x1234
    assert header.message_id == 42

    packed = header.pack()
    assert len(packed) == HEADER_SIZE

    unpacked = DRGNHeader.unpack(packed)
    assert unpacked.method_id == header.method_id
    assert unpacked.message_id == header.message_id
    print("   DRGNHeader pack/unpack: PASSED")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

print("\n[3/6] Testing BinaryParser...")
try:
    import struct
    data = struct.pack("!IHB", 0x12345678, 0xABCD, 0x42)
    parser = BinaryParser(data)

    assert parser.read_uint32() == 0x12345678
    assert parser.read_uint16() == 0xABCD
    assert parser.read_uint8() == 0x42
    assert parser.is_exhausted
    print("   BinaryParser read: PASSED")

    data2 = b"hello world"
    parser2 = BinaryParser(data2)
    view = parser2.read_bytes(5)
    assert isinstance(view, memoryview)
    assert bytes(view) == b"hello"
    print("   BinaryParser zero-copy: PASSED")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

print("\n[4/6] Testing BufferBuilder...")
try:
    builder = BufferBuilder()
    builder.write_uint32(0x12345678)
    builder.write_uint16(0xABCD)
    builder.write_bytes(b"test")

    result = builder.to_bytes()
    expected = struct.pack("!IH", 0x12345678, 0xABCD) + b"test"
    assert result == expected
    print("   BufferBuilder: PASSED")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

print("\n[5/6] Testing SENSEMessage...")
try:
    msg = SENSEMessage.create_request(
        method_id=METHOD_ID_AGENT_USER,
        payload={"content": "Hello, World!", "number": 42},
    )
    assert msg.method_id == METHOD_ID_AGENT_USER
    assert msg.is_request
    assert not msg.is_response

    wire_bytes = msg.to_bytes()
    assert len(wire_bytes) > HEADER_SIZE

    parsed = SENSEMessage.parse(wire_bytes)
    assert parsed.method_id == msg.method_id
    assert parsed.payload == msg.payload
    print("   SENSEMessage roundtrip: PASSED")

    response = SENSEMessage.create_response(msg, payload={"status": "ok"})
    assert response.is_response
    assert response.message_id == msg.message_id
    print("   SENSEMessage response: PASSED")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

print("\n[6/6] Testing CRC32...")
try:
    import zlib
    data = b"test data for crc verification"
    computed = compute_crc32(data)
    expected = zlib.crc32(data) & 0xFFFFFFFF
    assert computed == expected
    print("   CRC32 compute: PASSED")

    msg = SENSEMessage.create_request(method_id=1, payload="test")
    wire = bytearray(msg.to_bytes())
    wire[-1] ^= 0xFF

    try:
        SENSEMessage.parse(bytes(wire), verify_crc=True)
        print("   CRC verification: FAILED (should have raised)")
        sys.exit(1)
    except Exception as e:
        if "CRC" in str(type(e).__name__):
            print("   CRC verification: PASSED")
        else:
            raise
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("All tests PASSED!")
print("=" * 50)
