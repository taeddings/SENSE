"""
Tests for SENSE Protocol Async I/O

Tests async message reading/writing, channel operations,
and timeout handling.
"""

import pytest
import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

from sense.protocol import (
    SENSEMessage,
    METHOD_ID_PING,
    METHOD_ID_AGENT_USER,
    FLAG_RESPONSE,
)
from sense.protocol.async_io import (
    AsyncMessageReader,
    AsyncMessageWriter,
    AsyncMessageChannel,
    read_message,
    write_message,
)
from sense.protocol.exceptions import (
    ConnectionClosedError,
    ReadTimeoutError,
    WriteTimeoutError,
)


class MockStreamReader:
    """Mock asyncio.StreamReader for testing."""

    def __init__(self, data: bytes = b''):
        self._data = data
        self._position = 0

    async def read(self, n: int) -> bytes:
        """Read up to n bytes."""
        if self._position >= len(self._data):
            return b''
        result = self._data[self._position:self._position + n]
        self._position += len(result)
        return result

    async def readexactly(self, n: int) -> bytes:
        """Read exactly n bytes."""
        if self._position + n > len(self._data):
            raise asyncio.IncompleteReadError(
                self._data[self._position:],
                n
            )
        result = self._data[self._position:self._position + n]
        self._position += n
        return result

    def feed_data(self, data: bytes):
        """Add more data to the stream."""
        self._data = self._data + data


class MockStreamWriter:
    """Mock asyncio.StreamWriter for testing."""

    def __init__(self):
        self._buffer = io.BytesIO()
        self._closed = False

    def write(self, data: bytes):
        """Write data to buffer."""
        if self._closed:
            raise ConnectionResetError("Connection closed")
        self._buffer.write(data)

    async def drain(self):
        """Drain the write buffer."""
        pass

    def close(self):
        """Close the writer."""
        self._closed = True

    async def wait_closed(self):
        """Wait for writer to close."""
        pass

    def get_written_data(self) -> bytes:
        """Get all written data."""
        return self._buffer.getvalue()


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return SENSEMessage.create_request(
        method_id=METHOD_ID_AGENT_USER,
        payload={"content": "Hello, World!"},
    )


@pytest.fixture
def sample_message_bytes(sample_message):
    """Get wire bytes for sample message."""
    return sample_message.to_bytes()


class TestAsyncMessageReader:
    """Test AsyncMessageReader class."""

    @pytest.mark.asyncio
    async def test_read_single_message(self, sample_message_bytes):
        """Test reading a single complete message."""
        reader = MockStreamReader(sample_message_bytes)
        msg_reader = AsyncMessageReader(reader)

        async with msg_reader:
            msg = await msg_reader.read_message()

        assert msg.method_id == METHOD_ID_AGENT_USER
        assert msg.payload["content"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_multiple_messages(self):
        """Test reading multiple messages in sequence."""
        msg1 = SENSEMessage.create_request(method_id=1, payload="first")
        msg2 = SENSEMessage.create_request(method_id=2, payload="second")

        combined = msg1.to_bytes() + msg2.to_bytes()
        reader = MockStreamReader(combined)
        msg_reader = AsyncMessageReader(reader)

        async with msg_reader:
            read1 = await msg_reader.read_message()
            read2 = await msg_reader.read_message()

        assert read1.payload == "first"
        assert read2.payload == "second"

    @pytest.mark.asyncio
    async def test_async_iterator(self, sample_message_bytes):
        """Test using reader as async iterator."""
        # Create data with one message
        reader = MockStreamReader(sample_message_bytes)
        msg_reader = AsyncMessageReader(reader)

        messages = []
        async with msg_reader:
            try:
                async for msg in msg_reader:
                    messages.append(msg)
                    if len(messages) >= 1:
                        break
            except ConnectionClosedError:
                pass

        assert len(messages) == 1
        assert messages[0].method_id == METHOD_ID_AGENT_USER

    @pytest.mark.asyncio
    async def test_connection_closed_error(self):
        """Test that empty read raises ConnectionClosedError."""
        reader = MockStreamReader(b'')  # Empty stream
        msg_reader = AsyncMessageReader(reader)

        async with msg_reader:
            with pytest.raises(ConnectionClosedError):
                await msg_reader.read_message()


class TestAsyncMessageWriter:
    """Test AsyncMessageWriter class."""

    @pytest.mark.asyncio
    async def test_send_message(self, sample_message):
        """Test sending a message."""
        writer = MockStreamWriter()
        msg_writer = AsyncMessageWriter(writer)

        async with msg_writer:
            await msg_writer.send(sample_message)

        written = writer.get_written_data()
        assert len(written) > 0

        # Verify written data is valid message
        parsed = SENSEMessage.parse(written)
        assert parsed.payload == sample_message.payload

    @pytest.mark.asyncio
    async def test_send_request(self):
        """Test send_request convenience method."""
        writer = MockStreamWriter()
        msg_writer = AsyncMessageWriter(writer)

        async with msg_writer:
            msg_id = await msg_writer.send_request(
                method_id=METHOD_ID_PING,
                payload={"test": True},
            )

        assert msg_id > 0

        written = writer.get_written_data()
        parsed = SENSEMessage.parse(written)
        assert parsed.message_id == msg_id
        assert parsed.method_id == METHOD_ID_PING

    @pytest.mark.asyncio
    async def test_send_response(self, sample_message):
        """Test send_response convenience method."""
        writer = MockStreamWriter()
        msg_writer = AsyncMessageWriter(writer)

        async with msg_writer:
            await msg_writer.send_response(
                request=sample_message,
                payload={"response": "ok"},
            )

        written = writer.get_written_data()
        parsed = SENSEMessage.parse(written)
        assert parsed.is_response
        assert parsed.message_id == sample_message.message_id


class TestAsyncMessageChannel:
    """Test AsyncMessageChannel class."""

    @pytest.mark.asyncio
    async def test_channel_creation(self):
        """Test creating a channel."""
        reader = MockStreamReader(b'')
        writer = MockStreamWriter()

        channel = AsyncMessageChannel(reader, writer)
        assert channel is not None

    @pytest.mark.asyncio
    async def test_send_without_response(self, sample_message):
        """Test sending without waiting for response."""
        reader = MockStreamReader(b'')
        writer = MockStreamWriter()

        channel = AsyncMessageChannel(reader, writer)
        await channel.send(sample_message)

        written = writer.get_written_data()
        assert len(written) > 0


class TestStandaloneFunctions:
    """Test standalone async functions."""

    @pytest.mark.asyncio
    async def test_read_message_function(self, sample_message_bytes):
        """Test standalone read_message function."""
        reader = MockStreamReader(sample_message_bytes)

        msg = await read_message(reader)
        assert msg.method_id == METHOD_ID_AGENT_USER

    @pytest.mark.asyncio
    async def test_write_message_function(self, sample_message):
        """Test standalone write_message function."""
        writer = MockStreamWriter()

        await write_message(writer, sample_message)

        written = writer.get_written_data()
        parsed = SENSEMessage.parse(written)
        assert parsed.payload == sample_message.payload


class TestErrorHandling:
    """Test async error handling."""

    @pytest.mark.asyncio
    async def test_closed_writer_error(self, sample_message):
        """Test writing to closed writer raises error."""
        writer = MockStreamWriter()
        msg_writer = AsyncMessageWriter(writer)

        await msg_writer.close()

        with pytest.raises(ConnectionClosedError):
            await msg_writer.send(sample_message)


class TestMessageFraming:
    """Test message framing with partial data."""

    @pytest.mark.asyncio
    async def test_handles_partial_reads(self):
        """Test that reader handles data arriving in chunks."""
        msg = SENSEMessage.create_request(method_id=1, payload="test")
        full_data = msg.to_bytes()

        # Create reader that returns data in small chunks
        class ChunkedReader:
            def __init__(self, data, chunk_size=5):
                self._data = data
                self._chunk_size = chunk_size
                self._pos = 0

            async def read(self, n):
                if self._pos >= len(self._data):
                    return b''
                end = min(self._pos + min(n, self._chunk_size), len(self._data))
                result = self._data[self._pos:end]
                self._pos = end
                return result

        reader = ChunkedReader(full_data, chunk_size=5)
        msg_reader = AsyncMessageReader(reader)

        async with msg_reader:
            received = await msg_reader.read_message()

        assert received.payload == "test"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
