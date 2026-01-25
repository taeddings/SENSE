"""
SENSE Protocol Async I/O Support

Provides async/await patterns for non-blocking protocol I/O operations.

WHY ASYNC I/O?
==============
Synchronous I/O blocks the thread while waiting for data. In Python's
asyncio model, we can handle thousands of connections in a single thread
by yielding control while waiting for I/O.

This is critical for:
1. HIGH CONCURRENCY: Handle many agents/connections simultaneously
2. RESPONSIVE UI: Don't freeze while waiting for network
3. RESOURCE EFFICIENCY: Single thread handles many connections

STREAMING PROTOCOL:
===================
Network data arrives in chunks that may not align with message boundaries.
A single TCP packet might contain:
- Part of a message header
- Multiple complete messages
- A message that spans multiple packets

This module handles message framing:
    Network chunks → Framer → Complete messages

USAGE:
======
    # Reading messages
    async with AsyncMessageReader(reader) as reader:
        async for message in reader:
            await handle_message(message)

    # Writing messages
    async with AsyncMessageWriter(writer) as writer:
        await writer.send(message)
"""

import asyncio
from typing import AsyncIterator, Optional, Union
from dataclasses import dataclass, field

from .constants import (
    HEADER_SIZE,
    MAX_MESSAGE_SIZE,
    ASYNC_BUFFER_SIZE,
    ASYNC_READ_TIMEOUT,
    ASYNC_WRITE_TIMEOUT,
)
from .header import DRGNHeader
from .message import SENSEMessage
from .exceptions import (
    IncompleteMessageError,
    MessageTooLargeError,
    ReadTimeoutError,
    WriteTimeoutError,
    ConnectionClosedError,
    ProtocolError,
)


@dataclass
class AsyncMessageReader:
    """
    Async message reader with automatic framing.

    Reads complete SENSE messages from an asyncio StreamReader,
    handling partial reads and message boundaries.

    Example:
        reader, writer = await asyncio.open_connection('localhost', 8080)
        async with AsyncMessageReader(reader) as msg_reader:
            async for message in msg_reader:
                print(f"Received: {message}")
    """

    stream: asyncio.StreamReader
    verify_crc: bool = True
    read_timeout: float = ASYNC_READ_TIMEOUT
    max_message_size: int = MAX_MESSAGE_SIZE
    _buffer: bytearray = field(default_factory=bytearray, init=False)
    _closed: bool = field(default=False, init=False)

    async def __aenter__(self) -> 'AsyncMessageReader':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        self._closed = True
        return False

    async def read_message(self) -> SENSEMessage:
        """
        Read a single complete message.

        Blocks (asynchronously) until a complete message is available.

        Returns:
            Complete SENSEMessage

        Raises:
            ReadTimeoutError: If read times out
            ConnectionClosedError: If connection is closed
            ProtocolError: If message is invalid
        """
        # First, read header
        while len(self._buffer) < HEADER_SIZE:
            await self._read_more()

        # Parse header to get total size
        header = DRGNHeader.unpack(bytes(self._buffer[:HEADER_SIZE]))

        # Validate message size
        total_size = header.total_bytes + 4
        if total_size > self.max_message_size:
            raise MessageTooLargeError(
                total_size,
                self.max_message_size,
                field="message"
            )

        # Read remaining payload
        while len(self._buffer) < total_size:
            await self._read_more()

        # Extract complete message
        message_bytes = bytes(self._buffer[:total_size])
        del self._buffer[:total_size]

        return SENSEMessage.parse(message_bytes, verify_crc=self.verify_crc)

    async def _read_more(self) -> None:
        """
        Read more data from stream into buffer.

        Raises:
            ReadTimeoutError: If read times out
            ConnectionClosedError: If connection is closed
        """
        if self._closed:
            raise ConnectionClosedError()

        try:
            data = await asyncio.wait_for(
                self.stream.read(ASYNC_BUFFER_SIZE),
                timeout=self.read_timeout
            )
        except asyncio.TimeoutError:
            raise ReadTimeoutError(self.read_timeout)

        if not data:
            raise ConnectionClosedError()

        self._buffer.extend(data)

    def __aiter__(self) -> 'AsyncMessageReader':
        """Make this an async iterator."""
        return self

    async def __anext__(self) -> SENSEMessage:
        """
        Get next message (async iterator protocol).

        Raises:
            StopAsyncIteration: When connection is closed
        """
        try:
            return await self.read_message()
        except ConnectionClosedError:
            raise StopAsyncIteration


@dataclass
class AsyncMessageWriter:
    """
    Async message writer with flow control.

    Writes SENSE messages to an asyncio StreamWriter with
    proper draining to handle backpressure.

    Example:
        reader, writer = await asyncio.open_connection('localhost', 8080)
        async with AsyncMessageWriter(writer) as msg_writer:
            await msg_writer.send(message)
    """

    stream: asyncio.StreamWriter
    write_timeout: float = ASYNC_WRITE_TIMEOUT
    _closed: bool = field(default=False, init=False)

    async def __aenter__(self) -> 'AsyncMessageWriter':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit - close the stream."""
        await self.close()
        return False

    async def send(self, message: SENSEMessage) -> None:
        """
        Send a message over the stream.

        Handles serialization and flow control (draining).

        Args:
            message: Message to send

        Raises:
            WriteTimeoutError: If write times out
            ConnectionClosedError: If connection is closed
        """
        if self._closed:
            raise ConnectionClosedError("Writer is closed")

        try:
            data = message.to_bytes()
            self.stream.write(data)
            await asyncio.wait_for(
                self.stream.drain(),
                timeout=self.write_timeout
            )
        except asyncio.TimeoutError:
            raise WriteTimeoutError(self.write_timeout, len(data))
        except ConnectionResetError:
            raise ConnectionClosedError("Connection reset by peer")

    async def send_request(
        self,
        method_id: int,
        payload: any,
        message_id: Optional[int] = None,
    ) -> int:
        """
        Create and send a request message.

        Args:
            method_id: RPC method identifier
            payload: Request payload
            message_id: Optional message ID

        Returns:
            The message ID used (for correlation)
        """
        msg = SENSEMessage.create_request(
            method_id=method_id,
            payload=payload,
            message_id=message_id,
        )
        await self.send(msg)
        return msg.message_id

    async def send_response(
        self,
        request: SENSEMessage,
        payload: any,
        is_error: bool = False,
    ) -> None:
        """
        Create and send a response to a request.

        Args:
            request: The original request
            payload: Response payload
            is_error: Whether this is an error response
        """
        msg = SENSEMessage.create_response(request, payload, is_error)
        await self.send(msg)

    async def close(self) -> None:
        """Close the stream gracefully."""
        if not self._closed:
            self._closed = True
            self.stream.close()
            try:
                await self.stream.wait_closed()
            except Exception:
                pass  # Ignore errors during close


class AsyncMessageChannel:
    """
    Bidirectional async message channel.

    Combines reader and writer into a single interface for
    request-response patterns.

    Example:
        channel = await AsyncMessageChannel.connect('localhost', 8080)
        response = await channel.request(METHOD_ID_PING, {"timestamp": time.time()})
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        verify_crc: bool = True,
    ):
        """
        Initialize channel with streams.

        Args:
            reader: Async stream reader
            writer: Async stream writer
            verify_crc: Whether to verify CRC on incoming messages
        """
        self._reader = AsyncMessageReader(reader, verify_crc=verify_crc)
        self._writer = AsyncMessageWriter(writer)
        self._pending: dict[int, asyncio.Future] = {}
        self._closed = False
        self._receive_task: Optional[asyncio.Task] = None

    @classmethod
    async def connect(
        cls,
        host: str,
        port: int,
        verify_crc: bool = True,
    ) -> 'AsyncMessageChannel':
        """
        Connect to a SENSE protocol server.

        Args:
            host: Server hostname
            port: Server port
            verify_crc: Whether to verify CRC on messages

        Returns:
            Connected channel
        """
        reader, writer = await asyncio.open_connection(host, port)
        return cls(reader, writer, verify_crc)

    async def __aenter__(self) -> 'AsyncMessageChannel':
        """Start background receive task."""
        self._receive_task = asyncio.create_task(self._receive_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop receive task and close connection."""
        await self.close()
        return False

    async def _receive_loop(self) -> None:
        """Background task that receives messages and dispatches responses."""
        try:
            async for message in self._reader:
                if message.is_response and message.message_id in self._pending:
                    future = self._pending.pop(message.message_id)
                    future.set_result(message)
        except ConnectionClosedError:
            pass
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel all pending requests
            for future in self._pending.values():
                if not future.done():
                    future.cancel()
            self._pending.clear()

    async def request(
        self,
        method_id: int,
        payload: any,
        timeout: float = 30.0,
    ) -> SENSEMessage:
        """
        Send a request and wait for the response.

        Args:
            method_id: RPC method identifier
            payload: Request payload
            timeout: Response timeout in seconds

        Returns:
            Response message

        Raises:
            asyncio.TimeoutError: If response times out
        """
        msg = SENSEMessage.create_request(method_id=method_id, payload=payload)

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg.message_id] = future

        try:
            await self._writer.send(msg)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg.message_id, None)
            raise
        except asyncio.CancelledError:
            self._pending.pop(msg.message_id, None)
            raise

    async def send(self, message: SENSEMessage) -> None:
        """Send a message without waiting for response."""
        await self._writer.send(message)

    async def close(self) -> None:
        """Close the channel."""
        if not self._closed:
            self._closed = True
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
            await self._writer.close()


async def read_message(
    reader: asyncio.StreamReader,
    verify_crc: bool = True,
    timeout: float = ASYNC_READ_TIMEOUT,
) -> SENSEMessage:
    """
    Standalone function to read a single message.

    Convenience function when you don't need the full AsyncMessageReader.

    Args:
        reader: Async stream reader
        verify_crc: Whether to verify CRC
        timeout: Read timeout

    Returns:
        Parsed message
    """
    async with AsyncMessageReader(reader, verify_crc=verify_crc, read_timeout=timeout) as msg_reader:
        return await msg_reader.read_message()


async def write_message(
    writer: asyncio.StreamWriter,
    message: SENSEMessage,
    timeout: float = ASYNC_WRITE_TIMEOUT,
) -> None:
    """
    Standalone function to write a single message.

    Args:
        writer: Async stream writer
        message: Message to send
        timeout: Write timeout
    """
    msg_writer = AsyncMessageWriter(writer, write_timeout=timeout)
    await msg_writer.send(message)
