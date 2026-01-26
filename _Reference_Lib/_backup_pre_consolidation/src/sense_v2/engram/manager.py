"""
SENSE Engram Manager

Resource-safe buffer management for memory-mapped engram buffers,
optimized for Termux on Android.

WHY ExitStack?
==============
Termux on Android has limited file descriptors (typically 1024).
If we open mmap files without proper cleanup, the process can
hit the fd limit and crash. ExitStack ensures ALL resources are
closed, even if exceptions occur during processing.

ExitStack is like a "cleanup queue" - you register resources as
you open them, and they get cleaned up in reverse order when
the context exits (whether normally or via exception).

ZERO-COPY BUFFER ACCESS:
========================
Memory-mapped files appear as regular memory to your program.
Instead of:
    1. Read file → copy to buffer → parse buffer

We do:
    1. Map file → parse directly from OS page cache

The OS handles caching, so frequently accessed engrams stay
in RAM without explicit cache management.

USAGE:
======
    # Basic usage with context manager
    with EngramManager(buffer_path) as manager:
        parser = manager.get_parser()
        data = parser.read_fixed('!I')

    # Async usage
    async with AsyncEngramManager(buffer_path) as manager:
        data = await manager.read_at(offset, length)
"""

import mmap
import os
import asyncio
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, AsyncIterator
from pathlib import Path

from sense_v2.protocol.parser import BinaryParser
from sense_v2.core.config import EngramConfig
from sense_v2.protocol.constants import MAX_MESSAGE_SIZE
from sense_v2.protocol.exceptions import BufferError


class EngramManagerError(Exception):
    """Base exception for EngramManager errors."""
    pass


class BufferNotOpenError(EngramManagerError):
    """Raised when trying to use a buffer that isn't open."""
    pass


class BufferAccessError(EngramManagerError):
    """Raised when buffer access fails."""
    pass


@dataclass
class EngramManager:
    """
    Resource-safe manager for memory-mapped engram buffers.

    This class provides safe access to memory-mapped files with
    automatic resource cleanup. It's designed for Termux's
    constrained environment where file descriptor leaks are
    particularly problematic.

    Features:
    - Automatic resource cleanup via ExitStack
    - Zero-copy buffer access via memoryview
    - BinaryParser integration for structured reading
    - Thread-safe (each thread should use its own manager)

    Example:
        with EngramManager('/path/to/engram.dat') as manager:
            # Get a zero-copy parser
            parser = manager.get_parser()

            # Or get a slice
            view = manager.get_slice(0, 1024)
    """

    buffer_path: str
    max_size: int = MAX_MESSAGE_SIZE
    read_only: bool = True

    # Private state (initialized in __enter__)
    _exit_stack: ExitStack = field(default_factory=ExitStack, init=False, repr=False)
    _file: Optional[object] = field(default=None, init=False, repr=False)
    _mmap: Optional[mmap.mmap] = field(default=None, init=False, repr=False)
    _view: Optional[memoryview] = field(default=None, init=False, repr=False)
    _is_open: bool = field(default=False, init=False, repr=False)

    def __enter__(self) -> 'EngramManager':
        """
        Open the buffer for reading.

        Opens the file, creates the memory map, and wraps it in a
        memoryview for zero-copy access. All resources are registered
        with ExitStack for automatic cleanup.

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If buffer file doesn't exist
            EngramManagerError: If mmap creation fails
        """
        try:
            # Open file
            mode = 'rb' if self.read_only else 'r+b'
            self._file = open(self.buffer_path, mode)
            self._exit_stack.enter_context(self._file)

            # Get file size and validate
            file_size = os.fstat(self._file.fileno()).st_size
            if file_size == 0:
                raise EngramManagerError(f"Buffer file is empty: {self.buffer_path}")
            if file_size > self.max_size:
                raise EngramManagerError(
                    f"Buffer file too large: {file_size} > {self.max_size}"
                )

            # Create memory map
            access = mmap.ACCESS_READ if self.read_only else mmap.ACCESS_WRITE
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=access)

            # Note: memoryview created on demand to avoid persistent references

            self._is_open = True
            return self

        except Exception as e:
            # Clean up any resources we've opened
            self._exit_stack.close()
            if isinstance(e, (FileNotFoundError, EngramManagerError)):
                raise
            raise EngramManagerError(f"Failed to open buffer: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Close the buffer and release all resources.

        ExitStack handles closing all registered resources in reverse
        order, ensuring proper cleanup even if exceptions occurred.

        IMPORTANT: We must release the memoryview before closing the mmap,
        otherwise Python will raise "BufferError: cannot close exported
        pointers exist". We clear our reference first, then force garbage
        collection to release any dangling references before closing.

        Returns:
            False (don't suppress exceptions)
        """
        import gc

        self._is_open = False

        # Close mmap manually, catching BufferError if views are still exported
        try:
            if self._mmap:
                self._mmap.close()
        except Exception:
            # External code still holds references to memoryview slices
            # This is not ideal but we can't force them to release
            # The mmap will remain open until process exit
            pass

        # Force garbage collection to release any derived views
        gc.collect()

        self._mmap = None
        self._file = None

        # Close the exit stack
        return self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def _check_open(self) -> None:
        """Verify buffer is open."""
        if not self._is_open:
            raise BufferNotOpenError("Buffer is not open. Use 'with' statement.")

    @property
    def size(self) -> int:
        """Get the buffer size in bytes."""
        self._check_open()
        return self._mmap.size()

    @property
    def view(self) -> memoryview:
        """Get the underlying memoryview (zero-copy)."""
        self._check_open()
        return memoryview(self._mmap)

    def get_parser(self, offset: int = 0) -> BinaryParser:
        """
        Get a zero-copy parser for this buffer.

        Args:
            offset: Starting offset in buffer (default 0)

        Returns:
            BinaryParser instance for the buffer (or slice)
        """
        self._check_open()
        if offset == 0:
            return BinaryParser(memoryview(self._mmap))
        return BinaryParser(memoryview(self._mmap)[offset:])

    def get_slice(self, start: int, end: int) -> memoryview:
        """
        Get a zero-copy slice of the buffer.

        Args:
            start: Start offset
            end: End offset

        Returns:
            memoryview slice
        """
        self._check_open()
        return memoryview(self._mmap)[start:end]

    def read_at(self, offset: int, length: int) -> bytes:
        """
        Read bytes at specific offset (creates copy).

        Use get_slice() for zero-copy access when possible.

        Args:
            offset: Byte offset to read from
            length: Number of bytes to read

        Returns:
            bytes object (copy of data)
        """
        self._check_open()
        return bytes(self._view[offset:offset + length])

    def find(self, needle: bytes, start: int = 0) -> int:
        """
        Find pattern in buffer.

        Args:
            needle: Pattern to search for
            start: Starting offset

        Returns:
            Offset of first occurrence, or -1 if not found
        """
        self._check_open()
        return self._mmap.find(needle, start)


@dataclass
class MultiBufferManager:
    """
    Manager for multiple engram buffers.

    Useful when engram data is split across multiple files
    (e.g., sharded by n-gram order or layer index).

    Example:
        with MultiBufferManager(['shard_0.dat', 'shard_1.dat']) as manager:
            for i, parser in enumerate(manager.get_parsers()):
                process_shard(i, parser)
    """

    buffer_paths: List[str]
    max_size_each: int = MAX_MESSAGE_SIZE

    _managers: List[EngramManager] = field(default_factory=list, init=False, repr=False)
    _exit_stack: ExitStack = field(default_factory=ExitStack, init=False, repr=False)

    def __enter__(self) -> 'MultiBufferManager':
        """Open all buffers."""
        try:
            for path in self.buffer_paths:
                manager = EngramManager(path, max_size=self.max_size_each)
                self._exit_stack.enter_context(manager)
                self._managers.append(manager)
            return self
        except Exception:
            self._exit_stack.close()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Close all buffers."""
        self._managers.clear()
        return self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __getitem__(self, index: int) -> EngramManager:
        """Get a specific buffer manager by index."""
        return self._managers[index]

    def __len__(self) -> int:
        """Number of buffers."""
        return len(self._managers)

    def get_parsers(self) -> List[BinaryParser]:
        """Get parsers for all buffers."""
        return [m.get_parser() for m in self._managers]


class AsyncEngramManager:
    """
    Async wrapper for EngramManager.

    Provides async-friendly access to memory-mapped buffers.
    Operations that might block (like opening files) are run
    in a thread pool to avoid blocking the event loop.

    Example:
        async with AsyncEngramManager('/path/to/engram.dat') as manager:
            data = await manager.read_at(0, 1024)
    """

    def __init__(
        self,
        buffer_path: str,
        max_size: int = MAX_MESSAGE_SIZE,
        executor=None,
    ):
        """
        Initialize async manager.

        Args:
            buffer_path: Path to buffer file
            max_size: Maximum allowed buffer size
            executor: Optional ThreadPoolExecutor (default: asyncio default)
        """
        self.buffer_path = buffer_path
        self.max_size = max_size
        self._executor = executor
        self._manager: Optional[EngramManager] = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> 'AsyncEngramManager':
        """Async context manager entry."""
        loop = asyncio.get_event_loop()

        # Run blocking file operations in thread pool
        self._manager = await loop.run_in_executor(
            self._executor,
            self._open_sync
        )
        return self

    def _open_sync(self) -> EngramManager:
        """Synchronous open (run in executor)."""
        manager = EngramManager(self.buffer_path, max_size=self.max_size)
        manager.__enter__()
        return manager

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        if self._manager:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: self._manager.__exit__(exc_type, exc_val, exc_tb)
            )
            self._manager = None
        return False



    @property
    def size(self) -> int:
        """Buffer size."""
        if not self._manager:
            raise BufferNotOpenError("Buffer not open")
        return self._manager.size

    async def read_at(self, offset: int, length: int) -> bytes:
        """
        Async read at offset.

        For small reads, this is fast enough to run directly.
        For large reads, consider using get_parser() instead.
        """
        async with self._lock:
            if not self._manager:
                raise BufferNotOpenError("Buffer not open")
            return self._manager.read_at(offset, length)

    def get_parser(self, offset: int = 0) -> BinaryParser:
        """
        Get a parser for the buffer.

        Note: The parser itself is synchronous. Use for small,
        sequential reads.
        """
        if not self._manager:
            raise BufferNotOpenError("Buffer not open")
        return self._manager.get_parser(offset)

    def get_view(self) -> memoryview:
        """Get the underlying memoryview."""
        if not self._manager:
            raise BufferNotOpenError("Buffer not open")
        return self._manager.view


def create_buffer_file(
    path: str,
    size: int,
    fill: bytes = b'\x00',
) -> None:
    """
    Create a new buffer file with specified size.

    Useful for initializing engram storage files.

    Args:
        path: Path to create file at
        size: Size in bytes
        fill: Byte to fill with (default: null bytes)
    """
    with open(path, 'wb') as f:
        # Write in chunks to avoid memory issues for large files
        chunk_size = 1024 * 1024  # 1 MB
        remaining = size
        while remaining > 0:
            write_size = min(chunk_size, remaining)
            f.write(fill * write_size)
            remaining -= write_size


def verify_buffer_integrity(path: str, expected_magic: bytes = None) -> bool:
    """
    Verify buffer file integrity.

    Args:
        path: Path to buffer file
        expected_magic: Optional magic bytes to check at start

    Returns:
        True if file exists and passes checks
    """
    if not os.path.exists(path):
        return False

    try:
        with EngramManager(path) as manager:
            if expected_magic:
                magic = manager.read_at(0, len(expected_magic))
                return magic == expected_magic
            return manager.size > 0
    except Exception:
        return False

class EngramMemoryManager:
    """
    High-level memory manager for Engram conditional memory architecture.
    Handles storage, retrieval, and pruning with age and relevance.
    """
    def __init__(self, config: EngramConfig):
        self.config = config
        self.memories: list[dict] = []

    def store_memory(self, memory: dict):
        # Validate memory dict
        required_keys = {'content', 'age', 'relevance', 'timestamp'}
        if not all(key in memory for key in required_keys):
            raise ValueError(f"Invalid memory format: missing keys {required_keys - set(memory.keys())}")
        if memory['relevance'] < self.config.relevance_threshold:
            return  # Conditional storage
        if len(self.memories) >= self.config.max_memories:
            self.prune_old_memories()
        self.memories.append(memory)

    def retrieve_memories(self, query: str, max_results: int = 10) -> list[dict]:
        # TODO: Implement adaptive retrieval with AgeMem
        # For now, simple filter
        relevant = [m for m in self.memories if query.lower() in m.get('content', '').lower()]
        # Apply age decay
        decayed = [m for m in relevant if self._apply_age_decay(m) > self.config.relevance_threshold]
        return decayed[:max_results]

    def prune_old_memories(self):
        # TODO: Implement full decay
        self.memories = [m for m in self.memories if self._apply_age_decay(m) > 0.1]

    def _apply_age_decay(self, memory: dict) -> float:
        age = memory.get('age', 0)
        relevance = memory.get('relevance', 0)
        decayed = relevance * (1 - self.config.age_decay_rate * age)
        return max(0, decayed)
