"""
Tests for SENSE Engram Manager

Tests memory-mapped buffer management, resource cleanup,
and zero-copy parser integration.
"""

import pytest
import os
import tempfile
import mmap
import asyncio
from pathlib import Path

from sense_v2.engram.manager import (
    EngramManager,
    AsyncEngramManager,
    MultiBufferManager,
    EngramManagerError,
    BufferNotOpenError,
    BufferAccessError,
    create_buffer_file,
    verify_buffer_integrity,
)
from sense_v2.protocol.parser import BinaryParser


@pytest.fixture
def temp_buffer_file():
    """Create a temporary buffer file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as f:
        # Write some test data
        f.write(b'DRGN')  # Magic signature
        f.write(b'\x00' * 100)  # Padding
        f.write(b'Hello, World!')  # Test content
        f.flush()
        path = f.name

    yield path

    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_buffer_files():
    """Create multiple temporary buffer files."""
    paths = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_shard{i}.dat') as f:
            f.write(f'SHARD{i}'.encode())
            f.write(b'\x00' * 50)
            paths.append(f.name)

    yield paths

    # Cleanup
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


class TestEngramManager:
    """Test EngramManager class."""

    def test_open_and_close(self, temp_buffer_file):
        """Test opening and closing buffer."""
        with EngramManager(temp_buffer_file) as manager:
            assert manager.size > 0
            assert manager._is_open

        # After exiting context, should be closed
        assert not manager._is_open

    def test_get_parser(self, temp_buffer_file):
        """Test getting a BinaryParser from manager."""
        with EngramManager(temp_buffer_file) as manager:
            parser = manager.get_parser()

            assert isinstance(parser, BinaryParser)
            # Read magic signature
            magic = parser.read_bytes(4)
            assert bytes(magic) == b'DRGN'
            del parser
            import gc
            gc.collect()

    def test_get_slice(self, temp_buffer_file):
        """Test getting a memoryview slice."""
        with EngramManager(temp_buffer_file) as manager:
            view = manager.get_slice(0, 4)

            assert isinstance(view, memoryview)
            assert bytes(view) == b'DRGN'
            del view

    def test_read_at(self, temp_buffer_file):
        """Test reading bytes at offset."""
        with EngramManager(temp_buffer_file) as manager:
            # Read the test content at known offset
            data = manager.read_at(104, 13)  # 4 + 100 = 104
            assert data == b'Hello, World!'

    def test_find_pattern(self, temp_buffer_file):
        """Test finding pattern in buffer."""
        with EngramManager(temp_buffer_file) as manager:
            pos = manager.find(b'Hello')
            assert pos == 104

            pos = manager.find(b'NotFound')
            assert pos == -1

    def test_buffer_not_open_error(self, temp_buffer_file):
        """Test error when accessing closed buffer."""
        manager = EngramManager(temp_buffer_file)

        with pytest.raises(BufferNotOpenError):
            _ = manager.size

        with pytest.raises(BufferNotOpenError):
            manager.get_parser()

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            with EngramManager('/nonexistent/path/to/file.dat'):
                pass

    def test_parser_with_offset(self, temp_buffer_file):
        """Test getting parser at specific offset."""
        with EngramManager(temp_buffer_file) as manager:
            # Skip past magic and padding to content
            parser = manager.get_parser(offset=104)
            content = parser.read_string(13)
            assert content == 'Hello, World!'
            del parser

    def test_resource_cleanup_on_error(self, temp_buffer_file):
        """Test that resources are cleaned up even on error."""
        class TestException(Exception):
            pass

        try:
            with EngramManager(temp_buffer_file) as manager:
                # Verify buffer is open
                assert manager._is_open
                raise TestException("Simulated error")
        except TestException:
            pass

        # Buffer should be closed after exception
        assert not manager._is_open
        assert manager._mmap is None

    def test_memoryview_zero_copy(self, temp_buffer_file):
        """Test that memoryview provides zero-copy access."""
        with EngramManager(temp_buffer_file) as manager:
            view1 = manager.view
            view2 = manager.get_slice(0, 4)

            # Both views should reference the same underlying buffer
            # Modifying through the mmap should be visible in views
            # (Can't test mutation in read-only mode, but we verify views work)
            assert bytes(view2) == b'DRGN'
            del view1, view2


class TestMultiBufferManager:
    """Test MultiBufferManager class."""

    def test_open_multiple_buffers(self, temp_buffer_files):
        """Test opening multiple buffers."""
        with MultiBufferManager(temp_buffer_files) as manager:
            assert len(manager) == 3

            for i in range(3):
                buf = manager[i]
                data = buf.read_at(0, 6)
                assert data == f'SHARD{i}'.encode()

    def test_get_parsers(self, temp_buffer_files):
        """Test getting parsers for all buffers."""
        with MultiBufferManager(temp_buffer_files) as manager:
            parsers = manager.get_parsers()

            assert len(parsers) == 3
            for i, parser in enumerate(parsers):
                shard_id = parser.read_string(6)
                assert shard_id == f'SHARD{i}'
            del parsers

    def test_resource_cleanup(self, temp_buffer_files):
        """Test that all buffers are cleaned up."""
        with MultiBufferManager(temp_buffer_files) as manager:
            buffers = [manager[i] for i in range(3)]
            for buf in buffers:
                assert buf._is_open

        # All should be closed
        for buf in buffers:
            assert not buf._is_open


class TestAsyncEngramManager:
    """Test AsyncEngramManager class."""

    @pytest.mark.asyncio
    async def test_async_open_close(self, temp_buffer_file):
        """Test async context manager."""
        async with AsyncEngramManager(temp_buffer_file) as manager:
            assert manager.size > 0

    @pytest.mark.asyncio
    async def test_async_read_at(self, temp_buffer_file):
        """Test async read operation."""
        async with AsyncEngramManager(temp_buffer_file) as manager:
            data = await manager.read_at(0, 4)
            assert data == b'DRGN'

    @pytest.mark.asyncio
    async def test_get_parser_sync(self, temp_buffer_file):
        """Test getting synchronous parser from async manager."""
        async with AsyncEngramManager(temp_buffer_file) as manager:
            parser = manager.get_parser()
            magic = parser.read_bytes(4)
            assert bytes(magic) == b'DRGN'
            del parser

    @pytest.mark.asyncio
    async def test_get_view(self, temp_buffer_file):
        """Test getting memoryview from async manager."""
        async with AsyncEngramManager(temp_buffer_file) as manager:
            view = manager.get_view()
            assert bytes(view[:4]) == b'DRGN'
            del view


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_buffer_file(self):
        """Test creating a new buffer file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'new_buffer.dat')
            create_buffer_file(path, 1024)

            assert os.path.exists(path)
            assert os.path.getsize(path) == 1024

            with open(path, 'rb') as f:
                content = f.read()
                assert content == b'\x00' * 1024

    def test_create_buffer_file_with_fill(self):
        """Test creating buffer with custom fill byte."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'filled_buffer.dat')
            create_buffer_file(path, 100, fill=b'\xFF')

            with open(path, 'rb') as f:
                content = f.read()
                assert content == b'\xFF' * 100

    def test_verify_buffer_integrity_exists(self, temp_buffer_file):
        """Test verifying existing buffer."""
        assert verify_buffer_integrity(temp_buffer_file) is True

    def test_verify_buffer_integrity_not_exists(self):
        """Test verifying non-existent buffer."""
        assert verify_buffer_integrity('/nonexistent/file.dat') is False

    def test_verify_buffer_integrity_with_magic(self, temp_buffer_file):
        """Test verifying buffer with expected magic bytes."""
        assert verify_buffer_integrity(temp_buffer_file, expected_magic=b'DRGN') is True
        assert verify_buffer_integrity(temp_buffer_file, expected_magic=b'XXXX') is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_file(self):
        """Test handling of empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name

        try:
            with pytest.raises(EngramManagerError):
                with EngramManager(path):
                    pass
        finally:
            os.unlink(path)

    def test_very_small_file(self):
        """Test handling of very small file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'X')  # Just one byte
            path = f.name

        try:
            with EngramManager(path) as manager:
                assert manager.size == 1
                data = manager.read_at(0, 1)
                assert data == b'X'
        finally:
            os.unlink(path)

    def test_read_beyond_buffer(self, temp_buffer_file):
        """Test reading beyond buffer bounds."""
        with EngramManager(temp_buffer_file) as manager:
            size = manager.size
            # Reading beyond should return truncated data (Python slice behavior)
            data = manager.get_slice(size - 5, size + 100)
            assert len(data) == 5

    def test_concurrent_managers_same_file(self, temp_buffer_file):
        """Test multiple managers for same file."""
        with EngramManager(temp_buffer_file) as manager1:
            with EngramManager(temp_buffer_file) as manager2:
                # Both should be able to read
                data1 = manager1.read_at(0, 4)
                data2 = manager2.read_at(0, 4)
                assert data1 == data2 == b'DRGN'


class TestParserIntegration:
    """Test integration with BinaryParser."""

    def test_parser_reads_all_data(self, temp_buffer_file):
        """Test parser can read all buffer data."""
        with EngramManager(temp_buffer_file) as manager:
            parser = manager.get_parser()

            # Read until exhausted
            data_chunks = []
            while not parser.is_exhausted:
                chunk_size = min(parser.remaining, 16)
                data_chunks.append(bytes(parser.read_bytes(chunk_size)))

            total_read = sum(len(c) for c in data_chunks)
            assert total_read == manager.size
            del parser

    def test_parser_position_tracking(self, temp_buffer_file):
        """Test parser position tracking."""
        with EngramManager(temp_buffer_file) as manager:
            parser = manager.get_parser()

            assert parser.position == 0
            parser.read_bytes(4)
            assert parser.position == 4

            parser.skip(10)
            assert parser.position == 14
            del parser


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
