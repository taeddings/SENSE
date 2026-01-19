"""
SENSE Engram Module

Provides conditional memory architecture with efficient buffer management.

Components:
- EngramManager: Resource-safe memory-mapped buffer access
- AsyncEngramManager: Async wrapper for non-blocking operations
- EngramTokenizer: Token-level engram processing
- EngramStorage: Persistent engram table storage
- EngramModel: Neural engram lookup model
"""

from .manager import (
    EngramManager,
    AsyncEngramManager,
    MultiBufferManager,
    EngramManagerError,
    BufferNotOpenError,
    BufferAccessError,
    create_buffer_file,
    verify_buffer_integrity,
)

from .tokenizer import EngramTokenizer
from .storage import MMapEmbeddingStorage
from .model import EngramFusionLayer

__all__ = [
    # Manager
    "EngramManager",
    "AsyncEngramManager",
    "MultiBufferManager",
    "EngramManagerError",
    "BufferNotOpenError",
    "BufferAccessError",
    "create_buffer_file",
    "verify_buffer_integrity",

    # Other components
    "EngramTokenizer",
    "MMapEmbeddingStorage",
    "EngramFusionLayer",
]
