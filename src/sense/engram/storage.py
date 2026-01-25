import os
import numpy as np
import torch.nn as nn
import torch
from typing import Optional
import threading


class MMapEmbeddingStorage(nn.Module):
    """
    Memory-mapped embedding storage for large embedding tables.

    Uses numpy memmap for efficient disk-backed storage with support for:
    - Pinned memory transfers for faster GPU loading
    - Async non-blocking GPU copies
    - Thread-safe access
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, path: str):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.path = path
        self._lock = threading.Lock()
        self._pinned_buffer: Optional[torch.Tensor] = None

        if not os.path.exists(path):
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            # Create a new memory-mapped file and initialize with random normal noise
            self.mmap = np.memmap(path, dtype=np.float32, mode='w+', shape=(num_embeddings, embedding_dim))
            self.mmap[:] = np.random.normal(size=(num_embeddings, embedding_dim)).astype(np.float32)
            self.mmap.flush()
        else:
            # Load existing memory-mapped file
            self.mmap = np.memmap(path, dtype=np.float32, mode='r+', shape=(num_embeddings, embedding_dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve embeddings by indices.

        Args:
            indices: Embedding indices (any shape)

        Returns:
            Retrieved embeddings with shape [...indices.shape, embedding_dim]
        """
        # Ensure indices are on CPU for NumPy indexing
        indices_np = indices.cpu().numpy()

        with self._lock:
            # Slice the memmap using CPU indices
            retrieved_embeddings_np = self.mmap[indices_np]

        # Convert back to torch.Tensor and move to original device
        retrieved_embeddings = torch.from_numpy(retrieved_embeddings_np.copy()).to(indices.device)

        return retrieved_embeddings

    def pin_memory(self, buffer_size: Optional[int] = None) -> None:
        """
        Allocate pinned memory buffer for faster GPU transfers.

        Pinned (page-locked) memory enables faster DMA transfers to GPU.

        Args:
            buffer_size: Number of embeddings to buffer. Defaults to 1024.
        """
        if not torch.cuda.is_available():
            return

        buffer_size = buffer_size or 1024
        self._pinned_buffer = torch.empty(
            buffer_size, self.embedding_dim,
            dtype=torch.float32,
            pin_memory=True,
        )

    def to_gpu_async(
        self,
        indices: torch.Tensor,
        device: Optional[torch.device] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """
        Non-blocking GPU copy with optional CUDA stream.

        Args:
            indices: Embedding indices
            device: Target CUDA device
            stream: CUDA stream for async operation

        Returns:
            GPU tensor (may still be transferring if stream provided)
        """
        if not torch.cuda.is_available():
            return self.forward(indices)

        device = device or torch.device("cuda")
        indices_np = indices.cpu().numpy()
        num_indices = indices_np.size

        with self._lock:
            retrieved_np = self.mmap[indices_np]

        # Use pinned buffer if available and indices fit
        if self._pinned_buffer is not None and num_indices <= self._pinned_buffer.shape[0]:
            # Copy to pinned buffer
            flat_shape = (num_indices, self.embedding_dim)
            self._pinned_buffer[:num_indices].copy_(
                torch.from_numpy(retrieved_np.reshape(flat_shape))
            )

            # Async transfer to GPU
            if stream is not None:
                with torch.cuda.stream(stream):
                    gpu_tensor = self._pinned_buffer[:num_indices].to(
                        device, non_blocking=True
                    )
            else:
                gpu_tensor = self._pinned_buffer[:num_indices].to(
                    device, non_blocking=True
                )

            # Reshape to match original indices shape
            return gpu_tensor.view(*indices.shape, self.embedding_dim)

        # Fallback to regular transfer
        retrieved = torch.from_numpy(retrieved_np.copy())
        return retrieved.to(device, non_blocking=True)

    def update_embeddings(
        self,
        indices: torch.Tensor,
        embeddings: torch.Tensor,
        flush: bool = True,
    ) -> None:
        """
        Update embeddings at specified indices.

        Args:
            indices: Indices to update
            embeddings: New embedding values
            flush: Whether to flush to disk immediately
        """
        indices_np = indices.cpu().numpy()
        embeddings_np = embeddings.detach().cpu().numpy()

        with self._lock:
            self.mmap[indices_np] = embeddings_np
            if flush:
                self.mmap.flush()

    def get_stats(self) -> dict:
        """Get storage statistics."""
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "storage_mb": (self.num_embeddings * self.embedding_dim * 4) / 1024 / 1024,
            "path": self.path,
            "has_pinned_buffer": self._pinned_buffer is not None,
            "pinned_buffer_size": (
                self._pinned_buffer.shape[0] if self._pinned_buffer is not None else 0
            ),
        }

    def close(self) -> None:
        """Close the memory-mapped file."""
        with self._lock:
            if hasattr(self, 'mmap') and self.mmap is not None:
                del self.mmap
                self.mmap = None
