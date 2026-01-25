"""
SENSE-v2 Memory Hierarchy Module

Implements host RAM prefetching and hierarchical memory management
for efficient GPU <-> CPU memory transfers.

Key components:
- EmbeddingPrefetcher: Async prefetching from host RAM to GPU
- MemoryHierarchy: Tiered storage management (GPU -> Host -> Disk)
- PinnedMemoryPool: Reusable pinned memory buffers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import asyncio
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor

from sense.engram.storage import MMapEmbeddingStorage


@dataclass
class PrefetchRequest:
    """Request for async prefetch operation."""
    indices: torch.Tensor
    priority: int = 0
    callback: Optional[callable] = None


@dataclass
class PrefetchResult:
    """Result of a prefetch operation."""
    indices: torch.Tensor
    embeddings: torch.Tensor
    hit_rate: float
    transfer_time_ms: float


class PinnedMemoryPool:
    """
    Pool of pinned (page-locked) memory buffers for fast CPU-GPU transfers.

    Pinned memory enables faster DMA transfers and can be accessed
    asynchronously by the GPU.
    """

    def __init__(
        self,
        num_buffers: int = 4,
        buffer_size: int = 1024 * 1024,  # 1M elements
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize pinned memory pool.

        Args:
            num_buffers: Number of pre-allocated buffers
            buffer_size: Size of each buffer in elements
            dtype: Data type for buffers
        """
        self.num_buffers = num_buffers
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.logger = logging.getLogger(self.__class__.__name__)

        # Pre-allocate pinned buffers
        self._buffers: List[torch.Tensor] = []
        self._available: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Pre-allocate pinned memory buffers."""
        for _ in range(self.num_buffers):
            try:
                # Allocate pinned memory (requires CUDA)
                if torch.cuda.is_available():
                    buffer = torch.empty(
                        self.buffer_size,
                        dtype=self.dtype,
                        pin_memory=True,
                    )
                else:
                    # Fallback to regular memory on CPU-only systems
                    buffer = torch.empty(self.buffer_size, dtype=self.dtype)

                self._buffers.append(buffer)
                self._available.put(len(self._buffers) - 1)
            except Exception as e:
                self.logger.warning(f"Failed to allocate pinned buffer: {e}")

    def acquire(self, timeout: float = 1.0) -> Optional[Tuple[int, torch.Tensor]]:
        """
        Acquire a buffer from the pool.

        Args:
            timeout: Maximum time to wait for a buffer

        Returns:
            Tuple of (buffer_id, buffer) or None if timeout
        """
        try:
            buffer_id = self._available.get(timeout=timeout)
            return buffer_id, self._buffers[buffer_id]
        except queue.Empty:
            return None

    def release(self, buffer_id: int) -> None:
        """Release a buffer back to the pool."""
        self._available.put(buffer_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_buffers": self.num_buffers,
            "available_buffers": self._available.qsize(),
            "buffer_size": self.buffer_size,
            "dtype": str(self.dtype),
            "is_pinned": torch.cuda.is_available(),
        }


class EmbeddingPrefetcher:
    """
    Async prefetcher for host RAM -> GPU embedding transfers.

    Implements double-buffering and asynchronous transfers to overlap
    computation with data movement.
    """

    def __init__(
        self,
        storage: MMapEmbeddingStorage,
        batch_size: int = 1024,
        num_workers: int = 2,
        use_pinned_memory: bool = True,
    ):
        """
        Initialize embedding prefetcher.

        Args:
            storage: Memory-mapped embedding storage
            batch_size: Default batch size for prefetch operations
            num_workers: Number of background workers
            use_pinned_memory: Whether to use pinned memory for transfers
        """
        self.storage = storage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_pinned_memory = use_pinned_memory and torch.cuda.is_available()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        # Pinned memory pool for fast transfers
        if self.use_pinned_memory:
            buffer_size = batch_size * storage.embedding_dim
            self._memory_pool = PinnedMemoryPool(
                num_buffers=4,
                buffer_size=buffer_size,
            )
        else:
            self._memory_pool = None

        # Cache for recently prefetched embeddings
        self._cache: OrderedDict = OrderedDict()
        self._cache_max_size = 10000
        self._cache_lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_transfer_time_ms": 0.0,
        }

    def _fetch_from_storage(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fetch embeddings from storage (blocking).

        Args:
            indices: Embedding indices to fetch

        Returns:
            Fetched embeddings tensor
        """
        # Use mmap storage's forward method
        return self.storage(indices)

    def _transfer_to_gpu(
        self,
        embeddings: torch.Tensor,
        device: torch.device,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """
        Transfer embeddings to GPU.

        Args:
            embeddings: CPU embeddings tensor
            device: Target GPU device
            non_blocking: Use async transfer

        Returns:
            GPU embeddings tensor
        """
        if self.use_pinned_memory and embeddings.is_pinned():
            return embeddings.to(device, non_blocking=non_blocking)
        return embeddings.to(device)

    def prefetch(
        self,
        indices: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Prefetch embeddings with caching.

        Args:
            indices: Embedding indices [batch_size] or [batch_size, seq_len, ...]
            device: Target device (GPU or CPU)

        Returns:
            Fetched embeddings tensor
        """
        import time
        start_time = time.time()

        self._stats["total_requests"] += 1

        # Flatten indices for lookup
        original_shape = indices.shape
        flat_indices = indices.view(-1)

        # Check cache
        cache_hits = []
        cache_misses = []
        cache_miss_positions = []

        with self._cache_lock:
            for i, idx in enumerate(flat_indices.tolist()):
                if idx in self._cache:
                    cache_hits.append(self._cache[idx])
                    # Move to end (LRU)
                    self._cache.move_to_end(idx)
                else:
                    cache_misses.append(idx)
                    cache_miss_positions.append(i)

        self._stats["cache_hits"] += len(cache_hits)
        self._stats["cache_misses"] += len(cache_misses)

        # Fetch missing embeddings
        if cache_misses:
            miss_indices = torch.tensor(cache_misses, dtype=torch.long)
            fetched = self._fetch_from_storage(miss_indices)

            # Update cache
            with self._cache_lock:
                for i, idx in enumerate(cache_misses):
                    self._cache[idx] = fetched[i].clone()
                    # Evict if cache full
                    while len(self._cache) > self._cache_max_size:
                        self._cache.popitem(last=False)

        # Assemble full result
        embedding_dim = self.storage.embedding_dim
        result = torch.zeros(len(flat_indices), embedding_dim)

        # Fill from cache hits
        hit_idx = 0
        miss_idx = 0
        for i in range(len(flat_indices)):
            if i in cache_miss_positions:
                # This was a cache miss, get from fetched
                pos_in_misses = cache_miss_positions.index(i)
                result[i] = fetched[pos_in_misses]
                miss_idx += 1
            else:
                result[i] = cache_hits[hit_idx]
                hit_idx += 1

        # Reshape to original
        result = result.view(*original_shape, embedding_dim)

        # Transfer to device if specified
        if device is not None and device.type != "cpu":
            result = self._transfer_to_gpu(result, device)

        elapsed_ms = (time.time() - start_time) * 1000
        self._stats["total_transfer_time_ms"] += elapsed_ms

        return result

    async def prefetch_async(
        self,
        indices: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Async prefetch embeddings.

        Args:
            indices: Embedding indices
            device: Target device

        Returns:
            Fetched embeddings tensor
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.prefetch(indices, device)
        )

    def prefetch_batch(
        self,
        batch_indices: List[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        """
        Prefetch multiple batches in parallel.

        Args:
            batch_indices: List of index tensors
            device: Target device

        Returns:
            List of fetched embedding tensors
        """
        futures = [
            self._executor.submit(self.prefetch, indices, device)
            for indices in batch_indices
        ]

        return [f.result() for f in futures]

    def get_stats(self) -> Dict[str, Any]:
        """Get prefetcher statistics."""
        total = self._stats["cache_hits"] + self._stats["cache_misses"]
        hit_rate = self._stats["cache_hits"] / max(total, 1)

        return {
            **self._stats,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "cache_max_size": self._cache_max_size,
            "avg_transfer_time_ms": (
                self._stats["total_transfer_time_ms"] /
                max(self._stats["total_requests"], 1)
            ),
        }

    def clear_cache(self) -> None:
        """Clear the prefetch cache."""
        with self._cache_lock:
            self._cache.clear()

    def shutdown(self) -> None:
        """Shutdown the prefetcher."""
        self._executor.shutdown(wait=True)


class MemoryHierarchy:
    """
    Tiered memory hierarchy manager.

    Manages data placement across:
    - L1: GPU VRAM (fastest, limited)
    - L2: Host RAM with pinned memory (fast transfers)
    - L3: Disk via memory mapping (large capacity)
    """

    def __init__(
        self,
        gpu_cache_size_mb: int = 512,
        host_cache_size_mb: int = 2048,
        use_pinned_memory: bool = True,
    ):
        """
        Initialize memory hierarchy.

        Args:
            gpu_cache_size_mb: GPU cache size in MB
            host_cache_size_mb: Host cache size in MB
            use_pinned_memory: Use pinned memory for host cache
        """
        self.gpu_cache_size_mb = gpu_cache_size_mb
        self.host_cache_size_mb = host_cache_size_mb
        self.use_pinned_memory = use_pinned_memory and torch.cuda.is_available()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Caches
        self._gpu_cache: OrderedDict = OrderedDict()
        self._host_cache: OrderedDict = OrderedDict()
        self._gpu_cache_bytes = 0
        self._host_cache_bytes = 0

        # Locks for thread safety
        self._gpu_lock = threading.Lock()
        self._host_lock = threading.Lock()

        # Statistics
        self._stats = {
            "gpu_hits": 0,
            "host_hits": 0,
            "disk_reads": 0,
            "promotions": 0,  # L3 -> L2 -> L1
            "evictions": 0,
        }

    def get(
        self,
        key: str,
        fetch_fn: callable,
        target_device: torch.device,
    ) -> torch.Tensor:
        """
        Get data from hierarchy, fetching if necessary.

        Args:
            key: Cache key
            fetch_fn: Function to fetch data from disk if not cached
            target_device: Target device for the data

        Returns:
            Data tensor on target device
        """
        # Try GPU cache first
        if target_device.type == "cuda":
            with self._gpu_lock:
                if key in self._gpu_cache:
                    self._stats["gpu_hits"] += 1
                    self._gpu_cache.move_to_end(key)
                    return self._gpu_cache[key]

        # Try host cache
        with self._host_lock:
            if key in self._host_cache:
                self._stats["host_hits"] += 1
                self._host_cache.move_to_end(key)
                data = self._host_cache[key]

                if target_device.type == "cuda":
                    # Promote to GPU cache
                    gpu_data = data.to(target_device)
                    self._promote_to_gpu(key, gpu_data)
                    return gpu_data

                return data

        # Fetch from disk
        self._stats["disk_reads"] += 1
        data = fetch_fn()

        # Add to host cache
        self._add_to_host_cache(key, data)

        # Promote to GPU if needed
        if target_device.type == "cuda":
            gpu_data = data.to(target_device)
            self._promote_to_gpu(key, gpu_data)
            return gpu_data

        return data

    def _add_to_host_cache(self, key: str, data: torch.Tensor) -> None:
        """Add data to host cache with eviction."""
        data_bytes = data.numel() * data.element_size()
        max_bytes = self.host_cache_size_mb * 1024 * 1024

        with self._host_lock:
            # Evict until space available
            while self._host_cache_bytes + data_bytes > max_bytes and self._host_cache:
                evict_key, evict_data = self._host_cache.popitem(last=False)
                self._host_cache_bytes -= evict_data.numel() * evict_data.element_size()
                self._stats["evictions"] += 1

            # Add new data
            if self.use_pinned_memory:
                pinned_data = data.pin_memory()
                self._host_cache[key] = pinned_data
            else:
                self._host_cache[key] = data.clone()

            self._host_cache_bytes += data_bytes

    def _promote_to_gpu(self, key: str, data: torch.Tensor) -> None:
        """Promote data to GPU cache."""
        data_bytes = data.numel() * data.element_size()
        max_bytes = self.gpu_cache_size_mb * 1024 * 1024

        with self._gpu_lock:
            # Evict until space available
            while self._gpu_cache_bytes + data_bytes > max_bytes and self._gpu_cache:
                evict_key, evict_data = self._gpu_cache.popitem(last=False)
                self._gpu_cache_bytes -= evict_data.numel() * evict_data.element_size()
                self._stats["evictions"] += 1

            self._gpu_cache[key] = data
            self._gpu_cache_bytes += data_bytes
            self._stats["promotions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get hierarchy statistics."""
        total_accesses = (
            self._stats["gpu_hits"] +
            self._stats["host_hits"] +
            self._stats["disk_reads"]
        )

        return {
            **self._stats,
            "total_accesses": total_accesses,
            "gpu_hit_rate": self._stats["gpu_hits"] / max(total_accesses, 1),
            "host_hit_rate": self._stats["host_hits"] / max(total_accesses, 1),
            "gpu_cache_mb": self._gpu_cache_bytes / 1024 / 1024,
            "host_cache_mb": self._host_cache_bytes / 1024 / 1024,
        }

    def clear(self) -> None:
        """Clear all caches."""
        with self._gpu_lock:
            self._gpu_cache.clear()
            self._gpu_cache_bytes = 0

        with self._host_lock:
            self._host_cache.clear()
            self._host_cache_bytes = 0
