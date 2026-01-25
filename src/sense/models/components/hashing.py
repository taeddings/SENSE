"""
SENSE-v2 N-gram Hashing Module

Implements multi-head XOR hashing for efficient n-gram to embedding lookup.
This replaces the placeholder hash in engram/model.py with a proper
deterministic hashing scheme.

The key insight is using multiple hash heads with different prime multipliers
to reduce collision rates while maintaining O(1) lookup.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import math


def _generate_primes(n: int, seed: int = 42) -> List[int]:
    """
    Generate n prime numbers for use as hash multipliers.

    Uses a deterministic seeded approach to ensure reproducibility.

    Args:
        n: Number of primes to generate
        seed: Random seed for prime selection

    Returns:
        List of n prime numbers
    """
    # Pre-computed list of primes for efficiency
    # Using primes spread across different ranges for better distribution
    prime_pool = [
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
        353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    ]

    # Use seed to select starting position
    torch.manual_seed(seed)
    start_idx = torch.randint(0, max(1, len(prime_pool) - n), (1,)).item()

    return prime_pool[start_idx:start_idx + n]


class MultiHeadHash(nn.Module):
    """
    Deterministic multi-head XOR hash for N-grams.

    Each head uses a different prime multiplier to produce independent
    hash values, reducing collision probability. The XOR folding ensures
    all tokens in an n-gram contribute to the final hash.

    Hash formula for head h:
        hash_h(ngram) = XOR(token_i * prime_h^i) mod table_size

    Attributes:
        num_heads: Number of independent hash functions
        table_size: Size of the embedding table
        primes: Prime multipliers for each head
    """

    def __init__(
        self,
        num_heads: int,
        table_size: int,
        seed: int = 42,
    ):
        """
        Initialize multi-head hash.

        Args:
            num_heads: Number of hash heads
            table_size: Size of embedding lookup table
            seed: Random seed for prime generation
        """
        super().__init__()
        self.num_heads = num_heads
        self.table_size = table_size
        self.seed = seed

        # Generate prime multipliers
        primes = _generate_primes(num_heads, seed)
        self.register_buffer(
            "primes",
            torch.tensor(primes, dtype=torch.long)
        )

        # Pre-compute prime powers for efficiency
        # powers[h, i] = prime_h^i for i in 0..max_ngram_order
        max_power = 8  # Support up to 8-gram
        powers = torch.zeros(num_heads, max_power, dtype=torch.long)
        for h, p in enumerate(primes):
            for i in range(max_power):
                powers[h, i] = p ** i
        self.register_buffer("prime_powers", powers)

    def forward(
        self,
        ngram_ids: torch.Tensor,
        ngram_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-head hash indices for n-grams.

        Args:
            ngram_ids: Token IDs forming n-grams
                Shape: [batch_size, seq_len, ngram_order] or [batch_size, seq_len]
            ngram_mask: Optional mask for valid n-gram positions
                Shape: same as ngram_ids

        Returns:
            Hash indices [batch_size, seq_len, num_heads]
        """
        # Handle 2D input (single token per position)
        if ngram_ids.dim() == 2:
            ngram_ids = ngram_ids.unsqueeze(-1)

        batch_size, seq_len, ngram_order = ngram_ids.shape
        device = ngram_ids.device

        # Ensure ngram_order doesn't exceed prime powers
        ngram_order = min(ngram_order, self.prime_powers.shape[1])

        # Compute hash for each head
        # hash = XOR(token_i * prime^i) for i in ngram
        hashes = torch.zeros(
            batch_size, seq_len, self.num_heads,
            dtype=torch.long, device=device
        )

        for i in range(ngram_order):
            token_ids = ngram_ids[:, :, i]  # [batch, seq]

            for h in range(self.num_heads):
                # Multiply by prime power and XOR
                weighted = token_ids * self.prime_powers[h, i]
                hashes[:, :, h] ^= weighted

        # Apply table size modulo
        hashes = hashes % self.table_size

        # Apply mask if provided
        if ngram_mask is not None:
            # Mask should be broadcastable or same shape
            if ngram_mask.dim() == 3:
                ngram_mask = ngram_mask.any(dim=-1)  # [batch, seq]
            # Set masked positions to 0 (or a special index)
            hashes = hashes * ngram_mask.unsqueeze(-1).long()

        return hashes

    def hash_single(self, tokens: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """
        Compute hash for a single head (utility function).

        Args:
            tokens: Token IDs [batch_size, ngram_order]
            head_idx: Which hash head to use

        Returns:
            Hash indices [batch_size]
        """
        ngram_order = tokens.shape[-1]
        result = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)

        for i in range(ngram_order):
            weighted = tokens[:, i] * self.prime_powers[head_idx, i]
            result ^= weighted

        return result % self.table_size


class NGramExtractor(nn.Module):
    """
    Extract sliding window n-grams from token sequences.

    Converts a sequence of token IDs into overlapping n-grams
    for hash-based lookup.
    """

    def __init__(
        self,
        ngram_orders: List[int],
        pad_id: int = 0,
    ):
        """
        Initialize n-gram extractor.

        Args:
            ngram_orders: List of n-gram sizes to extract (e.g., [2, 3])
            pad_id: Padding token ID for sequence boundaries
        """
        super().__init__()
        self.ngram_orders = sorted(ngram_orders)
        self.max_order = max(ngram_orders)
        self.pad_id = pad_id

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract n-grams from input sequence.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Tuple of:
            - ngrams: [batch_size, seq_len, max_order] padded n-gram tokens
            - mask: [batch_size, seq_len] validity mask
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Pad sequence for n-gram extraction
        # Pad at the beginning to handle start of sequence
        pad_size = self.max_order - 1
        padded = torch.full(
            (batch_size, seq_len + pad_size),
            self.pad_id,
            dtype=input_ids.dtype,
            device=device
        )
        padded[:, pad_size:] = input_ids

        # Extract sliding windows
        ngrams = torch.zeros(
            batch_size, seq_len, self.max_order,
            dtype=input_ids.dtype, device=device
        )

        for i in range(self.max_order):
            start = pad_size - i
            end = start + seq_len
            ngrams[:, :, self.max_order - 1 - i] = padded[:, start:end]

        # Create mask (positions with all valid tokens)
        # A position is valid if it has enough context
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        mask[:, :pad_size] = False  # First (max_order-1) positions lack full context

        return ngrams, mask

    def extract_specific_order(
        self,
        input_ids: torch.Tensor,
        order: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract n-grams of a specific order.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            order: N-gram order to extract

        Returns:
            Tuple of (ngrams, mask) for the specific order
        """
        if order not in self.ngram_orders:
            raise ValueError(f"Order {order} not in configured orders {self.ngram_orders}")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        pad_size = order - 1
        padded = torch.full(
            (batch_size, seq_len + pad_size),
            self.pad_id,
            dtype=input_ids.dtype,
            device=device
        )
        padded[:, pad_size:] = input_ids

        ngrams = torch.zeros(
            batch_size, seq_len, order,
            dtype=input_ids.dtype, device=device
        )

        for i in range(order):
            start = pad_size - i
            end = start + seq_len
            ngrams[:, :, order - 1 - i] = padded[:, start:end]

        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        mask[:, :pad_size] = False

        return ngrams, mask


def compute_ngram_hashes(
    input_ids: torch.Tensor,
    hasher: MultiHeadHash,
    extractor: NGramExtractor,
    combine_orders: bool = True,
) -> torch.Tensor:
    """
    Compute n-gram hashes for a sequence.

    Convenience function that combines n-gram extraction and hashing.

    Args:
        input_ids: Token IDs [batch_size, seq_len]
        hasher: MultiHeadHash instance
        extractor: NGramExtractor instance
        combine_orders: If True, combine hashes from different n-gram orders

    Returns:
        Hash indices [batch_size, seq_len, num_heads]
    """
    ngrams, mask = extractor(input_ids)
    hashes = hasher(ngrams, mask)

    if combine_orders and len(extractor.ngram_orders) > 1:
        # Combine hashes from different orders using XOR
        combined = hashes.clone()
        for order in extractor.ngram_orders[:-1]:
            order_ngrams, order_mask = extractor.extract_specific_order(input_ids, order)
            order_hashes = hasher(order_ngrams, order_mask)
            combined ^= order_hashes
        # Re-apply modulo after combination
        combined = combined % hasher.table_size
        return combined

    return hashes


class LearnableHashEmbedding(nn.Module):
    """
    Learnable hash-based embedding with collision handling.

    Combines hash-based lookup with a small learnable correction
    to handle hash collisions gracefully.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_heads: int = 8,
        ngram_orders: List[int] = None,
        seed: int = 42,
    ):
        """
        Initialize learnable hash embedding.

        Args:
            num_embeddings: Size of embedding table
            embedding_dim: Dimension of embeddings
            num_heads: Number of hash heads
            ngram_orders: N-gram orders to use
            seed: Random seed
        """
        super().__init__()

        self.ngram_orders = ngram_orders or [2, 3]
        self.num_heads = num_heads

        # Core components
        self.hasher = MultiHeadHash(num_heads, num_embeddings, seed)
        self.extractor = NGramExtractor(self.ngram_orders)

        # Base embeddings (shared across heads)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        # Head-specific projection to combine multi-head lookups
        self.head_projection = nn.Linear(embedding_dim * num_heads, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute hash-based embeddings.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Get hash indices for all heads
        hashes = compute_ngram_hashes(input_ids, self.hasher, self.extractor)
        # hashes: [batch_size, seq_len, num_heads]

        # Look up embeddings for each head
        # [batch_size, seq_len, num_heads, embedding_dim]
        head_embeddings = self.embeddings(hashes)

        # Flatten heads and project
        # [batch_size, seq_len, num_heads * embedding_dim]
        flat = head_embeddings.view(batch_size, seq_len, -1)

        # Project to final dimension
        # [batch_size, seq_len, embedding_dim]
        output = self.head_projection(flat)

        return output
