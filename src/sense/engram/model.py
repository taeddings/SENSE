import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from sense.core.config import EngramConfig
from sense.engram.tokenizer import EngramTokenizer
from sense.engram.storage import MMapEmbeddingStorage
from sense.models.components.hashing import MultiHeadHash, NGramExtractor

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class EngramFusionLayer(nn.Module):
    """
    Engram Conditional Memory Fusion Layer.

    Implements the Engram architecture for offloading computation to memory:
    1. Compress input tokens via shadow map
    2. Extract N-grams and compute multi-head hashes
    3. Retrieve embeddings from memory-mapped storage
    4. Fuse with backbone hidden states via gated attention

    This provides significant memory savings by avoiding repeated computation
    for frequently occurring patterns.
    """

    def __init__(self, config: EngramConfig, backbone_hidden_size: int, layer_id: int):
        super().__init__()
        self.config = config
        self.backbone_hidden_size = backbone_hidden_size
        self.layer_id = layer_id

        # Components
        self.tokenizer = EngramTokenizer(config.shadow_map_path)
        self.storage = MMapEmbeddingStorage(config.table_size, config.engram_dim, config.storage_path)

        # N-gram hashing components (replaces placeholder)
        self.hasher = MultiHeadHash(
            num_heads=config.num_heads,
            table_size=config.table_size,
            seed=42 + layer_id,  # Different seed per layer for diversity
        )
        self.ngram_extractor = NGramExtractor(
            ngram_orders=config.ngram_orders,
            pad_id=0,
        )

        # Fusion Gates
        self.W_k = nn.Linear(config.engram_dim, backbone_hidden_size)
        self.W_v = nn.Linear(config.engram_dim, backbone_hidden_size)
        self.norm = RMSNorm(backbone_hidden_size)
        self.sigmoid = nn.Sigmoid()

        # Refinement
        self.conv1d = nn.Conv1d(backbone_hidden_size, backbone_hidden_size, kernel_size=3, padding=1)

    def _compute_ngram_hashes(self, compressed_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head hashes from compressed token IDs.

        Uses proper N-gram extraction with sliding windows and multi-head
        XOR hashing for collision-resistant lookups.

        Args:
            compressed_ids: Compressed token IDs [batch_size, seq_len]

        Returns:
            Hash indices [batch_size, seq_len, num_heads]
        """
        # Extract N-grams with sliding window
        ngrams, mask = self.ngram_extractor(compressed_ids)
        # ngrams: [batch_size, seq_len, max_order]

        # Compute multi-head hashes
        hashed_indices = self.hasher(ngrams, mask)
        # hashed_indices: [batch_size, seq_len, num_heads]

        return hashed_indices

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # 1. Compress Input -> Hash using proper N-gram hashing.
        compressed_ids = self.tokenizer(input_ids)
        hashed_indices = self._compute_ngram_hashes(compressed_ids)

        # 2. Retrieve Memory -> Project to Hidden Size.
        retrieved_memory = self.storage(hashed_indices) # [Batch, Seq, Heads, Engram_Dim]
        
        # Reshape for linear layers: [Batch * Seq * Heads, Engram_Dim]
        batch_size, seq_len, num_heads, engram_dim = retrieved_memory.shape
        retrieved_memory_flat = retrieved_memory.view(batch_size * seq_len * num_heads, engram_dim)

        K_flat = self.W_k(retrieved_memory_flat)
        V_flat = self.W_v(retrieved_memory_flat)

        K = K_flat.view(batch_size, seq_len, num_heads, self.backbone_hidden_size)
        V = V_flat.view(batch_size, seq_len, num_heads, self.backbone_hidden_size)

        # Aggregate K and V across heads (simple mean for now)
        K_agg = K.mean(dim=2) # [Batch, Seq, Hidden_Size]
        V_agg = V.mean(dim=2) # [Batch, Seq, Hidden_Size]

        # 3. Compute Gate alpha = sigmoid(Q . K).
        Q = self.norm(hidden_states) # [Batch, Seq, Hidden_Size]
        alpha = self.sigmoid(torch.sum(Q * K_agg, dim=-1, keepdim=True)) # [Batch, Seq, 1]

        # 4. Output = Hidden + (alpha * V) + Conv(alpha * V).
        gated_V = alpha * V_agg
        
        # Conv1d expects [Batch, Channels, Length]
        gated_V_permuted = gated_V.permute(0, 2, 1) # [Batch, Hidden_Size, Seq]
        conv_output = self.conv1d(gated_V_permuted).permute(0, 2, 1) # [Batch, Seq, Hidden_Size]

        output = hidden_states + gated_V + conv_output
        return output
