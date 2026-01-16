import os
import numpy as np
import torch.nn as nn
import torch

class MMapEmbeddingStorage(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, path: str):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.path = path
        
        if not os.path.exists(path):
            # Create a new memory-mapped file and initialize with random normal noise
            self.mmap = np.memmap(path, dtype=np.float32, mode='w+', shape=(num_embeddings, embedding_dim))
            self.mmap[:] = np.random.normal(size=(num_embeddings, embedding_dim)).astype(np.float32)
            self.mmap.flush()
        else:
            # Load existing memory-mapped file
            self.mmap = np.memmap(path, dtype=np.float32, mode='r+', shape=(num_embeddings, embedding_dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Ensure indices are on CPU for NumPy indexing
        indices_np = indices.cpu().numpy()
        
        # Slice the memmap using CPU indices
        retrieved_embeddings_np = self.mmap[indices_np]
        
        # Convert back to torch.Tensor and move to original device
        retrieved_embeddings = torch.from_numpy(retrieved_embeddings_np).to(indices.device)
        
        return retrieved_embeddings
