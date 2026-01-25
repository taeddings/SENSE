import pytest
import os
import numpy as np
import torch
from sense.engram.storage import MMapEmbeddingStorage # Uncommented this line

# Removed dummy MMapEmbeddingStorage class definition

# Test MMapEmbeddingStorage
def test_mmap_embedding_storage_initialization_new_file(tmp_path):
    path = tmp_path / "test_embeddings.dat"
    num_embeddings = 100
    embedding_dim = 128

    storage = MMapEmbeddingStorage(num_embeddings, embedding_dim, str(path))

    assert os.path.exists(path)
    assert isinstance(storage.mmap, np.memmap)
    assert storage.mmap.shape == (num_embeddings, embedding_dim)
    assert storage.mmap.dtype == np.float32
    # Check if initialized with random noise (not all zeros)
    assert not np.all(storage.mmap == 0)

def test_mmap_embedding_storage_initialization_existing_file(tmp_path):
    path = tmp_path / "existing_embeddings.dat"
    num_embeddings = 50
    embedding_dim = 64

    # Create a dummy existing file
    dummy_data = np.random.rand(num_embeddings, embedding_dim).astype(np.float32)
    dummy_mmap = np.memmap(path, dtype=np.float32, mode='w+', shape=(num_embeddings, embedding_dim))
    dummy_mmap[:] = dummy_data[:]
    dummy_mmap.flush()

    storage = MMapEmbeddingStorage(num_embeddings, embedding_dim, str(path))

    assert os.path.exists(path)
    assert isinstance(storage.mmap, np.memmap)
    assert storage.mmap.shape == (num_embeddings, embedding_dim)
    assert storage.mmap.dtype == np.float32
    # Check if data is loaded from existing file
    assert np.allclose(storage.mmap, dummy_data)

def test_mmap_embedding_storage_forward_method(tmp_path):
    path = tmp_path / "forward_embeddings.dat"
    num_embeddings = 10
    embedding_dim = 5
    
    # Create storage with some data
    storage = MMapEmbeddingStorage(num_embeddings, embedding_dim, str(path))
    initial_data = np.copy(storage.mmap)

    # Mock indices
    # [Batch, Seq, Heads]
    indices = torch.tensor([
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]]
    ], dtype=torch.long)

    output = storage.forward(indices)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (indices.shape[0], indices.shape[1], indices.shape[2], embedding_dim)
    assert output.dtype == torch.float32
    assert output.device == indices.device # Should be on the same device as input indices

    # Verify retrieved values
    expected_output = torch.empty_like(output)
    for b in range(indices.shape[0]):
        for s in range(indices.shape[1]):
            for h in range(indices.shape[2]):
                idx = indices[b, s, h].item()
                expected_output[b, s, h] = torch.from_numpy(initial_data[idx])
    
    assert torch.allclose(output, expected_output)

def test_mmap_embedding_storage_forward_method_out_of_bounds(tmp_path):
    path = tmp_path / "oob_embeddings.dat"
    num_embeddings = 5
    embedding_dim = 10
    
    storage = MMapEmbeddingStorage(num_embeddings, embedding_dim, str(path))

    # Indices with out-of-bounds value
    indices = torch.tensor([
        [[0, 1], [4, 5]] # 5 is out of bounds for num_embeddings=5
    ], dtype=torch.long)

    with pytest.raises(IndexError):
        storage.forward(indices)