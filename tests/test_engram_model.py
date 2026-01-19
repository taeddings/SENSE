import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
import os

# Mock necessary components for EngramFusionLayer
from sense_v2.core.config import EngramConfig
from sense_v2.engram.tokenizer import EngramTokenizer
from sense_v2.engram.storage import MMapEmbeddingStorage
from sense_v2.engram.model import EngramFusionLayer

# Fixture to create dummy files for EngramTokenizer and MMapEmbeddingStorage
@pytest.fixture
def setup_engram_files(tmp_path):
    # Create dummy shadow_map.npy
    shadow_map_path = tmp_path / "shadow_map.npy"
    # Needs to be large enough to cover token_ids up to 14 from mock_auto_tokenizer
    np.save(shadow_map_path, np.arange(15), allow_pickle=False)

    # Create dummy engram_table.dat
    engram_table_path = tmp_path / "engram_table.dat"
    num_embeddings = 10_000_000 # From EngramConfig default
    engram_dim = 1024 # From EngramConfig default
    dummy_mmap = np.memmap(engram_table_path, dtype=np.float32, mode='w+', shape=(num_embeddings, engram_dim))
    dummy_mmap[:] = np.random.normal(size=(num_embeddings, engram_dim)).astype(np.float32)
    dummy_mmap.flush()

    # Patch EngramConfig to use these dummy paths
    with patch('sense_v2.core.config.EngramConfig') as MockEngramConfig:
        mock_config_instance = MockEngramConfig.return_value
        mock_config_instance.shadow_map_path = str(shadow_map_path)
        mock_config_instance.storage_path = str(engram_table_path)
        mock_config_instance.table_size = num_embeddings
        mock_config_instance.engram_dim = engram_dim
        mock_config_instance.num_heads = 8 # Default from EngramConfig
        yield mock_config_instance

# Test EngramFusionLayer
def test_engram_fusion_layer_initialization(setup_engram_files):
    config = setup_engram_files
    backbone_hidden_size = 768
    layer_id = 2
    layer = EngramFusionLayer(config, backbone_hidden_size, layer_id)

    assert isinstance(layer, EngramFusionLayer)
    assert isinstance(layer.W_k, nn.Linear)
    assert isinstance(layer.W_v, nn.Linear)
    assert isinstance(layer.norm, nn.Module) # RMSNorm is a nn.Module
    assert isinstance(layer.sigmoid, nn.Sigmoid)
    assert isinstance(layer.conv1d, nn.Conv1d)
    assert isinstance(layer.tokenizer, EngramTokenizer)
    assert isinstance(layer.storage, MMapEmbeddingStorage)

def test_engram_fusion_layer_forward_pass(setup_engram_files):
    config = setup_engram_files
    backbone_hidden_size = 768
    layer_id = 2
    layer = EngramFusionLayer(config, backbone_hidden_size, layer_id)

    batch_size = 2
    seq_len = 10
    
    hidden_states = torch.randn(batch_size, seq_len, backbone_hidden_size)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)) # Dummy input_ids

    output = layer.forward(hidden_states, input_ids)

    assert isinstance(output, torch.Tensor)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype

def test_engram_fusion_layer_hashing_logic(setup_engram_files):
    config = setup_engram_files
    backbone_hidden_size = 768
    layer_id = 2
    layer = EngramFusionLayer(config, backbone_hidden_size, layer_id)

    compressed_ids = torch.randint(0, config.table_size, (2, 10)) # Batch, Seq
    hashed_indices = layer._multi_head_multiplicative_xor_hash(compressed_ids)

    assert isinstance(hashed_indices, torch.Tensor)
    assert hashed_indices.shape == (compressed_ids.shape[0], compressed_ids.shape[1], config.num_heads)
    assert hashed_indices.dtype == torch.long
