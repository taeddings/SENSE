import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch
# Import EngramTokenizerBuilder and EngramTokenizer directly from the module under test
from sense_v2.engram.tokenizer import EngramTokenizerBuilder, EngramTokenizer
import torch

# Mock AutoTokenizer for testing purposes, patching it where it's used in the module
@pytest.fixture
def mock_auto_tokenizer():
    with patch('sense_v2.engram.tokenizer.AutoTokenizer.from_pretrained') as mock_from_pretrained:
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab = {
            "hello": 1, "Hello": 2, "HELLO": 3,
            "world": 4, "World": 5, "WORLD": 6,
            "apple": 7, "Apple": 8, "APPLE": 9,
            "banana": 10, "Banana": 11,
            " ": 12, "!": 13, ".": 14 # Max token_id is 14
        }
        mock_from_pretrained.return_value = mock_tokenizer
        yield mock_from_pretrained # Yield the mock object itself

# Test EngramTokenizerBuilder
def test_engram_tokenizer_builder_initialization(mock_auto_tokenizer):
    builder = EngramTokenizerBuilder("mock_model")
    assert builder.hf_model_name == "mock_model"
    mock_auto_tokenizer.assert_called_once_with("mock_model") # Changed to assert on the mock object itself

def test_engram_tokenizer_builder_build_mapping(mock_auto_tokenizer, tmp_path):
    output_path = tmp_path / "shadow_map.npy"
    builder = EngramTokenizerBuilder("mock_model")
    builder.build_mapping(output_path)

    assert os.path.exists(output_path)
    shadow_map = np.load(output_path, mmap_mode='r')
    # The vocab size should be max_token_id + 1
    assert shadow_map.shape[0] == max(mock_auto_tokenizer.return_value.vocab.values()) + 1
    
    # Example assertions for mapping logic
    # Assuming 'hello', 'Hello', 'HELLO' map to the same canonical ID
    assert shadow_map[mock_auto_tokenizer.return_value.vocab["hello"]] == shadow_map[mock_auto_tokenizer.return_value.vocab["Hello"]]
    assert shadow_map[mock_auto_tokenizer.return_value.vocab["hello"]] == shadow_map[mock_auto_tokenizer.return_value.vocab["HELLO"]]
    assert shadow_map[mock_auto_tokenizer.return_value.vocab["world"]] != shadow_map[mock_auto_tokenizer.return_value.vocab["hello"]]


# Test EngramTokenizer
def test_engram_tokenizer_initialization(tmp_path):
    dummy_map_path = tmp_path / "dummy_shadow_map.npy"
    # Create a dummy map that can accommodate token_id up to 14 (size 15)
    np.save(dummy_map_path, np.arange(15), allow_pickle=False) 

    tokenizer = EngramTokenizer(dummy_map_path)
    assert os.path.exists(tokenizer.shadow_map_path)
    assert isinstance(tokenizer.shadow_map, np.memmap)

def test_engram_tokenizer_call_method(tmp_path):
    dummy_map_path = tmp_path / "dummy_shadow_map.npy"
    # Create a dummy map that can accommodate token_id up to 14 (size 15)
    # Example map: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]
    np.save(dummy_map_path, np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]), allow_pickle=False) 
    
    tokenizer = EngramTokenizer(dummy_map_path)
    
    # Mock input_ids tensor, ensuring max ID is within dummy map bounds
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) 
    
    compressed_ids = tokenizer(input_ids)
    
    assert isinstance(compressed_ids, torch.Tensor)
    assert compressed_ids.device == input_ids.device
    assert compressed_ids.shape == input_ids.shape
    # Further assertions would check the actual compression logic
    # For example, if token_id 1 and 2 map to 0 and 1 respectively in the dummy map
    assert compressed_ids[0] == 1 # input_ids[0] is 1, dummy_map[1] is 1
    assert compressed_ids[4] == 1 # input_ids[4] is 5, dummy_map[5] is 1 (Corrected assertion)
