import os
import numpy as np
from transformers import AutoTokenizer
from typing import List
import unicodedata
import torch # Added for EngramTokenizer __call__ method

class EngramTokenizerBuilder:
    def __init__(self, hf_model_name: str):
        self.hf_model_name = hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        # Corrected vocab_size calculation
        self.vocab_size = max(self.tokenizer.vocab.values()) + 1 if self.tokenizer.vocab else 0 

    def _normalize_token(self, token: str) -> str:
        """Applies NFKC normalization, lowercasing, and stripping."""
        token = unicodedata.normalize('NFKC', token)
        token = token.lower()
        token = token.strip()
        return token

    def build_mapping(self, output_path: str):
        """
        Builds a surjective mapping from the HuggingFace tokenizer's vocabulary
        to canonical semantic IDs and saves it as a NumPy array.
        """
        canonical_map = np.full(self.vocab_size, -1, dtype=np.int32)
        
        canonical_ids = {}
        next_canonical_id = 0

        for token_str, token_id in self.tokenizer.vocab.items():
            normalized_token = self._normalize_token(token_str)
            
            if normalized_token not in canonical_ids:
                canonical_ids[normalized_token] = next_canonical_id
                next_canonical_id += 1
            
            canonical_map[token_id] = canonical_ids[normalized_token]
        
        np.save(output_path, canonical_map, allow_pickle=False)

class EngramTokenizer:
    def __init__(self, shadow_map_path: str):
        self.shadow_map_path = shadow_map_path
        self.shadow_map = np.load(shadow_map_path, mmap_mode='r')

    def __call__(self, input_ids: torch.Tensor):
        input_ids_np = input_ids.cpu().numpy()
        
        compressed_ids_np = self.shadow_map[input_ids_np]
        
        compressed_ids = torch.from_numpy(compressed_ids_np).to(input_ids.device)
        
        return compressed_ids