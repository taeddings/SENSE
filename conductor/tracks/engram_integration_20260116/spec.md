# Specification: Engram Conditional Memory Architecture Integration

## Overview
Implement the "Engram" Conditional Memory architecture into the SENSE-v2 framework. This is a Hybrid Sparse Model upgrade that offloads static knowledge to a massive CPU-resident embedding table, enabling massive static memory lookups (O(1)) via a "Dual Tokenizer" system. The target architecture is a generalized implementation using Memory-Mapped files for hardware-agnostic deployment (Intel/Edge compatible).

## Functional Requirements

### Phase 1: Infrastructure & Configuration
*   **Directory Structure:** Create `sense_v2/engram/` and an `__init__.py` file within it.
*   **Configuration Update:**
    *   Modify `sense_v2/core/config.py`.
    *   Define a new dataclass `EngramConfig` with the following fields:
        *   `enabled: bool = True`
        *   `storage_path: str = "data/engram_table.dat"`
        *   `shadow_map_path: str = "data/shadow_map.npy"`
        *   `table_size: int = 10_000_000`
        *   `engram_dim: int = 1024`
        *   `num_heads: int = 8`
        *   `ngram_orders: List[int] = field(default_factory=lambda: [2, 3])`
        *   `layer_indices: List[int] = field(default_factory=lambda: [2, 15])`
    *   **Note:** Ensure `field` and `List` are imported from the `dataclasses` and `typing` modules respectively in `config.py` to support the default factories for `ngram_orders` and `layer_indices`.
    *   Add an `engram` field of type `EngramConfig` to the main `Config` class.

### Phase 2: The Shadow Tokenizer Engine
*   **File Creation:** Create `sense_v2/engram/tokenizer.py`.
*   **`EngramTokenizerBuilder` Class:**
    *   Accepts a HuggingFace model name.
    *   Iterates the vocabulary to build a surjective mapping (P: V -> V').
    *   **Logic:** Normalize text using NFKC + Lowercase + Strip. Map semantically equivalent tokens to a single canonical ID.
    *   **Output:** Saves the mapping as a NumPy array to disk.
*   **`EngramTokenizer` Class:**
    *   Loads the saved NumPy map in `mmap_mode='r'` (read-only memory map).
    *   **Call Method:** Accepts `input_ids` (Tensor), converts to CPU/NumPy, performs the lookup, and returns the compressed Tensor on the original device.

### Phase 3: Scalable Storage Backend
*   **File Creation:** Create `sense_v2/engram/storage.py`.
*   **`MMapEmbeddingStorage(nn.Module)` Class:**
    *   **Init:** Accepts `num_embeddings`, `embedding_dim`, and `path`.
    *   **Logic:**
        *   If `path` does not exist: Create a new `np.memmap` (mode='w+'), initialize with random normal noise, and flush.
        *   If `path` exists: Load `np.memmap` (mode='r+').
    *   **Forward:** Accepts indices `[Batch, Seq, Heads]`. Slice the memmap using CPU indices, convert to Tensor, move to device, and return `[Batch, Seq, Heads, Dim]`.

### Phase 4: Fusion & Modeling
*   **File Creation:** Create `sense_v2/engram/model.py`.
*   **`EngramFusionLayer(nn.Module)` Class:**
    *   **Init:** Accepts `config`, `backbone_hidden_size`, and `layer_id`.
    *   **Components:**
        *   `EngramTokenizer` (from Phase 2).
        *   `MMapEmbeddingStorage` (from Phase 3).
        *   **Fusion Gates:** `nn.Linear` projectors ($W_k, W_v$) mapping `engram_dim` -> `backbone_hidden_size`, `RMSNorm`, and a Sigmoid gate mechanism.
        *   **Refinement:** `nn.Conv1d` (kernel=3, padding=1).
    *   **Hashing Logic:** Implement a generic Multi-Head Multiplicative-XOR hash function (NumPy/Torch compatible). This hashing logic must handle the multi-head aspect, producing multiple indices per token to match the collision-avoidance strategy.
    *   **Forward Pass:**
        1.  Compress Input -> Hash.
        2.  Retrieve Memory -> Project to Hidden Size.
        3.  Compute Gate $\alpha = \text{sigmoid}(Q \cdot K)$.
        4.  Output $= \text{Hidden} + (\alpha \cdot V) + \text{Conv}(\alpha \cdot V)$.

### Phase 5: Setup Scripts
*   **File Creation:** Create `scripts/setup_engram.py`.
*   **Logic:**
    *   Import `EngramTokenizerBuilder`.
    *   Download the tokenizer (default: "deepseek-ai/deepseek-coder-6.7b-base").
    *   Build and save `shadow_map.npy` to the `data/` directory.

## Non-Functional Requirements
*   **Hardware Agnosticism:** The implementation must be generalized and avoid hardcoding platform-specific paths, ensuring compatibility across various hardware (e.g., Intel/Edge).
*   **Library Usage:** Utilize standard PyTorch and NumPy libraries.
*   **Performance:** Achieve O(1) constant-time static memory lookups.
*   **Modularity:** The architecture should maintain a modular design with clear separation of concerns for tokenizer, storage, and fusion layers.
*   **Memory Efficiency (Setup Script):** The `EngramTokenizerBuilder` should process the vocabulary without loading unnecessary model weights, as only the tokenizer is needed for building the shadow map.

## Acceptance Criteria
*   All specified files and directories are created as per the plan.
*   The `EngramConfig` dataclass is correctly defined and integrated into the main `Config` class in `sense_v2/core/config.py`, including necessary imports for `field` and `List`.
*   `EngramTokenizerBuilder` successfully generates and saves the `shadow_map.npy` mapping, processing the vocabulary efficiently without loading unnecessary model weights.
*   `EngramTokenizer` correctly loads the memory-mapped NumPy array and performs O(1) lookups.
*   `MMapEmbeddingStorage` correctly initializes and accesses memory-mapped files for embedding storage.
*   `EngramFusionLayer` accurately integrates the tokenizer, storage, hashing (including multi-head aspect), and fusion logic as described.
*   The `scripts/setup_engram.py` script successfully builds the `shadow_map.npy`.
