# Implementation Plan: Engram Conditional Memory Architecture Integration

## Phase 1: Infrastructure & Configuration
- [ ] Task: Create directory structure for Engram module
    - [x] Create `sense_v2/engram/` directory
    - [x] Create `sense_v2/engram/__init__.py`
- [x] Task: Update `sense_v2/core/config.py` for EngramConfig
    - [x] Write Failing Tests: Add tests for `EngramConfig` dataclass definition and its integration into the main `Config` class.
    - [x] Implement: Define `EngramConfig` dataclass with specified fields and default values.
    - [x] Implement: Add `engram: EngramConfig` field to the main `Config` class.
    - [x] Implement: Add necessary imports (`field`, `List`) to `config.py`.
    - [x] Refactor: Review and refactor `config.py` changes.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure & Configuration' (Protocol in workflow.md) [checkpoint: 266cafd]

## Phase 2: The Shadow Tokenizer Engine
- [ ] Task: Create `sense_v2/engram/tokenizer.py`
    - [x] Create `sense_v2/engram/tokenizer.py` file
- [x] Task: Implement `EngramTokenizerBuilder` class
    - [x] Write Failing Tests: Add tests for `EngramTokenizerBuilder` initialization, vocabulary iteration, normalization logic (NFKC, lowercase, strip), and saving the mapping to disk.
    - [x] Implement: Define `EngramTokenizerBuilder` class.
    - [x] Implement: Logic to accept HuggingFace model name.
    - [x] Implement: Logic to iterate vocabulary and build surjective mapping with normalization.
    - [x] Implement: Logic to save mapping as NumPy array.
    - [x] Refactor: Review and refactor `EngramTokenizerBuilder`.
- [x] Task: Implement `EngramTokenizer` class
    - [x] Write Failing Tests: Add tests for `EngramTokenizer` initialization (loading mmap), and the `__call__` method for performing lookups and returning compressed tensors.
    - [x] Implement: Define `EngramTokenizer` class.
    - [x] Implement: Logic to load NumPy map in `mmap_mode='r'`.
    - [x] Implement: `__call__` method for lookup and tensor conversion.
    - [x] Refactor: Review and refactor `EngramTokenizer`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: The Shadow Tokenizer Engine' (Protocol in workflow.md)

## Phase 3: Scalable Storage Backend
- [ ] Task: Create `sense_v2/engram/storage.py`
    - [ ] Create `sense_v2/engram/storage.py` file
- [ ] Task: Implement `MMapEmbeddingStorage(nn.Module)` class
    - [ ] Write Failing Tests: Add tests for `MMapEmbeddingStorage` initialization (creating/loading memmap, random noise initialization), and the `forward` method for slicing and returning tensors.
    - [ ] Implement: Define `MMapEmbeddingStorage` class inheriting from `nn.Module`.
    - [ ] Implement: `__init__` logic for creating/loading `np.memmap` and initialization.
    - [ ] Implement: `forward` method for slicing memmap and tensor conversion.
    - [ ] Refactor: Review and refactor `MMapEmbeddingStorage`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Scalable Storage Backend' (Protocol in workflow.md)

## Phase 4: Fusion & Modeling
- [ ] Task: Create `sense_v2/engram/model.py`
    - [ ] Create `sense_v2/engram/model.py` file
- [ ] Task: Implement `EngramFusionLayer(nn.Module)` class
    - [ ] Write Failing Tests: Add tests for `EngramFusionLayer` initialization, component integration, hashing logic (multi-head aspect), and the `forward` pass computations.
    - [ ] Implement: Define `EngramFusionLayer` class inheriting from `nn.Module`.
    - [ ] Implement: `__init__` to integrate `EngramTokenizer` and `MMapEmbeddingStorage`.
    - [ ] Implement: Fusion Gates (`nn.Linear`, `RMSNorm`, Sigmoid) and Refinement (`nn.Conv1d`).
    - [ ] Implement: Multi-Head Multiplicative-XOR hash function.
    - [ ] Implement: `forward` pass logic.
    - [ ] Refactor: Review and refactor `EngramFusionLayer`.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Fusion & Modeling' (Protocol in workflow.md)

## Phase 5: Setup Scripts
- [ ] Task: Create `scripts/setup_engram.py`
    - [ ] Create `scripts/setup_engram.py` file
- [ ] Task: Implement `setup_engram.py` logic
    - [ ] Write Failing Tests: Add tests for `setup_engram.py` to ensure it correctly imports `EngramTokenizerBuilder`, downloads the tokenizer, and builds/saves `shadow_map.npy` efficiently.
    - [ ] Implement: Import `EngramTokenizerBuilder`.
    - [ ] Implement: Logic to download tokenizer (default: "deepseek-ai/deepseek-coder-6.7b-base").
    - [ ] Implement: Logic to build and save `shadow_map.npy` to `data/` directory.
    - [ ] Refactor: Review and refactor `setup_engram.py`.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Setup Scripts' (Protocol in workflow.md)