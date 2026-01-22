# SENSE v2 Unified Evolutionary Architecture

## Project Overview
SENSE v2 is a closed-loop evolutionary engine with three concurrent optimization vectors:
1. **Model Weights**: Hyperparameters for LoRA/fine-tuning
2. **Reasoning Depth**: Token budget and verification loops
3. **Physical Grounding**: Sensor alignment strategies

## Implementation Progress

### ✅ Sprint 1: The Core (COMPLETED)

#### Task 1.1: ReasoningGenome ✅
**File:** `SENSE/core/evolution/genome.py`
- `Genome` abstract base class with mutation, crossover, serialization
- `ReasoningGenome` with hyperparameters, reasoning_budget, thinking_patterns
- Drift-adaptive mutation (HIGH drift → +10-20% reasoning_budget)
- FAISS embedding vector generation for LTM storage
- Backward transfer calculation

#### Task 1.2: PluginABC ✅
**File:** `SENSE/core/plugins/interface.py`
- `PluginABC` abstract interface for hardware/sensor plugins
- `PluginManifest` for capabilities declaration
- `SensorReading` and `SafetyConstraint` dataclasses
- `get_grounding_truth()` for hallucination detection
- `emergency_stop()` for safety interrupts

**File:** `SENSE/core/plugins/mock_sensor.py`
- `MockSensorPlugin` for testing with configurable noise
- `MockActuatorPlugin` with safety-checked operations
- Simulated temperature, humidity, pressure, light sensors

#### Task 1.3: Hardened Tools ✅
**File:** `SENSE/sense_v2/tools/terminal.py`
- Added `ExecutionError` class for self-correction loops
- Extended blocked command patterns (30+ dangerous patterns)
- Caution patterns with logging
- Command history tracking for security auditing
- Critical error detection in stderr

**File:** `SENSE/sense_v2/tools/filesystem.py`
- Added `FileSystemError` for proper error propagation
- `PROTECTED_PATHS` and `SENSITIVE_PATTERNS` for security
- `sanitize_path()` for traversal attack prevention
- `is_path_sensitive()` for sensitive file detection

#### Task 1.4: Memory Operations ✅
**File:** `SENSE/sense_v2/tools/memory_ops.py`
- `MemoryStoreTool` - Direct AgeMem store
- `MemoryQueryTool` - Semantic search
- `MemoryRetrieveTool` - Key-based retrieval
- `MemoryConsolidateTool` - Force STM→LTM consolidation
- `MemoryStatsTool` - Usage statistics
- `MemoryDeleteTool` - Entry deletion
- `MemorySummarizeTool` - Content summarization

#### Task 1.5: PopulationManager ✅
**File:** `SENSE/core/evolution/population.py`
- Population initialization and management
- Selection (tournament via DEAP if available, fitness-proportionate fallback)
- Crossover and mutation with drift adaptation
- `RetentionPolicy` for elite preservation (top 10% indefinitely)
- LTM checkpointing via AgeMem
- Generation statistics tracking
- Diversity metric calculation

### 🔄 Sprint 2: The Brain (IN PROGRESS)

#### Task 2.1-2.2: AdaptiveReasoningBudget ✅
**File:** `SENSE/sense_v2/llm/reasoning/compute_allocation.py`
- `AdaptiveReasoningBudget` with VRAM monitoring
- `ReasoningMode`: EFFICIENT, BALANCED, EXPLORATORY, COMPENSATORY
- Compensatory check: sensor OFFLINE → 2x budget, 2x verification
- Resource guard: VRAM >90% → cap at MIN_THRESHOLD
- Memory integration: >0.95 similarity → -30% budget

#### Task 2.3: ReasoningTrace Schema ✅
**File:** `SENSE/sense_v2/memory/engram_schemas.py`
- `ReasoningTrace` for complete reasoning process capture
- `DriftSnapshot` for concept drift state
- `GroundingRecord` for sensor verification
- Embedding generation for FAISS retrieval
- LTM flagging logic (grounding_score >0.8 AND success)

#### Task 2.4: ReasoningMemoryManager 🔲
**Planned:** `SENSE/sense_v2/memory/agemem_integration.py`

#### Task 2.5: EngramIngestionPipeline 🔲
**Planned:** `SENSE/core/plugins/engram_ingestion.py`

### 📋 Sprint 3: The Loop (PENDING)

- Task 3.1: DEAP integration for PopulationManager
- Task 3.2-3.3: Enhanced GRPOTrainer fitness function with backward transfer
- Task 3.4: CurriculumAgent workplace feedback
- Task 3.5: ReasoningOrchestrator with RestrictedPython sandbox

## Directory Structure (New Files)

```
SENSE/
├── core/                          # NEW
│   ├── __init__.py
│   ├── evolution/
│   │   ├── __init__.py
│   │   ├── genome.py             ✅ ReasoningGenome
│   │   └── population.py         ✅ PopulationManager
│   └── plugins/
│       ├── __init__.py
│       ├── interface.py          ✅ PluginABC
│       └── mock_sensor.py        ✅ MockSensorPlugin
├── sense_v2/
│   ├── llm/
│   │   └── reasoning/            # NEW
│   │       ├── __init__.py
│   │       └── compute_allocation.py  ✅ AdaptiveReasoningBudget
│   ├── memory/
│   │   └── engram_schemas.py     ✅ ReasoningTrace
│   └── tools/
│       ├── terminal.py           ✅ Hardened
│       ├── filesystem.py         ✅ Hardened
│       └── memory_ops.py         ✅ NEW
```

## Existing Infrastructure (Reused)

- `sense_v2/core/base.py` - BaseTool, BaseAgent, BaseMemory, AgentState, ToolRegistry
- `sense_v2/core/config.py` - Config system (needs extension for new modules)
- `sense_v2/core/schemas.py` - ToolSchema, ToolResult, RewardSignal
- `sense_v2/memory/agemem.py` - AgeMem with STM/LTM tiering
- `sense_v2/engram/manager.py` - EngramManager for mmap buffer management
- `sense_v2/agents/agent_0/trainer.py` - GRPOTrainer (needs extension)
- `sense_v2/agents/agent_0/curriculum.py` - CurriculumAgent (needs extension)

## Dependencies

**Required:**
```bash
pip install deap RestrictedPython psutil numpy
```

**Optional (for enhanced features):**
```bash
pip install nltk scikit-learn sentence-transformers chromadb
```

## Key Design Decisions

1. **Drift-Adaptive Mutation**: HIGH drift (>0.5) increases reasoning budget 10-20%, LOW drift (<0.2) decreases for efficiency
2. **Retention Policy**: Top 10% performers kept indefinitely, bottom 50% pruned after 5 generations
3. **VRAM Guards**: 75% warning threshold, 90% critical cap at MIN_THRESHOLD
4. **Grounding Verification**: Claims verified against sensor ground truth, <0.5 score triggers hallucination alert
5. **Compensatory Reasoning**: Sensor offline doubles verification depth and budget

## Next Steps

1. Complete ReasoningMemoryManager (Task 2.4)
2. Implement EngramIngestionPipeline (Task 2.5)
3. Add DEAP integration to PopulationManager (Task 3.1)
4. Extend config.py with ReasoningConfig and GroundingConfig
5. Create unit tests for verification
