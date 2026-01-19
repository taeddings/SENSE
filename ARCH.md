# SENSE-v2 Architecture Documentation

## 1. System Overview

### Vision and Purpose

SENSE-v2 is a self-evolving, agent-driven framework merging neural evolution with autonomous operational control. The system is designed to:

- Enable autonomous learning through tool-verified success
- Provide hierarchical task orchestration for OS-level operations
- Maintain structured knowledge persistence with intelligent memory management

### Three-Layer Architecture

```
+------------------------------------------------------------------+
|                         SENSE-v2 Framework                        |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------------+  +------------------------+            |
|  |    Agent 0 (School)    |  |  Agent Zero (Workplace)|            |
|  |                        |  |                        |            |
|  |  +-----------------+   |  |  +-----------------+   |            |
|  |  | CurriculumAgent |   |  |  |   MasterAgent   |   |            |
|  |  | (Teacher)       |   |  |  | (Orchestrator)  |   |            |
|  |  +-----------------+   |  |  +-----------------+   |            |
|  |          |             |  |          |             |            |
|  |  +-----------------+   |  |  +-----------------+   |            |
|  |  | ExecutorAgent   |   |  |  |   Sub-Agents    |   |            |
|  |  | (Student)       |   |  |  | Terminal/FS/Web |   |            |
|  |  +-----------------+   |  |  +-----------------+   |            |
|  |          |             |  |                        |            |
|  |  +-----------------+   |  +------------------------+            |
|  |  |  GRPOTrainer    |   |                                        |
|  |  +-----------------+   |                                        |
|  +------------------------+                                        |
|                                                                    |
|  +------------------------------------------------------------+   |
|  |                    AgeMem (Filing Cabinet)                  |   |
|  |  +------------------+         +------------------+          |   |
|  |  |  STM (Working)   |  <-->   |  LTM (Vector DB) |          |   |
|  |  +------------------+         +------------------+          |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  +------------------------------------------------------------+   |
|  |                       Tool System                           |   |
|  |  [Terminal] [FileSystem] [Memory] [Anomaly Detection]       |   |
|  +------------------------------------------------------------+   |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 2. Component Architecture

### A. Agent 0 (The School)

Agent 0 implements a co-evolutionary learning system using Step-wise Group Relative Policy Optimization (GRPO).

#### CurriculumAgent (Teacher)
**Location:** `sense_v2/agents/agent_0/curriculum.py`

Responsibilities:
- Generate progressively harder tasks based on student performance
- Adapt curriculum difficulty dynamically
- Track curriculum stages (Foundation -> Expert)
- Provide hints when student struggles

```python
# Curriculum stages with difficulty ranges
STAGES = [
    ("Foundation", (0.1, 0.3)),      # Basic operations
    ("Basic Operations", (0.2, 0.5)),
    ("Intermediate", (0.4, 0.7)),
    ("Advanced", (0.6, 0.85)),
    ("Expert", (0.8, 1.0)),
]
```

#### ExecutorAgent (Student)
**Location:** `sense_v2/agents/agent_0/executor.py`

Responsibilities:
- Execute curriculum tasks using available tools
- Learn from success/failure via reward signals
- Evolve execution strategies over time
- Implement self-correction based on error analysis

Execution Strategies:
1. **Direct Execution** - Attempt task immediately
2. **Plan-then-Execute** - Analyze before executing
3. **Iterative Refinement** - Retry with corrections

#### GRPOTrainer
**Location:** `sense_v2/agents/agent_0/trainer.py`

Implements Step-wise GRPO training:
1. Generate task groups from curriculum
2. Run multiple executors on each task
3. Compute relative advantages within groups
4. Evolve population through selection, crossover, and mutation

```python
# GRPO Parameters
temperature = 1.0      # Sampling temperature
kl_coeff = 0.01        # KL divergence coefficient
population_size = 5    # Number of executor agents
group_size = 4         # Tasks per training step
```

---

### B. Agent Zero (The Workplace)

Agent Zero provides hierarchical orchestration for OS-level task execution.

#### MasterAgent (Orchestrator)
**Location:** `sense_v2/agents/agent_zero/master.py`

Per SYSTEM_PROMPT requirements:
- **Never performs heavy computation** - delegates to sub-agents
- Aggregates results to keep context window lean
- Enforces maximum delegation depth

Task Types:
- `TERMINAL` - Shell command execution
- `FILESYSTEM` - File operations
- `BROWSER` - Web interactions
- `REASONING` - Analytical tasks
- `COMPOSITE` - Multi-step tasks

#### Sub-Agents
**Location:** `sense_v2/agents/agent_zero/sub_agents.py`

Specialized agents for:
- **TerminalAgent** - Command execution
- **FileSystemAgent** - File operations
- **BrowserAgent** - Web interactions

---

### C. AgeMem (The Filing Cabinet)

AgeMem provides structured knowledge persistence with automatic tiering.

**Location:** `sense_v2/memory/agemem.py`

#### Short-Term Memory (STM)
**Location:** `sense_v2/memory/stm.py`

- Fast, token-limited working memory
- Priority-based retention
- Automatic summarize-and-prune at 80% capacity

#### Long-Term Memory (LTM)
**Location:** `sense_v2/memory/ltm.py`

- Persistent vector database storage
- Semantic similarity search
- Tiered storage (hot -> warm -> cold)

#### Key Features:
1. **Automatic Tiering** - Entries age from hot to cold based on access patterns
2. **Context-Aware Pruning** - Triggers at 80% of model's context limit
3. **Memory Consolidation** - High-priority STM entries promote to LTM
4. **Semantic Search** - Vector embeddings for similarity queries

```python
# Memory Configuration
stm_token_limit = 4096     # Maximum STM tokens
prune_threshold = 0.8      # Trigger pruning at 80%
embedding_dim = 384        # Vector embedding dimension
```

---

## 3. Tool System

### Schema-Based Tool Registration

All tools inherit from `BaseTool` and register via `@ToolRegistry.register`.

**Location:** `sense_v2/core/base.py`

```python
@ToolRegistry.register
class MyTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="my_tool",
            description="Tool description",
            parameters=[...],
        )

    async def execute(self, **kwargs) -> ToolResult:
        # Implementation
```

### Self-Correction via stderr Feedback

Tools implement automatic retry with error analysis:
1. Parse stderr for recoverable error patterns
2. Apply exponential backoff between retries
3. Return detailed error information for learning

### Available Tools

| Tool | Location | Purpose |
|------|----------|---------|
| `terminal_exec` | `tools/terminal.py` | Execute shell commands |
| `terminal_interactive` | `tools/terminal.py` | Interactive terminal sessions |
| `file_read` | `tools/filesystem.py` | Read file contents |
| `file_write` | `tools/filesystem.py` | Write file contents |
| `file_list` | `tools/filesystem.py` | List directory contents |
| `file_exists` | `tools/filesystem.py` | Check file existence |
| `memory_store` | `tools/memory_tools.py` | Store in memory |
| `memory_search` | `tools/memory_tools.py` | Search memory |
| `memory_retrieve` | `tools/memory_tools.py` | Retrieve by key |
| `memory_stats` | `tools/memory_tools.py` | Get memory statistics |
| `memory_profile` | `tools/memory_tools.py` | Profile VRAM/RAM with recommendations |
| `anomaly_detect` | `tools/anomaly.py` | Detect anomalies in data |

---

## 4. Hardware Optimization

### Target Configuration
- **Memory:** 128GB Unified Memory Architecture (UMA)
- **Bus:** 256-bit memory bus
- **GPU:** AMD RDNA 3.5 via ROCm

### Design Principles

1. **UMA Awareness** - All inference and data handling optimized for massive VRAM overhead with bandwidth efficiency priority

2. **AMD/ROCm Priority** - Avoid NVIDIA-exclusive (CUDA-only) libraries in favor of:
   - ROCm-compatible PyTorch
   - vLLM with ROCm support
   - Cross-platform implementations

3. **Memory Efficiency**
   - Streaming data processing where possible
   - Lazy loading for large datasets
   - Intelligent caching with tiered eviction

---

## 5. Memory-Aware Inference Layer

SENSE-v2 includes a memory-aware inference system that optimizes for constrained environments through cross-platform kernel fusion and conditional memory (Engram).

### Cross-Platform Strategy

| Hardware | Backend | Memory Savings |
|----------|---------|----------------|
| NVIDIA GPU (CUDA) | Triton kernel fusion | ~84% |
| AMD GPU (ROCm) | Triton experimental | ~84% |
| Apple Silicon (MPS) | PyTorch chunked | ~50-60% |
| CPU / ARM64 | PyTorch chunked | ~50-60% |

**Design Principle:** Runtime feature detection with graceful degradation. Never hard-fail on missing Triton.

### Fused Linear Cross-Entropy Kernels
**Location:** `sense_v2/llm/kernels/`

The fused kernel avoids materializing the full logits tensor (batch_size x vocab_size) by computing cross-entropy loss in vocabulary chunks.

```python
# Usage
from sense_v2.llm.kernels import fused_linear_cross_entropy, get_backend

# Auto-detects best backend
loss = fused_linear_cross_entropy(hidden, weight, labels)

# Check current backend
print(get_backend())  # "triton", "pytorch_cuda", "pytorch_mps", or "pytorch_cpu"
```

Key files:
- `kernels/__init__.py` - Backend detection and exports
- `kernels/functional.py` - PyTorch autograd wrapper with chunked fallback
- `kernels/triton_src.py` - Triton CUDA/ROCm kernels

### Conditional Memory (Engram)
**Location:** `sense_v2/engram/`

Engram provides static lookup to offload repeated computations to memory:

```
Input IDs → Shadow Map Compression → N-gram Hash → Table Lookup → Fusion with Hidden States
```

#### N-gram Hashing
**Location:** `sense_v2/models/components/hashing.py`

Multi-head XOR hashing for collision-resistant lookups:
```python
hash_h(ngram) = XOR(token_i * prime_h^i) mod table_size
```

- Multiple hash heads with different prime multipliers
- Supports configurable n-gram orders (default: 2, 3)
- Deterministic and reproducible

### Memory Hierarchy
**Location:** `sense_v2/memory/hierarchy.py`

Three-tier memory hierarchy for efficient data placement:

```
┌─────────────────┐
│  L1: GPU VRAM   │  ← Fastest, limited capacity
├─────────────────┤
│  L2: Host RAM   │  ← Pinned memory for fast DMA
├─────────────────┤
│  L3: Disk       │  ← Memory-mapped, largest capacity
└─────────────────┘
```

Key components:
- `EmbeddingPrefetcher` - Async prefetching from host RAM to GPU
- `MemoryHierarchy` - LRU caching across tiers
- `PinnedMemoryPool` - Pre-allocated pinned buffers

### Memory-Aware Configuration
**Location:** `sense_v2/core/config.py`

```python
@dataclass
class MemoryAwareConfig:
    max_ram_usage_mb: int = 4096
    use_fused_kernels: bool = True
    enable_host_offload: bool = True
    memory_warning_threshold: float = 0.60  # Activate Engram
    memory_critical_threshold: float = 0.75  # Enable fused kernels
```

### Automatic Activation

The system monitors memory pressure and automatically enables optimizations:

| Memory Usage | Actions |
|--------------|---------|
| < 60% | Normal operation |
| 60-75% | Activate Engram conditional memory |
| 75-90% | Enable fused kernels |
| > 90% | Critical mode, reduce batch size |

---

## 5.1 Binary Protocol Layer (DRGN)

SENSE-v2 includes a high-performance binary protocol for inter-agent communication.

### Protocol Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENSE Binary Protocol Stack                   │
├─────────────────────────────────────────────────────────────────┤
│  SENSEMessage (High-Level API)                                  │
│    ├── create_request() / create_response()                     │
│    └── Uses DRGNHeader + BinaryParser                           │
├─────────────────────────────────────────────────────────────────┤
│  DRGNHeader (Fixed 29-byte wire format)                         │
│    └── Network byte order (!), CRC32 integrity                  │
├─────────────────────────────────────────────────────────────────┤
│  BinaryParser (Zero-Copy In-Place Parsing)                      │
│    └── memoryview slices, lazy UTF-8 decoding                   │
├─────────────────────────────────────────────────────────────────┤
│  EngramManager (Resource-Safe Buffer Management)                │
│    └── mmap + ExitStack for Termux file descriptor safety       │
└─────────────────────────────────────────────────────────────────┘
```

### DRGN Header Format (29 bytes)

```
Offset  Size   Field            Type      Description
──────────────────────────────────────────────────────────────
0       4      Signature        uint32    Magic 'DRGN' (0x4452474E)
4       4      TotalBytes       uint32    Message size after this field
8       1      ProtocolVersion  uint8     Version (0x00 for v1)
9       4      MethodID         uint32    Remote function identifier
13      4      Flags            uint32    Payload type bitmask
17      8      MessageID        uint64    Async request/response tracking
25      4      CRC32            uint32    Checksum of payload
```

**Why Network Byte Order (`!`)?**
- Standardizes to big-endian across all architectures
- Prevents ARM64 unaligned access faults (common on Android/Termux)
- Standard wire format matching TCP/IP conventions

### Key Components

**Location:** `sense_v2/protocol/`

| File | Description |
|------|-------------|
| `constants.py` | Protocol constants, limits, flags |
| `exceptions.py` | Custom exception classes |
| `header.py` | DRGNHeader dataclass |
| `parser.py` | BinaryParser with zero-copy |
| `message.py` | SENSEMessage high-level API |
| `async_io.py` | Async stream reading/writing |
| `adapters.py` | AgentMessage ↔ SENSEMessage adapters |
| `serializers.py` | MessagePack/JSON serialization |

### Usage Examples

```python
# Creating a request
from sense_v2.protocol import SENSEMessage, METHOD_ID_AGENT_USER

msg = SENSEMessage.create_request(
    method_id=METHOD_ID_AGENT_USER,
    payload={"content": "Hello!"},
)
wire_bytes = msg.to_bytes()

# Parsing a message
msg = SENSEMessage.parse(wire_bytes)
print(msg.payload)  # {'content': 'Hello!'}

# AgentMessage integration
from sense_v2.protocol import AgentMessageAdapter
binary = agent_msg.to_binary()  # Added via monkey-patching
agent_msg = AgentMessage.from_binary(binary)
```

### EngramManager

**Location:** `sense_v2/engram/manager.py`

Resource-safe buffer management for memory-mapped files:

```python
with EngramManager('/path/to/engram.dat') as manager:
    parser = manager.get_parser()  # Zero-copy BinaryParser
    view = manager.get_slice(0, 1024)  # Direct memoryview
```

**Why ExitStack?**
Termux on Android has limited file descriptors. ExitStack ensures ALL resources are closed even if exceptions occur during processing.

### Memory-Aware Fitness Function
**Location:** `sense_v2/agents/agent_0/trainer.py`

Multi-objective evolutionary fitness that penalizes memory usage:

```python
fitness = (
    accuracy * 0.6 +
    memory_efficiency * 0.2 +
    drift_resistance * 0.2
)
```

This ensures evolved agents optimize for both accuracy AND memory efficiency.

---

## 6. API Layer

### Flask REST Endpoints
**Location:** `sense_v2/api/app.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register` | POST | User registration |
| `/login` | POST | User authentication |
| `/logout` | POST | End session |
| `/profile` | GET | Get user profile |

### Session-Based Authentication

- Server-side session storage
- Secure password hashing via bcrypt
- Session cookie management

---

## 7. Configuration

### Main Configuration
**Location:** `sense_v2/core/config.py`

```python
@dataclass
class Config:
    evolution: EvolutionConfig      # Agent 0 settings
    orchestration: OrchestrationConfig  # Agent Zero settings
    memory: MemoryConfig            # AgeMem settings
    engram: EngramConfig            # Engram conditional memory
    memory_aware: MemoryAwareConfig # Memory-aware inference
    protocol: ProtocolConfig        # Binary protocol settings [NEW]
```

### Key Configuration Classes:

- **EvolutionConfig** - GRPO parameters, population size, curriculum stages
- **OrchestrationConfig** - Delegation depth, timeouts, context limits
- **MemoryConfig** - Token limits, persistence paths, embedding settings
- **EngramConfig** - Table size, n-gram orders, layer indices
- **MemoryAwareConfig** - Fused kernels, host offload, memory thresholds
- **ProtocolConfig** - Message size limits, serialization, async I/O settings

---

## 8. Validation & Documentation Rules

Per SYSTEM_PROMPT.md requirements:

1. Every PR/refactor must update `ARCH.md` or `SENSE_DOCS.md`
2. New tools in `tools/` must include `test_[toolname].py`
3. Maintain "State Log" in `dev_log.json` for evolutionary progress
4. Binary/scalar rewards based on Unit Test Success or Terminal Exit Codes

---

## 9. Directory Structure

```
sense_v2/
├── agents/
│   ├── agent_0/          # The School
│   │   ├── curriculum.py # CurriculumAgent (Teacher)
│   │   ├── executor.py   # ExecutorAgent (Student)
│   │   └── trainer.py    # GRPOTrainer (with memory-aware fitness)
│   └── agent_zero/       # The Workplace
│       ├── master.py     # MasterAgent (with resource checks)
│       └── sub_agents.py # Terminal/FS/Browser agents
├── api/
│   └── app.py            # Flask REST API
├── core/
│   ├── base.py           # Base classes
│   ├── config.py         # Configuration (includes MemoryAwareConfig)
│   └── schemas.py        # Schema definitions
├── engram/               # Conditional Memory
│   ├── manager.py        # EngramManager (mmap + ExitStack) [NEW]
│   ├── model.py          # EngramFusionLayer
│   ├── storage.py        # MMapEmbeddingStorage
│   └── tokenizer.py      # Shadow map tokenizer
├── protocol/             # Binary Protocol Layer [NEW]
│   ├── __init__.py       # Module exports
│   ├── constants.py      # Protocol constants, limits, flags
│   ├── exceptions.py     # Custom exception classes
│   ├── header.py         # DRGNHeader dataclass
│   ├── parser.py         # BinaryParser with zero-copy
│   ├── message.py        # SENSEMessage high-level API
│   ├── async_io.py       # Async stream reading/writing
│   ├── adapters.py       # AgentMessage adapters
│   └── serializers.py    # MessagePack/JSON serialization
├── llm/
│   ├── base.py           # LLM base classes
│   ├── providers.py      # LLM providers
│   └── kernels/          # Memory-aware kernels [NEW]
│       ├── __init__.py   # Backend detection
│       ├── functional.py # PyTorch chunked fallback
│       └── triton_src.py # Triton CUDA/ROCm kernels
├── memory/
│   ├── agemem.py         # Unified memory system
│   ├── stm.py            # Short-term memory
│   ├── ltm.py            # Long-term memory
│   ├── embeddings.py     # Embedding providers
│   └── hierarchy.py      # Host prefetcher [NEW]
├── models/
│   ├── user.py           # User model
│   └── components/       # Reusable components [NEW]
│       └── hashing.py    # N-gram hashing
├── tools/
│   ├── terminal.py       # Terminal tools
│   ├── filesystem.py     # File system tools
│   ├── memory_tools.py   # Memory tools (includes MemoryProfileTool)
│   └── anomaly.py        # Anomaly detection
├── utils/
│   ├── security.py       # Security utilities
│   ├── dev_log.py        # Development logging
│   └── health.py         # Health checks
└── database/
    ├── database.py       # Database connection
    └── schema.py         # Database schema
```
