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

## 5. API Layer

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

## 6. Configuration

### Main Configuration
**Location:** `sense_v2/core/config.py`

```python
@dataclass
class Config:
    evolution: EvolutionConfig      # Agent 0 settings
    orchestration: OrchestrationConfig  # Agent Zero settings
    memory: MemoryConfig            # AgeMem settings
```

### Key Configuration Classes:

- **EvolutionConfig** - GRPO parameters, population size, curriculum stages
- **OrchestrationConfig** - Delegation depth, timeouts, context limits
- **MemoryConfig** - Token limits, persistence paths, embedding settings

---

## 7. Validation & Documentation Rules

Per SYSTEM_PROMPT.md requirements:

1. Every PR/refactor must update `ARCH.md` or `SENSE_DOCS.md`
2. New tools in `tools/` must include `test_[toolname].py`
3. Maintain "State Log" in `dev_log.json` for evolutionary progress
4. Binary/scalar rewards based on Unit Test Success or Terminal Exit Codes

---

## 8. Directory Structure

```
sense_v2/
├── agents/
│   ├── agent_0/          # The School
│   │   ├── curriculum.py # CurriculumAgent (Teacher)
│   │   ├── executor.py   # ExecutorAgent (Student)
│   │   └── trainer.py    # GRPOTrainer
│   └── agent_zero/       # The Workplace
│       ├── master.py     # MasterAgent
│       └── sub_agents.py # Terminal/FS/Browser agents
├── api/
│   └── app.py            # Flask REST API
├── core/
│   ├── base.py           # Base classes
│   ├── config.py         # Configuration
│   └── schemas.py        # Schema definitions
├── memory/
│   ├── agemem.py         # Unified memory system
│   ├── stm.py            # Short-term memory
│   ├── ltm.py            # Long-term memory
│   └── embeddings.py     # Embedding providers
├── tools/
│   ├── terminal.py       # Terminal tools
│   ├── filesystem.py     # File system tools
│   ├── memory_tools.py   # Memory tools
│   └── anomaly.py        # Anomaly detection
├── utils/
│   ├── security.py       # Security utilities
│   ├── dev_log.py        # Development logging
│   └── health.py         # Health checks
├── models/
│   └── user.py           # User model
└── database/
    ├── database.py       # Database connection
    └── schema.py         # Database schema
```
