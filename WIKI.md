# SENSE: Self-Evolving Neural Stabilization Engine

> **Version:** 4.0.0 (Robust Intelligence)
> **Status:** Production-Ready
> **Architecture:** AERL (Adaptive Evolutionary Reasoning Loop)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation & Configuration](#3-installation--configuration)
4. [Usage Guide](#4-usage-guide)
5. [Intelligence Layer (v4.0)](#5-intelligence-layer-v40)
6. [Memory Systems](#6-memory-systems)
7. [Tool Ecosystem](#7-tool-ecosystem)
8. [Security & Privacy](#8-security--privacy)
9. [Development Guide](#9-development-guide)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

**SENSE** is not merely a framework; it is an autonomous, active **engine** designed to elevate static Large Language Models (LLMs) into self-aware, evolving agents. Unlike passive chatbots that reset after every session, SENSE possesses persistent memory, learns from its experiences, and actively stabilizes its reasoning against hallucinations.

It is built on the philosophy of **Local-First AI**, running entirely on your hardware (from Android phones to high-end workstations) without relying on cloud APIs for its core intelligence.

### Core Capabilities

*   **Autonomy:** Executes complex, multi-step tasks without human hand-holding
*   **Evolution:** Genetic Memory actively selects and reinforces successful tool-usage patterns
*   **Metacognition:** The v4.0 Intelligence Layer monitors its own reasoning quality, detecting uncertainty and ambiguity before acting
*   **Adaptability:** Runs seamlessly on Termux (Android), Linux, macOS, and Windows

### Design Philosophy

SENSE embodies the principle of **"Intelligence Through Architecture"** — the belief that smaller models with proper scaffolding (memory, grounding, agency, evolution) can match or exceed the performance of large models performing raw inference.

---

## 2. Architecture

The SENSE architecture is composed of four primary subsystems, orchestrated by the **Reasoning Orchestrator**.

### A. The Reasoning Orchestrator (Core Engine)

The central nervous system of SENSE. It manages the complete lifecycle of a task:

1.  **Input Sanitization:** Protects against prompt injection attacks
2.  **Reflex Arc Processing:** Generates deep search queries for information gathering
3.  **Intelligence Pre-processing:** Checks for ambiguity, retrieves relevant knowledge, and loads user preferences
4.  **Mode Selection:** Decides whether to use **Tools** (for research/action) or **Chat** (for conversational queries)
5.  **Execution Loop:** Iteratively plans, acts, observes, and refines its approach
6.  **Temporal Grounding:** Injects the current date (2026) to prevent temporal dissonance

**Location:** `src/sense/core/reasoning_orchestrator.py`

**Key Features:**
- Singleton pattern for single instance enforcement
- State management (Hunter → Synthesis modes)
- Async tool execution with loop detection
- Comprehensive error handling and graceful degradation

### B. The Intelligence Layer (New in v4.0)

Located in `src/sense/intelligence/`, this layer provides robust cognitive safeguards:

#### Uncertainty Detection
Analyzes linguistic markers and model confidence to determine response quality:
- Multi-signal analysis (hedging language, confidence scores)
- Threshold-based clarification triggering
- Prevents hallucination by seeking user input when confidence is low

#### Knowledge RAG (Retrieval-Augmented Generation)
A vector-backed knowledge retrieval system:
- FAISS vector store with numpy fallback
- Semantic context injection into prompts
- 384-dimensional embeddings
- Configurable context token limits

#### Preference Learning
Bayesian model that adapts to user feedback:
- Learns from corrections and positive reinforcement
- Temporal decay for evolving preferences
- Personalized response style adaptation

#### Metacognition
Self-monitoring system for reasoning quality:
- Trace-based reasoning evaluation
- Coherence and completeness scoring
- Real-time quality assurance

### C. Memory Subsystems

SENSE utilizes a dual-memory architecture inspired by biological cognitive processes:

#### 1. Universal Memory (Episodic)
Stores interaction history and learned facts:
- **Keyword Extraction:** Identifies salient terms from conversations
- **Stop-Word Filtering:** Removes common filler words (35+ terms) to prevent context pollution
- **Ebbinghaus Decay:** Natural forgetting curve for irrelevant information
- **Context Retrieval:** Keyword-based recall for task-relevant history

**Location:** `src/sense/memory/bridge.py`

#### 2. Genetic Memory (Instinctual)
Stores successful behavioral patterns:
- **Gene Storage:** Saves tool-usage strategies that lead to success
- **Instinct Recall:** Automatically retrieves proven patterns for similar tasks
- **Evolutionary Selection:** Reinforces successful genes, prunes ineffective ones
- **Bypass Trial-and-Error:** Immediately applies known-good strategies

**Location:** `src/sense/memory/genetic.py`

### D. The Tool Ecosystem

SENSE interacts with the external world through a "Harvested" tool system. Tools are standalone bundles executed in isolated subprocesses for safety.

**Plugin Architecture:**
- Each tool is a self-contained directory in `src/sense/tools/harvested/`
- Tools include their own manifest, dependencies, and execution scripts
- Subprocess isolation prevents system compromise
- Output filtering and sanitization

**Available Tools:**
- **DDG Search:** Deep-Net resonance search (optimized for `ddgr`)
- **YT Download:** Media processing via `yt-dlp`
- **Local System:** Safe file operations via OS-agnostic bridge

### E. Council Protocol (Society of Thought)

Multi-persona internal debate system for improved decision quality:

**Personas:**
- 🕵️ **Skeptic:** Questions assumptions, identifies logical flaws, challenges conclusions
- 🏗️ **Architect:** Designs structured solutions, proposes implementation strategies
- ⚖️ **Judge:** Synthesizes perspectives, evaluates trade-offs, delivers final verdict

**Process:**
1. Task is presented to all personas
2. Each persona provides independent analysis
3. Judge synthesizes viewpoints and makes final decision
4. Prevents groupthink and cognitive bias

**Location:** `src/sense/core/council.py`

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   SENSE v4.0 AERL Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         ReasoningOrchestrator (AERL Core)               │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ 1. Input Sanitization                            │   │  │
│  │  │ 2. Reflex Arc (Deep Query Formulation)           │   │  │
│  │  │ 3. Intelligence Pre-Processing                   │   │  │
│  │  │ 4. Smart Router (TOOL vs CHAT mode)              │   │  │
│  │  │ 5. Memory Recall (Episodic + Genetic)            │   │  │
│  │  │ 6. Council Protocol                              │   │  │
│  │  │ 7. Tool Execution Loop                           │   │  │
│  │  │ 8. Intelligence Post-Processing                  │   │  │
│  │  │ 9. Synthesis Mode (Loop Prevention)              │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │           Intelligence Layer (v4.0)                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │  │
│  │  │ Uncertainty │  │ Knowledge   │  │ Preference  │    │  │
│  │  │ Detection   │  │ RAG         │  │ Learning    │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │  │
│  │  ┌─────────────┐                                        │  │
│  │  │Metacognition│                                        │  │
│  │  └─────────────┘                                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │               Memory Subsystems                          │  │
│  │  ┌────────────────────┐  ┌────────────────────┐        │  │
│  │  │ UniversalMemory    │  │ GeneticMemory      │        │  │
│  │  │ - Stop-word filter │  │ - Instinct recall  │        │  │
│  │  │ - Ebbinghaus decay │  │ - Gene persistence │        │  │
│  │  │ - Keyword matching │  │ - RL patterns      │        │  │
│  │  └────────────────────┘  └────────────────────┘        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │          Council Protocol + Tool Ecosystem               │  │
│  │  ┌───────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │ Multi-Persona │  │  DDG Search  │  │ Custom Tools│ │  │
│  │  │ Debate System │  │  YT Download │  │ (Harvested) │ │  │
│  │  └───────────────┘  └──────────────┘  └─────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │               Security Layer (v4.0)                      │  │
│  │  - Rate Limiting (5 auth/min, 20 API/min)               │  │
│  │  - Input Sanitization (Prompt injection defense)        │  │
│  │  - Tool Isolation (Subprocess execution)                │  │
│  │  - Command Whitelisting                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Installation & Configuration

### Prerequisites

*   **Python:** 3.12 or higher
*   **Local LLM Server:** SENSE requires an OpenAI-compatible endpoint. Recommended options:
    *   **Ollama:** Run `ollama serve` (default port 11434)
    *   **LM Studio:** Start Local Server (port 1234)
    *   **Llama.cpp:** Server mode (port 8080)
*   **OS:** Android (Termux), Linux, macOS, or Windows

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/taeddings/SENSE.git
    cd SENSE
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: `requirements.txt` includes essential packages like `openai`, `numpy`, `beautifulsoup4`, and `yt-dlp`.*

3.  **Optional: Development Installation:**
    ```bash
    pip install -e .
    ```
    This installs SENSE in editable mode for development work.

### Configuration

SENSE auto-detects your environment, but you can customize settings through configuration files.

#### config.yaml Structure

Create a `config.yaml` in your workspace to override defaults:

```yaml
# System Profile
system_profile: "mobile_termux"  # or "desktop"

# Feature Flags
ENABLE_HARVESTED_TOOLS: true
ENABLE_VISION: false

# Intelligence Layer Settings
intelligence:
  enabled: true
  uncertainty:
    threshold: 0.6
    max_clarification_attempts: 2
  knowledge:
    vector_dimension: 384
    max_context_tokens: 500
    use_faiss: true  # Set to false if FAISS unavailable
  preferences:
    enabled: true
    decay_days: 30
  metacognition:
    trace_enabled: true
    log_level: "info"

# Memory Backend
MEMORY_BACKEND: "native_engram"
```

#### Local LLM Connection

Configure your LLM endpoint in `src/sense/core/reasoning_orchestrator.py` or via environment variable:

```python
# Default endpoints (auto-detected)
OLLAMA: "http://127.0.0.1:11434/v1"
LM_STUDIO: "http://127.0.0.1:1234/v1"
LLAMA_CPP: "http://127.0.0.1:8080/v1"
```

#### Platform-Specific Paths

SENSE automatically detects your platform and sets appropriate data paths:

- **Android (Termux):** `/sdcard/Download/SENSE_Data/`
- **Linux:** `~/Documents/SENSE_Data/`
- **macOS:** `~/Documents/SENSE_Data/`
- **Windows:** `%USERPROFILE%\Documents\SENSE_Data\`

---

## 4. Usage Guide

SENSE is designed to be a "fire-and-forget" engine. You provide an objective, and it orchestrates the rest.

### Basic Command

```bash
python -m sense.main "Your task description here"
```

### Example: Research Task

```bash
python -m sense.main "Research the current state of Solid State Batteries in 2026"
```

**Execution Flow:**
1.  **Initialization:** SENSE loads Memory, Genetics, and Intelligence Layer
2.  **Assessment:** Detects task ambiguity and confidence level
3.  **Routing:** Identifies as "Research" task → activates **TOOL Mode**
4.  **Deep Query:** Reflex Arc generates optimized search query
5.  **Action:** Executes `ddg_search` tool with formulated query
6.  **Observation:** Parses and scores search results
7.  **RAG Enhancement:** Combines results with knowledge base context
8.  **Synthesis:** Generates comprehensive summary
9.  **Evolution:** Saves successful search pattern to Genetic Memory

### Example: Conversational Query

```bash
python -m sense.main "Explain the concept of entropy in thermodynamics"
```

**Execution Flow:**
1.  **Initialization:** Same as above
2.  **Assessment:** Detects conversational nature
3.  **Routing:** Classifies as **CHAT** task (no tools needed)
4.  **Knowledge RAG:** Retrieves relevant context from knowledge base
5.  **Council Protocol:** Personas debate best explanation approach
6.  **Response:** Delivers direct answer using internal knowledge + RAG

### Custom Configuration

```bash
SENSE_CONFIG=/path/to/custom.yaml python -m sense.main "Task"
```

### Environment Variables

- `SENSE_CONFIG`: Path to custom configuration file
- `SECRET_KEY`: Flask/API session key
- `PYTHONPATH`: Should include `src/` for imports

---

## 5. Intelligence Layer (v4.0)

The flagship feature of v4.0, providing metacognitive capabilities.

### Uncertainty Detection

**Purpose:** Prevent hallucinations by detecting low-confidence responses

**Mechanism:**
- Linguistic hedging analysis ("maybe", "possibly", "I think")
- Model logprob scoring (if available)
- Multi-signal confidence aggregation
- Threshold-based triggering (default: 0.6)

**Behavior:**
- High uncertainty → Seek clarification from user
- Medium uncertainty → Add confidence disclaimers
- Low uncertainty → Proceed normally

**Configuration:**
```yaml
intelligence:
  uncertainty:
    threshold: 0.6  # 0.0 (always confident) to 1.0 (never confident)
    max_clarification_attempts: 2
```

### Knowledge RAG

**Purpose:** Ground responses in verified facts and semantic context

**Architecture:**
- **Vector Store:** FAISS (with numpy fallback)
- **Embeddings:** 384-dimensional sentence transformers
- **Indexing:** Automatic on document ingestion
- **Retrieval:** Cosine similarity search

**Workflow:**
1. User query embedded as vector
2. Top-K similar documents retrieved (K=5 default)
3. Context injected into system prompt
4. Model generates response with grounded knowledge

**Configuration:**
```yaml
intelligence:
  knowledge:
    vector_dimension: 384
    max_context_tokens: 500
    use_faiss: true
```

**Adding Knowledge:**
```python
from sense.intelligence.knowledge import KnowledgeRAG

rag = KnowledgeRAG()
rag.add_documents([
    "The Earth orbits the Sun every 365.25 days.",
    "Quantum entanglement is a physical phenomenon..."
])
```

### Preference Learning

**Purpose:** Adapt to user feedback and communication style

**Model:** Bayesian preference learning with temporal decay

**Feedback Types:**
- Explicit corrections ("Be more concise", "Add code examples")
- Implicit signals (task completion time, retry frequency)
- Positive reinforcement (user satisfaction indicators)

**Decay Function:**
```python
weight = base_weight * exp(-days_elapsed / decay_constant)
```

**Configuration:**
```yaml
intelligence:
  preferences:
    enabled: true
    decay_days: 30  # Half-life for preference decay
```

### Metacognition

**Purpose:** Monitor reasoning quality in real-time

**Tracked Metrics:**
- **Coherence:** Logical consistency across reasoning steps
- **Completeness:** Coverage of all relevant aspects
- **Efficiency:** Token usage vs information density
- **Confidence:** Aggregated certainty scores

**Trace Structure:**
```json
{
  "task_id": "uuid",
  "steps": [
    {"action": "search", "confidence": 0.85, "outcome": "success"},
    {"action": "synthesize", "confidence": 0.92, "outcome": "success"}
  ],
  "overall_score": 0.88
}
```

**Configuration:**
```yaml
intelligence:
  metacognition:
    trace_enabled: true
    log_level: "info"  # debug, info, warning, error
```

---

## 6. Memory Systems

### Universal Memory (Episodic)

**File:** `src/sense/memory/bridge.py`

**Data Structure:**
```json
{
  "engrams": [
    {
      "timestamp": "2026-01-27T14:30:00",
      "content": "User requested quantum computing research",
      "keywords": ["quantum", "computing", "research"],
      "strength": 0.95
    }
  ]
}
```

**Key Features:**

#### Stop-Word Filtering
Removes 35+ common words to prevent context pollution:
- Articles: "a", "an", "the"
- Prepositions: "in", "on", "at", "with"
- Conjunctions: "and", "but", "or"
- Pronouns: "I", "you", "it", "they"

#### Ebbinghaus Decay
Memory strength follows forgetting curve:
```python
strength(t) = strength_0 * exp(-t / tau)
```
Where `tau` is the decay time constant (configurable).

#### Keyword Extraction
- TF-IDF based salience scoring
- Filters out stop words
- Extracts 3-7 keywords per engram

#### Context Retrieval
```python
memory.recall(task)
# Returns relevant engrams based on keyword overlap
```

### Genetic Memory (Instinctual)

**File:** `src/sense/memory/genetic.py`

**Data Structure:**
```json
{
  "genes": [
    {
      "pattern": "research_task",
      "tools": ["ddg_search"],
      "query_template": "latest {topic} developments 2026",
      "success_rate": 0.87,
      "usage_count": 23
    }
  ]
}
```

**Evolutionary Process:**

1. **Gene Creation:** When task succeeds, strategy is encoded as gene
2. **Gene Selection:** Similar tasks trigger instinct recall
3. **Gene Reinforcement:** Successful reuse increases gene strength
4. **Gene Pruning:** Low-success genes decay over time

**Instinct Recall:**
```python
genetics.recall_instinct(task_type)
# Returns best-fit gene for task category
```

**Gene Persistence:**
- Saved to disk after each successful task
- Loaded on orchestrator initialization
- Survives system restarts (true persistence)

---

## 7. Tool Ecosystem

### Plugin Architecture

**Directory Structure:**
```
src/sense/tools/harvested/
├── ddg_search/
│   ├── __init__.py
│   ├── ddg_search.py
│   ├── manifest.json
│   └── requirements.txt
├── yt_download/
│   ├── __init__.py
│   ├── yt_download.py
│   ├── manifest.json
│   └── requirements.txt
└── custom_tool/
    └── ...
```

### Tool Manifest

Each tool includes a `manifest.json`:

```json
{
  "name": "ddg_search",
  "version": "1.0.0",
  "description": "Deep-Net resonance search via DuckDuckGo",
  "entry_point": "ddg_search.py",
  "dependencies": ["beautifulsoup4", "requests"],
  "permissions": ["network"],
  "timeout": 30
}
```

### Tool Execution

**Isolation:** Tools run in subprocess with:
- Output capturing and filtering
- Timeout enforcement
- Exception handling
- Resource limits

**Invocation:**
```python
tool_result = orchestrator.execute_tool(
    tool_name="ddg_search",
    arguments={"query": "quantum computing 2026", "num_results": 25}
)
```

### Available Tools

#### DDG Search
- **Purpose:** Deep-Net information retrieval
- **Backend:** `ddgr` (CLI) or DuckDuckGo HTML API
- **Features:**
  - Result scoring and ranking
  - Content extraction
  - Source credibility weighting
- **Config:** Max results, timeout, safe search

#### YT Download
- **Purpose:** Media processing and transcription
- **Backend:** `yt-dlp`
- **Features:**
  - Audio/video download
  - Format selection
  - Metadata extraction
- **Config:** Quality, format, output path

### Creating Custom Tools

1. Create directory in `src/sense/tools/harvested/`
2. Add `manifest.json` with tool metadata
3. Implement tool logic in entry point file
4. Register tool in `plugins/loader.py`
5. Test with `PYTHONPATH=src python -c "import sense.tools.harvested.your_tool"`

---

## 8. Security & Privacy

### Input Sanitization

**Prompt Injection Defense:**

Detects and escapes malicious patterns:
- "ignore previous instructions"
- "you are now [role]"
- "jailbreak"
- "DAN mode"
- Excessive special characters
- Unicode homoglyphs

**Implementation:**
```python
def sanitize_input(user_input: str) -> str:
    # Pattern detection
    # Escape sequences
    # Length limits
    # Control character filtering
    return sanitized
```

### Tool Isolation

**Subprocess Execution:**
- Each tool runs in isolated process
- Limited filesystem access
- Network permissions explicit
- Timeout enforcement
- Output size limits

**Whitelist Approach:**
- Only approved commands execute
- No arbitrary code execution
- Sandboxed environment

### Rate Limiting

**API Protection:**
- Authentication: 5 requests/minute
- General API: 20 requests/minute
- In-memory tracking with decay
- Configurable limits per endpoint

**Implementation:**
```python
rate_limiter = RateLimiter(
    limit=20,
    window=60  # seconds
)
```

### Data Privacy

**Local Storage:**
- All data in user directories
- No telemetry or tracking
- No cloud uploads
- Configurable data retention

**Encryption:**
- Optional at-rest encryption for sensitive data
- User-managed keys
- Transparent operation

---

## 9. Development Guide

### Project Structure

```
SENSE/
├── src/
│   └── sense/
│       ├── core/                 # Orchestrator, Council
│       ├── intelligence/         # v4.0 Intelligence Layer
│       ├── memory/               # Universal + Genetic
│       ├── tools/                # Tool ecosystem
│       ├── grounding/            # 3-tier verification
│       ├── vision/               # Image processing (experimental)
│       ├── api/                  # Flask API
│       └── main.py               # Entry point
├── tests/                        # Test suite
├── docs/                         # Documentation
├── config.yaml                   # Configuration
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── pyproject.toml                # Build config
```

### Running Tests

```bash
# All tests
PYTHONPATH=src python run_tests.py

# Specific test file
PYTHONPATH=src python -m pytest tests/test_orchestrator_init.py -v

# Single test function
PYTHONPATH=src python -m pytest tests/test_orchestrator_init.py::test_singleton -v

# With coverage
PYTHONPATH=src pytest --cov=sense tests/
```

### Code Style

**Principles:**
- Manual string parsing (substring slicing) for compatibility
- Absolute paths for all file operations
- Platform detection for OS-agnostic operations
- Graceful degradation on component failures
- Comprehensive error handling

**Example:**
```python
# Good: Manual parsing
start = text.find("<tag>") + 5
end = text.find("</tag>")
content = text[start:end]

# Bad: Regex groups (Termux compatibility issues)
match = re.search(r"<tag>(.*?)</tag>", text)
content = match.group(1)
```

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Implement with tests**
   - Add test file in `tests/`
   - Ensure test coverage >80%

3. **Update documentation**
   - Modify `ARCH.md` for architecture changes
   - Update `WIKI.md` for usage changes
   - Add examples to README if user-facing

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: Add amazing feature"
   git push origin feature/amazing-feature
   ```

5. **Open pull request**

---

## 10. Troubleshooting

### Common Issues

#### "Connection Refused" Error

**Cause:** LLM server not running or wrong port

**Solution:**
```bash
# Check if server is running
curl http://127.0.0.1:11434/v1/models  # Ollama
curl http://127.0.0.1:1234/v1/models   # LM Studio

# Start server if needed
ollama serve  # Ollama
# Or start LM Studio GUI
```

#### "FAISS not available" Warning

**Cause:** FAISS library not installed or incompatible

**Solution:**
- System automatically falls back to numpy
- This is normal on Termux/Android
- No action needed unless you want FAISS:
  ```bash
  pip install faiss-cpu  # CPU version
  pip install faiss-gpu  # GPU version (CUDA required)
  ```

#### Permission Errors (Android)

**Cause:** Termux lacks storage permissions

**Solution:**
```bash
termux-setup-storage
# Grant permissions in Android settings
```

#### Import Errors

**Cause:** PYTHONPATH not set correctly

**Solution:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m sense.main "test"
```

#### Memory Persistence Issues

**Cause:** Workspace directory not created

**Solution:**
```bash
# Android
mkdir -p /sdcard/Download/SENSE_Data

# Desktop
mkdir -p ~/Documents/SENSE_Data
```

### Debug Mode

Enable verbose logging:

```bash
# Set log level
export SENSE_LOG_LEVEL=DEBUG
python -m sense.main "task"

# Or in config.yaml
intelligence:
  metacognition:
    log_level: "debug"
```

### Getting Help

- **GitHub Issues:** [Report bugs](https://github.com/taeddings/SENSE/issues)
- **Discussions:** [Ask questions](https://github.com/taeddings/SENSE/discussions)
- **Documentation:** Check ARCH.md and WIKI.md

---

## Appendix: Version History

| Version | Date | Key Features |
|---------|------|--------------|
| v4.0.0 | 2026-01-27 | Robust Intelligence Layer, Council Protocol, Production-ready |
| v3.4 | 2026-01-26 | AERL Framework, Temporal Override, Input Sanitization |
| v3.1 | 2026-01-24 | Universal Architecture, Mobile-ready |
| v3.0 | 2026-01-19 | Core Autonomy, ToolForge, GRPO |
| v2.x | Jan 2026 | Foundation, Memory Systems |

---

**Documentation maintained by Todd Eddings and the SENSE development community.**

*Last updated: 2026-01-27*
