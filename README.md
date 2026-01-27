# SENSE (Self-Evolving Neural Stabilization Engine) v4.0

> **"Robust Intelligence: From Autonomy to Self-Awareness"**

[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](https://github.com/taeddings/SENSE/releases/tag/v4.0.0)
[![Python](https://img.shields.io/badge/python-3.12+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Android%20%7C%20Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#universal-architecture)

SENSE is an autonomous AI framework that transforms local LLMs into **self-aware agents** through architectural scaffolding. Unlike passive chatbots that reset after every session, SENSE possesses persistent memory, learns from experience, and actively stabilizes its reasoning against hallucinations.

**Core Philosophy:** *"Intelligence Through Architecture"* — Small models with proper scaffolding (memory, grounding, agency, evolution) can match or exceed large models doing raw inference.

---

## 🧠 What Makes SENSE Different

### 1. **Robust Intelligence Layer (v4.0)**
SENSE doesn't just respond—it *thinks about its thinking*:
- **Uncertainty Detection:** Knows when it doesn't know, triggering clarification instead of hallucination
- **Knowledge RAG:** Retrieves semantic context from vector-backed knowledge base (FAISS/numpy)
- **Preference Learning:** Adapts to your feedback style using Bayesian modeling
- **Metacognition:** Monitors reasoning quality in real-time, ensuring coherent, complete responses

### 2. **Council Protocol (Society of Thought)**
Multi-persona internal debate system:
- 🕵️ **Skeptic:** Critiques assumptions and identifies logical flaws
- 🏗️ **Architect:** Proposes structured solutions
- ⚖️ **Judge:** Synthesizes perspectives and delivers final verdict

This prevents groupthink and improves decision quality by simulating diverse viewpoints.

### 3. **Adaptive Evolutionary RL (AERL)**
Natural selection for tool-usage patterns:
- **Genetic Memory:** Successful strategies are "instinctually" recalled for similar tasks
- **Neural Stabilization:** Prevents model drift by anchoring to historical successes
- **Continuous Learning:** Each successful task strengthens the agent's behavioral repertoire

### 4. **Universal Architecture**
One codebase, all platforms:
- ✅ **Android (Termux)** — Full functionality on mobile devices
- ✅ **Linux** — Native performance on servers/workstations
- ✅ **macOS** — Seamless integration with Apple Silicon
- ✅ **Windows** — WSL and native Python support

### 5. **Local-First Privacy**
- **Data Sovereignty:** All memories and preferences stored locally
- **No Cloud Dependency:** Works with self-hosted LLMs (Ollama, LM Studio, llama.cpp)
- **Secure Execution:** Sandboxed tools with input sanitization

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/taeddings/SENSE.git
cd SENSE

# Install dependencies
pip install -r requirements.txt

# Optional: Install in editable mode for development
pip install -e .
```

### Prerequisites

1. **Python 3.12+** (required)
2. **Local LLM Server** (choose one):
   - [Ollama](https://ollama.ai/) — `ollama serve`
   - [LM Studio](https://lmstudio.ai/) — Start local server (port 1234)
   - [llama.cpp](https://github.com/ggerganov/llama.cpp) — Server mode

### Basic Usage

```bash
# Research task (activates tools automatically)
python -m sense.main "Research the latest developments in quantum computing"

# Conversational query (direct response)
python -m sense.main "Explain the concept of entropy"

# With custom configuration
SENSE_CONFIG=custom.yaml python -m sense.main "Your task here"
```

---

## 🛠️ Core Architecture

### The Reasoning Pipeline

```
User Input
    ↓
Input Sanitization (prompt injection defense)
    ↓
Reflex Arc (deep query formulation for search)
    ↓
Intelligence Pre-Processing
    ├─→ Uncertainty Detection (confidence analysis)
    ├─→ Knowledge RAG (semantic context injection)
    └─→ Preference Retrieval (user feedback history)
    ↓
Memory Retrieval
    ├─→ Genetic Memory (instinct recall)
    └─→ Episodic Memory (keyword matching)
    ↓
Council Protocol (multi-persona debate)
    ↓
Mode Decision (TOOL vs CHAT)
    ↓
Tool Execution Loop (max 5 iterations)
    ├─→ Hunter Mode (information gathering)
    └─→ Synthesis Mode (answer generation)
    ↓
Intelligence Post-Processing (quality assurance)
    ↓
Final Answer
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **ReasoningOrchestrator** | `src/sense/core/reasoning_orchestrator.py` | Main orchestration engine |
| **Intelligence Layer** | `src/sense/intelligence/` | Uncertainty, RAG, preferences, metacognition |
| **Council Protocol** | `src/sense/core/council.py` | Multi-persona debate system |
| **Memory Systems** | `src/sense/memory/` | Genetic (instincts) + Universal (episodic) |
| **Tool Ecosystem** | `src/sense/tools/harvested/` | Isolated plugin bundles (DDG search, YT download, etc.) |
| **Grounding Layers** | `src/sense/grounding/` | 3-tier verification (synthetic, real-world, experiential) |

---

## 🔬 Testing & Validation

```bash
# Run all tests
PYTHONPATH=src python run_tests.py

# Specific test suite
PYTHONPATH=src python -m pytest tests/test_orchestrator_init.py -v

# Quick import verification
PYTHONPATH=src python test_imports_quick.py

# v4.0 intelligence layer verification
PYTHONPATH=src python verify_sense.py
```

### Test Coverage

- ✅ **Component Initialization** — All subsystems load correctly
- ✅ **Data Flow Validation** — Complete pipeline integrity
- ✅ **Error Handling** — Graceful degradation on failures
- ✅ **Security** — Input sanitization and code fence escaping

---

## 🛡️ Security & Privacy

### Built-In Protections

- **Prompt Injection Defense:** Pattern detection and escaping
- **Command Whitelisting:** Only approved operations execute
- **Tool Isolation:** Subprocess execution with output filtering
- **Rate Limiting:** API protection (5 auth/min, 20 req/min)
- **Input Sanitization:** Length limits and control character filtering

### Data Storage

All data stored locally in OS-appropriate directories:
- **Android:** `/sdcard/Download/SENSE_Data/`
- **Desktop:** `~/Documents/SENSE_Data/`

No telemetry. No cloud uploads. Your data stays on your device.

---

## 📖 Documentation

- **[WIKI.md](WIKI.md)** — Comprehensive technical documentation
- **[ARCH.md](ARCH.md)** — Detailed architecture documentation
- **[CHANGELOG.md](CHANGELOG.md)** — Version history and updates

---

## 🗺️ Roadmap

| Version | Status | Key Features |
|---------|--------|--------------|
| **v4.0** | ✅ **Current** | Robust Intelligence Layer, Council Protocol |
| v4.1 | Planned | Grounding Runner integration, VisionInterface wiring |
| v5.0 | Q2 2026 | Swarm intelligence, multi-agent orchestration |
| v6.0 | Q3 2026 | Advanced tool synthesis, ToolForge v2.0 |
| v7.0 | Q4 2026 | Neural architecture search |

---

## 🤝 Contributing

SENSE is an open research project. Contributions welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes with descriptive messages
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Add tests for new features
- Update `ARCH.md` after significant changes
- Test on multiple platforms (especially Termux)
- Follow existing code patterns and architecture

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

**Architecture & Development:** Todd Eddings
**AI Collaboration:** Claude Sonnet 4.5 (Anthropic)
**Community:** Open-source contributors and testers

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/taeddings/SENSE/issues)
- **Discussions:** [GitHub Discussions](https://github.com/taeddings/SENSE/discussions)
- **Documentation:** [WIKI.md](WIKI.md)

---

**Built with ❤️ for the open-source AI community.**

*SENSE v4.0 — Transforming local LLMs into self-aware agents.*
