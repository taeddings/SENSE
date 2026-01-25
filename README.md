# 🧠 SENSE - Self-Evolving Network & Semantic Engine

**Version:** 3.1.0 (Universal Architecture)
**Author:** Todd Eddings
**Status:** Production-Ready (Mobile/Desktop/Cloud)
**Last Updated:** 2026-01-24

---

## Overview

SENSE is a **Universal AI Architecture** designed to run anywhere. It transforms any LLM—from a 1.2B model on an Android phone to GPT-4 in the cloud—into a grounded, autonomous agent.

**Key Features:**
- 📱 **Universal:** Runs on Termux (Android), Linux, Mac, and Windows without code changes.
- 🔌 **Hot-Swappable:** Switch between Local LLMs and Cloud APIs instantly.
- 🛡️ **Safe:** Strict CLI isolation for tools and "Bridge" for OS commands.
- 🧠 **Adaptive:** Reasoning engine adapts to the model's intelligence (Smart Router + Polyglot Parsing).
- 🧬 **Self-Evolving:** Uses genetic algorithms (GRPO) to improve over time.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/taeddings/SENSE.git
cd SENSE

# Install dependencies
pip install -e .

# Verify installation
python -m sense.main --task "Hello SENSE"
```

### Basic Usage

```bash
# Solve a single task using the active profile
python -m sense.main --task "Download the video https://example.com/video"

# Override LLM settings on the fly
python -m sense.main --url http://192.168.1.5:1234/v1 --model "llama-3-70b" --task "Explain quantum physics"
```

---

## 🏗️ Universal Architecture

SENSE v3.1 introduces a unified architecture with "Bridges" that adapt to the host environment:

*   **Universal Memory Bridge:** Automatically uses high-performance FAISS vector storage on powerful machines, but falls back to lightweight Engram storage on mobile devices.
*   **Optic Nerve (Vision):** Lazy-loads heavy vision libraries (`torch`, `transformers`) only when needed, preventing RAM crashes on constrained devices.
*   **Tool Adapter:** Safely executes harvested tools (like `yt_download`) in isolated subprocesses, filtering noise for smaller models.

---

## ⚙️ Configuration

SENSE uses `config.yaml` profiles to manage environments.

```yaml
# config.yaml
system_profile: "mobile_termux"

llm_profiles:
  termux_local:
    base_url: "http://127.0.0.1:8080/v1"
    model_name: "lfm2.5-1.2b-thinking"
  
  desktop_local:
    base_url: "http://192.168.1.5:1234/v1"
    model_name: "llama-3-70b"
  
  openai_cloud:
    provider: "openai"
    api_key: "${OPENAI_API_KEY}"
```

---

## 🗺️ Roadmap

### Completed (v3.1)
- ✅ **Universal Architecture:** Unified codebase for all platforms.
- ✅ **Subsystem Bridges:** Memory & Vision bridges for mobile stability.
- ✅ **Tool Harvesting:** Integration of Agent Zero instruments.
- ✅ **Adaptive Reasoning:** Smart Router & Polyglot Regex.

### Planned
- **v4.0:** Human Alignment & Knowledge
- **v5.0:** Tool Ecosystem (Marketplace)
- **v6.0:** Meta-Learning (Curriculum Evolution)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file.