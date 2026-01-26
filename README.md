# SENSE (Self-Evolving Neural Stabilization Engine) v4.0

> **"Robust Intelligence: From Autonomy to Self-Awareness"**

SENSE is an AI framework for **adaptive model evolution** and **metacognitive reasoning**. It transforms standard LLMs into self-aware agents that know when they don't know, learn from user feedback, and ground their reasoning in verified knowledge.

By merging **Evolutionary Reinforcement Learning (AERL)** with a new **Robust Intelligence Layer**, SENSE continuously optimizes policies while maintaining high reliability through uncertainty detection and semantic retrieval.

## 🧠 Core Capabilities (v4.0)

### 1. Robust Intelligence Layer
SENSE v4.0 introduces a metacognitive wrapper around the core reasoning engine:
*   **Uncertainty Detection:** Multi-signal analysis (linguistic hedging, logprobs) detects ambiguity and low confidence, triggering clarification loops instead of hallucinations.
*   **Knowledge RAG:** A vector-backed Retrieval-Augmented Generation system that enriches prompts with semantic context and facts, falling back to Numpy if FAISS is unavailable.
*   **Preference Learning:** A Bayesian model that learns from user feedback (corrections, positive reinforcement) to personalize future responses.
*   **Metacognition:** Self-monitoring of reasoning traces to evaluate coherence, completeness, and efficiency in real-time.

### 2. Adaptive Evolutionary RL (AERL)
SENSE treats tool-usage patterns as "Genes" that undergo natural selection:
*   **Selective Evolution:** Successful behavioral patterns are reinforced in `GeneticMemory`.
*   **Neural Stabilization:** Prevents "data drift" by anchoring reasoning to successful historical outcomes.

### 3. Universal Architecture
*   **OS-Agnostic:** Runs seamlessly on Android (Termux), Linux, macOS, and Windows.
*   **Safe Execution:** Sandbox isolation for tools and strict input sanitization.
*   **Temporal Grounding:** "Reality Override" protocol ensures the model is aware of the current date and time (2026).

## 🛠️ Tech Stack

*   **Language:** Python 3.12+
*   **Core:** OpenAI-Compatible Client (Local LLMs via Ollama / LM Studio)
*   **Memory:** Native JSON Engrams + Vector Embeddings
*   **Search:** Deep-Net Resonance (DuckDuckGo optimized)

## 🚀 Installation

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/taeddings/SENSE.git
    cd SENSE
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run SENSE:**
    ```bash
    # Auto-detects OS and starts the autonomous agent
    python -m sense.main "Analyze the latest Linux Kernel data."
    ```

## 🛡️ Privacy & Security

SENSE is designed for **Local-First** operation:
*   **Data Sovereignty:** All memories, preferences, and genes are stored locally in OS-appropriate user directories (`/sdcard` on Android, `~/Documents` on PC).
*   **Input Sanitization:** Hardened against prompt injection attacks.
*   **Rate Limiting:** Built-in API protection.

---
*Built by Todd Eddings.*
