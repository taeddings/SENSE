# SENSE: Self-Evolving Neural Stabilization Engine

> **Version:** 4.0.0 (Robust Intelligence)  
> **Status:** Production-Ready  
> **Architecture:** AERL (Adaptive Evolutionary Reasoning Loop)

---

## 1. Overview

**SENSE** is not merely a framework; it is an autonomous, active **engine** designed to elevate static Large Language Models (LLMs) into self-aware, evolving agents. Unlike passive chatbots that reset after every session, SENSE possesses persistent memory, learns from its experiences, and actively stabilizes its reasoning against hallucinations.

It is built on the philosophy of **Local-First AI**, running entirely on your hardware (from Android phones to high-end workstations) without relying on cloud APIs for its core intelligence.

### Core Capabilities
*   **Autonomy:** Executes complex, multi-step tasks without human hand-holding.
*   **Evolution:** "Genetic Memory" actively selects and reinforces successful tool-usage patterns.
*   **Metacognition:** The v4.0 Intelligence Layer monitors its own reasoning quality, detecting uncertainty and ambiguity before acting.
*   **Adaptability:** Runs seamlessly on Termux (Android), Linux, macOS, and Windows.

---

## 2. Architectural Details

The SENSE architecture is composed of four primary subsystems, orchestrated by the **Reasoning Orchestrator**.

### A. The Reasoning Orchestrator (The Core)
The central nervous system of SENSE. It manages the lifecycle of a task:
1.  **Input Sanitization:** Protects against prompt injection attacks.
2.  **Intelligence Pre-processing:** Checks for ambiguity, retrieves relevant knowledge, and loads user preferences.
3.  **Mode Selection:** Decides whether to use **Tools** (for research/action) or **Chat** (for conversational queries).
4.  **Execution Loop:** Iteratively plans, acts, observes, and refines its approach.
5.  **Temporal Grounding:** Injects the current date (2026) to prevent "temporal dissonance" (confusing past/future events).

### B. The Intelligence Layer (New in v4.0)
Located in `src/sense/intelligence`, this layer provides robust cognitive safeguards:
*   **Uncertainty Detection:** Analyzes linguistic markers and logprobs to determine confidence. If confidence is low, it seeks clarification instead of guessing.
*   **Knowledge RAG:** A FAISS-backed vector store (with Numpy fallback) that retrieves semantic context to ground the agent's responses in fact.
*   **Preference Learning:** A Bayesian model that adapts to your feedback over time (e.g., "Be more concise", "Show code examples").
*   **Metacognition:** Monitors the logical coherence and completeness of the reasoning trace.

### C. Memory Subsystems
SENSE utilizes a dual-memory architecture to mimic biological cognitive processes:
1.  **Universal Memory (Episodic):**
    *   Stores interaction history and facts.
    *   Uses **Ebbinghaus Decay** to naturally "forget" irrelevant details over time.
    *   Implements **Stop-Word Filtering** to prevent context poisoning from common filler words.
2.  **Genetic Memory (Instinctual):**
    *   Stores "genes" (successful tool-use patterns).
    *   When a task succeeds, the strategy is saved.
    *   Future similar tasks trigger an "instinctive" recall of the successful strategy, bypassing trial-and-error.

### D. The Tool Ecosystem
SENSE interacts with the world through a "Harvested" tool system. Tools are standalone bundles executed in isolated subprocesses for safety.
*   **DDG Search:** Deep-Net resonance search (optimized for `ddgr`) to fetch high-fidelity information.
*   **YT Download:** Media processing via `yt-dlp`.
*   **Local System:** Safe file operations and system checks via the **Bridge** (an OS-agnostic abstraction layer).

---

## 3. Installation & Configuration

### Prerequisites
*   **Python:** 3.12 or higher.
*   **Local LLM Server:** SENSE requires an OpenAI-compatible endpoint. Recommended:
    *   **Ollama:** Run `ollama serve`
    *   **LM Studio:** Start Local Server (port 1234)
    *   **Llama.cpp:** Server mode
*   **OS:** Android (Termux), Linux, macOS, or Windows.

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

3.  **Configuration:**
    SENSE auto-detects your environment, but you can override settings in `src/sense/config.py` or by creating a `config.yaml`.
    
    **Essential Config (`src/sense/config.py`):**
    ```python
    # Example Intelligence Settings
    INTELLIGENCE_CONFIG = {
        "uncertainty": { "threshold": 0.6 },
        "knowledge": { "use_faiss": True }, # Set False if FAISS is not installed
        "preferences": { "enabled": True }
    }
    ```

    **Local LLM Connection:**
    Ensure your local server is running. SENSE defaults to `http://127.0.0.1:8080/v1` (common for Llama.cpp) or `http://127.0.0.1:1234/v1` (LM Studio) depending on your `config.local.py`.

---

## 4. Usage

SENSE is designed to be a "fire-and-forget" engine. You provide an objective, and it orchestrates the rest.

### Basic Command
To start the engine with a specific task:

```bash
python -m sense.main "Research the current state of Solid State Batteries in 2026."
```

### The Execution Flow
1.  **Initialization:** SENSE loads Memory, Genetics, and the Intelligence Layer.
2.  **Assessment:** It detects if the task is ambiguous. If so, it may ask for details (or log the ambiguity).
3.  **Routing:** It determines this is a "Research" task and activates **TOOL Mode**.
4.  **Action:** It generates a search query for `ddg_search`.
5.  **Observation:** It parses the search results.
6.  **Synthesis:** It uses RAG to combine search results with any existing knowledge.
7.  **Output:** It delivers a finalized summary.
8.  **Evolution:** If successful, the search pattern is saved to Genetic Memory.

### Interactive Mode (Chat)
For simple queries that don't require tools:

```bash
python -m sense.main "Explain the concept of entropy."
```
SENSE will detect this as a **CHAT** task and respond directly using its internal knowledge and RAG context.

---

## 5. Troubleshooting

*   **"Connection Refused":** Ensure your Local LLM server (Ollama/LM Studio) is running and the port in `src/sense/core/reasoning_orchestrator.py` (or config) matches.
*   **"FAISS not available":** The system will automatically fallback to Numpy. This is normal on some platforms (like Termux) where compiling FAISS is difficult.
*   **Permission Errors:** On Android, ensure Termux has storage permissions (`termux-setup-storage`).

---

*Documentation maintained by the SENSE Development Team.*
