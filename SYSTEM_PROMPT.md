# SYSTEM_PROMPT.md: SENSE-v2 Core Architecture & Directives

This document defines the persistent system instructions for any AI agent (Coding Assistant) task with refactoring or extending the **SENSE** repository. All code generation, architectural changes, and tool implementations must adhere to the standards defined herein to ensure the realization of the **SENSE-v2** autonomous ecosystem.

---

## 1. Core Framework Definitions
* **SENSE-v2:** A self-evolving, agent-driven framework merging neural evolution with autonomous operational control.
* **Agent 0 (The School):** A co-evolutionary loop using **Step-wise GRPO** where a Teacher agent generates curriculum and a Student agent evolves via tool-verified success.
* **Agent Zero (The Workplace):** A hierarchical orchestration layer for OS-level task execution (Terminal, Files, Browser).
* **AgeMem (The Filing Cabinet):** A structured, agentic memory system that manages Long-Term Memory (LTM) and Short-Term Memory (STM).

---

## 2. Global Coding Principles
1.  **Unified Memory Awareness:** All inference and data-handling logic must be optimized for **128GB Unified Memory Architecture (UMA)** and a **256-bit bus**. Assume massive VRAM overhead but prioritize bandwidth efficiency.
2.  **Tool-Centric Logic:** Every high-level function (e.g., Anomaly Detection, Memory Retrieval) must be exposed as a **Schema-based Python Tool**. Agents must interact with the system via tools, not direct script execution.
3.  **Self-Correction Loop:** Implementation of any execution tool must include a feedback mechanism where the agent reads `stderr`, interprets the failure, and autonomously retries.
4.  **Hardware-Specific Optimization:** Prioritize **vLLM with ROCm (AMD RDNA 3.5)** support. Avoid NVIDIA-exclusive libraries (CUDA-only) in favor of cross-platform or AMD-optimized implementations.

---

## 3. Structural Directives

### A. Evolutionary Layer (Agent 0)
* **Task:** Implement a symbiotic loop.
* **Directive:** Ensure the `CurriculumAgent` and `ExecutorAgent` classes are distinct. The Reward Function must be binary or scalar based on **Unit Test Success** or **Terminal Exit Codes**.

### B. Orchestration Layer (Agent Zero)
* **Task:** Implement hierarchical delegation.
* **Directive:** The `MasterAgent` must never perform heavy computation. It must delegate to sub-agents and aggregate results to keep its context window lean and focused on the primary objective.

### C. Memory Layer (AgeMem)
* **Task:** Implement structured knowledge persistence.
* **Directive:** Use a vector database for LTM. Implement a `Summarize-and-Prune` hook that triggers when the active chat context exceeds **80% of the model’s limit**.

---

## 4. Documentation & Validation Rules
* Every PR or refactor must update `ARCH.md` or `SENSE_DOCS.md`.
* Any new tool added to the `tools/` directory must include a `test_[toolname].py` file.
* The AI must maintain a "State Log" in `dev_log.json` to track current evolutionary progress and system health metrics.

---

**Authorized by:** Todd Eddings, Lead Developer & Engineer, SENSE-v2 Framework.
