# CLAUDE.md — SENSE v6.0 Development Protocol

**Repository:** `taeddings/SENSE`
**Design Rationale:** `RATIONALE.md`
**Last Updated:** 2026-01-24
**Status:** v6.0 Complete (Meta-Learning Active)
**Author:** Todd Eddings

---

## Overview
SENSE is a self-evolving autonomous agent architecture designed to improve itself through recursive iterations. It combines LLM reasoning, evolutionary algorithms (GRPO), and dynamic tool creation.

**Core Principles:**
1.  **Architecture > Scale:** Intelligence emerges from the loop, not just the model size.
2.  **Self-Correction:** Every action is verified (Critic) and refined.
3.  **Evolution:** The system code and prompts evolve based on fitness (task success).
4.  **Autonomy:** Capable of continuous operation without human intervention.

---

## Architecture (v6.0)

### 1. The Brain (Reasoning)
*   **Orchestrator:** `src/sense/core/reasoning_orchestrator.py`
    *   **Phases:** Alignment → Discovery → Architect → Worker → Critic → Integration
*   **Knowledge:** `src/sense/knowledge/` (RAG, Web Search)
*   **Alignment:** `src/sense/alignment/` (Uncertainty Detection, Feedback)

### 2. The Body (Action & Tools)
*   **ToolForge:** `src/sense/core/plugins/forge.py` (Crystallizes repeated patterns)
*   **Tool Discovery:** `src/sense/tools/` (Finds & wraps external libraries)
*   **Bridge:** `src/sense/bridge/` (Safe OS execution)

### 3. The Evolution (Meta-Learning)
*   **MetaCurriculum:** `src/sense/meta_learning/` (Evolves task strategies)
*   **GRPO Trainer:** `src/sense/core/evolution/grpo.py` (Optimizes prompts)

---

## Usage

### CLI Modes
```bash
# Continuous self-evolution (Meta-Curriculum -> Tasks -> Execution -> Feedback)
sense --mode continuous

# Solve single task
sense --mode single --task "Analyze this dataset"

# Run evolution only
sense --mode evolve --generations 10
```

### Dashboard
```bash
sense-dashboard
```

### API
```bash
sense-api --port 8000
```

---

## Development Guidelines

1.  **Safety First:** All OS commands must go through `Bridge`.
2.  **Verify:** Use `src/sense/tests/` for verification.
3.  **Modular:** Keep components loosely coupled.
4.  **Documentation:** Update `IMPLEMENTATION_STATE.md` after every major change.

---

## Upcoming Phases
*   **v7.0:** World Model & Attention
*   **v8.0:** Embodied Grounding
*   **v9.0:** Self-Modification