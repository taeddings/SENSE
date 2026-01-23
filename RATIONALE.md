# RATIONALE.md — SENSE v2.3 Design Rationale

**Companion to:** `CLAUDE.md`
**Purpose:** Explains the "why" behind design decisions

---

## Core Insight 1: Intelligence Through Architecture

### The Problem
Most approaches to "better AI" focus on bigger models (more parameters), creating a barrier to entry and diminishing returns.

### SENSE's Solution
**Wrap any model with an intelligence amplification layer.**
A small model with proper scaffolding (memory, grounding, agency, evolution) can match or exceed a large model doing raw inference.


```
┌────────────────────────────────────────────────────────────┐
│         SENSE Approach: Scale the Architecture             │
│   Any Model + Memory + Grounding + Agency + Evolution      │
└────────────────────────────────────────────────────────────┘
```

---

## Core Insight 2: The Tool Forge (Crystallizing Intelligence)

### The Problem: The "Context Trap"
Without a mechanism to save skills, SENSE is stuck in a loop of "re-invention."
- **Inefficiency:** It writes the same `check_gpu_temp.py` script 50 times.
- **Drift:** Variation in the script introduces random bugs.
- **Amnesia:** It forgets *how* it solved a complex problem once the session ends.

### The Solution
**Convert repeated successful behaviors into immutable tools.**
The "Tool Forge" acts as a compiler for agency. It monitors the memory stream for repetitive, successful code blocks and "crystallizes" them into permanent, optimized Python plugins.


```
┌─────────────────────────────────────────────────────────────┐
│                    THE TOOL FORGE PIPELINE                  │
│   1. DETECT    →  "I've written this logic 3x recently"     │
│   2. ABSTRACT  →  "Refactor specific values to parameters"  │
│   3. VERIFY    →  "Run Tier 1 Synthetic Tests on new tool"  │
│   4. PERSIST   →  "Save to local /plugins/user_defined/"    │
│   5. REGISTER  →  "Hot-load into ToolRegistry"              │
└─────────────────────────────────────────────────────────────┘
```

**Why This is "Evolution":**
- **Speed:** Calling a tool takes 1 turn. Writing/debugging takes 5+ turns.
- **Reliability:** Forged tools are deterministic and pre-verified.
- **Compound Growth:** SENSE builds a library of custom capabilities tailored to *your* environment.

---

## Model-Agnostic Design

### The Backend Abstraction
All SENSE logic talks to a universal `ModelBackend` interface, never to a specific model.
- **Supported:** LM Studio, Ollama, Transformers, vLLM.
- **Benefit:** Prevents lock-in and allows hardware-adaptive scaling.

---

## Memory Hierarchy: STM → LTM → Engram

### Three-Tier Solution
1.  **STM (Short-Term):** Working memory for active reasoning. Pruned via LRU.
2.  **LTM (Episodic):** Vector-indexed (FAISS). Persists across sessions.
3.  **Engram (Compressed):** Shadow-map compressed patterns. Informs "intuition" without explicit recall.

**Critical Upgrade:** `AgeMem` is not just a log; it is a **Procedural RAG system**. It indexes *successful workflows* so the agent can retrieve "How I installed PyTorch last time" before attempting it again.

---

## Three-Tier Grounding & Reflexion

### The "Critic" Loop (Reflexion)
Agents fail when they assume success. SENSE enforces a **Reflexion** step via the **Reasoning Orchestrator**.
- **Phase 1: Architect:** Plan the approach.
- **Phase 2: Worker:** Execute the code.
- **Phase 3: Critic:** Verify the result *before* reporting success.

### Grounding Tiers
1.  **Synthetic (Math/Logic):** Deterministic checks (e.g., `assert result == 42`).
2.  **Real-World (Web/Docs):** External fact verification.
3.  **Experiential (Agency):** "Did the file actually appear on disk?"

---

## Evolution Strategy

### What We Evolve
We evolve the **behavioral layer**, not just model weights:
- `thinking_patterns`: Prompt fragments.
- `reasoning_budget`: Tokens allocated per complexity.
- `tool_library`: The growing collection of Forged Tools.

### The Evolution Loop
1.  **Curriculum:** Generates/Selects a task.
2.  **Genome:** Attempts solution using current tools/patterns.
3.  **Grounding:** Verifies success.
4.  **Fitness:** Calculated (Success - Cost).
5.  **Forge:** If successful pattern repeated, crystallize into Tool.
6.  **Selection:** Evolve population parameters.

---

## Safety Design (Defense in Depth)

1.  **Command Whitelist:** Allowed binaries only.
2.  **Pattern Matching:** Block `rm -rf /` and similar.
3.  **AST Validation:** Prevent malicious Python injection.
4.  **RestrictedPython:** Sandbox runtime.
5.  **Emergency Stop:** Human/System override at any layer.

```python
# Checked in every execution loop
if EmergencyStop.check():
    raise EmergencyStopTriggered("Halted by safety system")
```
