# SENSE v2.3 Handoff Document

## Project Location
```
/data/data/com.termux/files/home/project/SENSE/SENSE/
```

## Current Status: Phase 2 Core COMPLETE

All major Phase 2 components are implemented and committed to `origin/main`.

---

## Completed Work

### 1. ReasoningOrchestrator (Reflexion Loop)
**File:** `sense/core/reasoning_orchestrator.py`

Implements Architect → Worker → Critic → Integration phases with:
- `solve_task(task: str)` async method
- UnifiedGrounding for three-tier verification
- Retry loop with feedback refinement (max 3 attempts)
- ToolForge crystallization check on success

### 2. ToolForge (Dynamic Tool Creation)
**File:** `sense/core/plugins/forge.py`

Pipeline: DETECT → ABSTRACT → VERIFY → PERSIST → REGISTER
- PatternMatcher (Jaccard similarity)
- CodeAbstractor (literal extraction, parameterization)
- SyntheticVerifier (syntax + execution testing)
- PluginGenerator (PluginABC code generation)

### 3. Personas
**Directory:** `sense/interface/personas/`
- `architect.md` - Planning & decomposition
- `worker.md` - Execution & tool usage
- `critic.md` - Verification & feedback

---

## Phase 3 Complete: Production-Ready

All Next Steps implemented and optimized:
- ✅ **Model Backend**: Integrated ModelBackend for OpenAI, Anthropic, Ollama, LM Studio, Transformers, vLLM.
- ✅ **Grounding Tiers**: Connected Tier1, Tier2, Tier3 with real APIs and caching.
- ✅ **Integration Testing**: 464 tests pass; full flow verified.
- ✅ **Self-Evolution**: Curriculum Agent, GRPO Trainer, AgeMem RAG.
- ✅ **Agency & Safety**: Bridge with EmergencyStop, whitelisted OS commands.
- ✅ **Deployment**: Dashboard (Streamlit), API (FastAPI), Docker (GPU support).

### v3.0 Features
- Multi-LLM backend with config switching.
- Self-evolution loop: Curriculum → GRPO → Orchestrator → Memory.
- Safe OS interactions via Bridge.
- Web UI and API for integration.
- Dockerized deployment.

Run: `python SENSE/sense/main.py` for self-evolution; `streamlit run SENSE/sense/dashboard.py` for UI.

---

## Key Files to Read First

1. `PROGRESS.md` - Full implementation status
2. `IMPLEMENTATION_STATE.md` - Current state + next actions
3. `CLAUDE.md` - Development protocol + specifications
4. `RATIONALE.md` - Architecture philosophy
5. `sense/core/reasoning_orchestrator.py` - Main orchestrator
6. `sense/core/plugins/forge.py` - ToolForge implementation

---

## Git State
- Branch: `main`
- Remote: `origin/main` (up to date)
- Last commit: `836ea40` - Phase 2 Engram integration complete
- Credit: Todd Eddings

---

## Commands to Verify

```bash
cd /data/data/com.termux/files/home/project/SENSE/SENSE

# Check imports work
python -c "from sense.core import ReasoningOrchestrator, ToolForge; print('OK')"

# Run tests
python -m pytest tests/ -v

# Type check
python -m mypy sense/ --ignore-missing-imports
```

---

## Dependencies

```bash
pip install deap RestrictedPython psutil numpy pyyaml
pip install faiss-cpu sentence-transformers requests beautifulsoup4  # optional
```
