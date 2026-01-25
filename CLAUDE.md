# CLAUDE.md — SENSE v3.0+ Development Protocol

**Repository:** `taeddings/SENSE`
**Design Rationale:** `RATIONALE.md`
**Last Updated:** 2026-01-24
**Status:** v3.0 Complete + Comprehensive Integration → v4.0-v9.0 Roadmap Defined
**Author:** Todd Eddings

---

## HARD BOUNDARIES — READ FIRST

1.  **No "Context Stuffing":** Always use `AgeMem` retrieval, do not rely on infinite context windows.
2.  **No Unverified Actions:** Every major state change must pass Tier 3 (Experiential) Grounding.
3.  **No Direct Subprocesses:** ALL OS interactions must go through the `Bridge`.
4.  **No Hardcoded Models:** Use `config.model` for all inference.

---

## Architecture Overview

### Layer 1: The Brain (Reasoning)
**Location:** `sense/core/`
- `reasoning_orchestrator.py`: Manages the Architect/Worker/Critic phases.
- `evolution/`: Genetic algorithms and curriculum.
- `plugins/forge.py`: **NEW** Dynamic tool creation engine.

### Layer 2: The Bridge (Agency)
**Location:** `sense/bridge/`
- Translate intent (`install numpy`) to OS command (`pip install numpy`).
- Enforce Safety/Whitelists.

### Layer 3: Grounding (Truth)
**Location:** `sense/grounding/`
- `synthetic.py`, `realworld.py`, `experiential.py`.
- **Reflexion:** The Critic uses these to validate Worker output.

### Layer 4: Interface (Personas)
**Location:** `sense/interface/personas/`
- `architect.md`: Planning & Decomposition.
- `worker.md`: Code Execution (from Agent-Zero).
- `critic.md`: Review & Verification.

### Layer 5: Evolution & Memory
**Location:** `sense/core/evolution/`, `sense/core/memory/`
- `curriculum.py`: Adaptive task generation.
- `grpo.py`: Evolutionary optimization.
- `ltm.py`: AgeMem procedural RAG.

### Layer 6: Agency & Safety
**Location:** `sense/bridge/`, `sense/llm/`
- `bridge.py`: Safe OS interactions with EmergencyStop.
- `model_backend.py`: Multi-provider LLM support.

### Layer 7: Deployment
**Location:** `sense/dashboard.py`, `sense/api.py`, `Dockerfile`
- Streamlit dashboard, FastAPI server, Docker deployment.

---

## Core Components Specification

### 1. The Tool Forge (Dynamic Tool Creation)
**File:** `sense/core/plugins/forge.py`

```python
class ToolForge:
    """Monitors execution history to crystallize reusable skills."""

    REPETITION_THRESHOLD: int = 3

    def scan_memory(self, memory: AgeMem) -> List[CandidateSkill]:
        """Finds repeated successful code patterns in LTM."""
        ...

    def forge_tool(self, candidate: CandidateSkill) -> ProposedPlugin:
        """Refactors raw script into a standardized PluginABC class."""
        ...

    def verify_tool(self, plugin: ProposedPlugin) -> bool:
        """
        CRITICAL: Generates and runs Tier 1 synthetic tests.
        Only saves if 100% pass.
        """
        ...

    def install_tool(self, plugin: ProposedPlugin) -> str:
        """Writes to sense/plugins/user_defined/ and hot-reloads Registry."""
        ...
```

### 2. Reasoning Orchestrator (The Phased Loop)
**File:** `sense/core/reasoning_orchestrator.py`

```python
class ReasoningOrchestrator:
    def solve_task(self, task: str):
        # Phase 1: Architect
        plan = self.prompt(persona="architect", input=task)

        # Phase 2: Worker
        result = self.prompt(persona="worker", input=plan, tools=self.registry)

        # Phase 3: Critic (Reflexion)
        verification = self.grounding.verify(result)
        if not verification.passed:
             return self.retry(plan, verification.feedback)

        # Phase 4: Integration (Tool Forge)
        self.tool_forge.check_for_crystallization(result)
        return result
```

### 3. AgeMem (Procedural RAG)
**File:** `sense/core/memory/ltm.py`
- **Input:** Not just text, but ReasoningTrace objects (Goal -> Plan -> Code -> Result).
- **Retrieval:** When Goal is similar, retrieve the successful Code from the past.
- **Indexing:** Use FAISS for semantic similarity of Goals.

---

## Development Roadmap

### Phase 1: Foundation (Completed)
- [x] ReasoningGenome & Evolution Core
- [x] AgeMem Basic Structure (STM/LTM)
- [x] Bridge Driver Interface
- [x] Context Engineering: AdaptiveReasoningBudget

### Phase 2: Reasoning & Agency (Completed)
- [x] Reflexion Loop: Implemented ReasoningOrchestrator phases (Architect/Worker/Critic).
- [x] The Tool Forge: Implemented `sense/core/plugins/forge.py`.
- [x] Tool Persistence: PluginManager hot-loads user scripts.
- [x] Three-Tier Grounding: Connected grounding/ modules to the Orchestrator.

### Phase 3: Self-Evolution & Production (Completed)
- [x] Curriculum Agent: Adaptive task generation.
- [x] GRPO Trainer: Evolutionary optimization.
- [x] AgeMem RAG: Procedural memory retrieval.
- [x] Bridge: Safe OS interactions.
- [x] ModelBackend: Multi-LLM support.
- [x] Dashboard: Streamlit UI.
- [x] API Server: FastAPI endpoints.
- [x] Docker: Deployment with GPU.

### Phase 4: Comprehensive System Integration (v3.0 → v3.0+) — COMPLETED 2026-01-24
- [x] **Wire ToolForge**: Replaced ToolForgeStub with real implementation
- [x] **Wire UnifiedGrounding**: Connected Tier1/2/3 grounding to verification methods
- [x] **Wire Worker to Bridge**: Commands now route through safe execution layer
- [x] **Enable GRPO**: Evolution loop active with genome fitness tracking
- [x] **Autonomous Runner**: Created `autonomous.py` with 3 modes (continuous/single/evolve)
- [x] **Complete Dashboard**: Added Evolution, Bridge, Logs, Stats tabs (9 total)
- [x] **Config Propagation**: Unified configuration flow through all components
- [x] **CLI Entry Points**: Added `sense`, `sense-dashboard`, `sense-api` commands

**Key Improvements:**
- Full end-to-end integration: All components now communicate properly
- Autonomous mode: Can run continuously with self-evolution
- Safe execution: All OS commands go through Bridge whitelist
- Real grounding: Verification uses actual Tier 1/2/3 implementations
- Evolution active: Population evolves based on task performance
- Intuitive usage: Simple CLI commands for all modes

---

## Future Roadmap: v4.0 → v9.0 (18-Month Plan)

### v4.0: Human Alignment & Knowledge (Months 1-3)
**Enhancement #8: Human-in-the-Loop Alignment**
- Uncertainty detection (low confidence, ambiguity, novel tasks)
- Feedback collection (CLI, Streamlit, API interfaces)
- Preference learning (Bayesian updating, pattern detection)
- Integration with Orchestrator for clarification requests

**Enhancement #4: Internet-Scale Knowledge Integration**
- Multi-source web search (Google, ArXiv, StackOverflow)
- Retrieval-Augmented Generation (RAG) with FAISS
- Fact-checking (cross-reference claims against multiple sources)
- Knowledge caching with vector similarity search

### v5.0: Tool Ecosystem & Discovery (Months 4-6)
**Enhancement #5: Tool Discovery & Auto-Integration**
- Discovery engine (PyPI, GitHub, RapidAPI search)
- Documentation parser (Sphinx, docstrings, introspection)
- Wrapper generator (auto-generate PluginABC wrappers)
- Sandbox testing (security scanning, safe execution)
- Hot-loading (automatic plugin integration)

**Enhancement #9: Plugin Marketplace**
- Marketplace client (browse, search, install plugins)
- Publisher (auto-publish forged tools)
- Reputation system (ratings, reviews, downloads)
- Security scanner (malware detection, code analysis)

### v6.0: Meta-Learning & Curriculum (Months 7-9)
**Enhancement #2: Meta-Learning Curriculum**
- Curriculum genome (evolves task generation strategies)
- Task generator (templates + LLM creative generation)
- Difficulty estimator (ML-based prediction, ZPD optimization)
- Learning trajectory tracker (plateau detection, visualization)
- Skill graph (dependency tracking, mastery levels)

### v7.0: World Model & Memory (Months 10-12)
**Enhancement #6: Persistent World Model**
- Knowledge graph (facts, beliefs, relationships)
- Temporal reasoning (event tracking, causality)
- Belief updating (probabilistic, contradiction resolution)
- Entity tracking (persistent object identities)

**Enhancement #11: Attention & Working Memory**
- Attention buffer (Miller's Law: 7±2 items)
- Context switching (state preservation)
- Priority allocation (importance-based compute)
- Memory consolidation (STM → LTM transfer)

### v8.0: Embodied Grounding (Months 13-15)
**Enhancement #7: Embodied Simulation Grounding**
- Physics simulation (PyBullet/MuJoCo integration)
- Plan validation (verify physical feasibility)
- Intuitive physics learning (object permanence, gravity)
- Spatial reasoning (3D environment understanding)

### v9.0: Self-Modification & Introspection (Months 16-18)
**Enhancement #1: Self-Modifying Architecture**
- Code generation (improve own orchestrator)
- A/B testing (validate improvements)
- Hot-swap (safe architecture updates)
- Performance profiling (bottleneck detection)

**Enhancement #12: Introspection & Self-Awareness**
- Reasoning traces (decision explanation)
- Error detection (circular logic, contradictions)
- Meta-critic (validate reasoning quality)
- Confidence calibration (accurate uncertainty)

**Deferred to Future:**
- Enhancement #3: Swarm Intelligence (multi-agent systems)
- Enhancement #10: Federated Learning (distributed knowledge)

---

## Safety Protocol

### Command Whitelist
```python
COMMAND_WHITELIST = ["ls", "cat", "echo", "pwd", "python", "pip", "git", "grep", "curl"]
FORBIDDEN_PATTERNS = [r"rm\s+-rf\s+/", r"mkfs", r"chmod\s+777", r">\s*/dev/"]
```

### Emergency Stop
Available at every layer. If `EmergencyStop.check()` is True, immediately raise generic exception and halt.

---

## Testing Protocol

### Before ANY Commit
```bash
python -m pytest tests/ -v
python -m mypy sense/ --ignore-missing-imports
```

### New Tests Required
- `test_tool_forge.py`: Verify a mock script can be converted to a Plugin and loaded.
- `test_reflexion.py`: Verify the Critic rejects a bad output from the Worker.
