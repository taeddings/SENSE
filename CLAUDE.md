# CLAUDE.md — SENSE v2.3 Development Protocol

**Repository:** `taeddings/SENSE`
**Design Rationale:** `RATIONALE.md`
**Last Updated:** 2026-01-23
**Status:** Active Development — Phase 2 (Reasoning & Agency)

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
- `critic.md`: **NEW** Review & Verification.

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

### Phase 2: Reasoning & Agency (Current Sprint)
- [ ] Reflexion Loop: Implement ReasoningOrchestrator phases (Architect/Worker/Critic).
- [ ] The Tool Forge: Implement `sense/core/plugins/forge.py`.
- [ ] Tool Persistence: Allow PluginManager to hot-load user scripts.
- [ ] Three-Tier Grounding: Connect grounding/ modules to the Orchestrator.

### Phase 3: Self-Evolution
- [ ] Curriculum Agent: Auto-generate coding challenges.
- [ ] Full GRPO: Optimize prompt fragments based on fitness.

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
