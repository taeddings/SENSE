# DIRECTIVE_ORCHESTRATOR.md — Master Instruction for Phase 2 Implementation

**Target Agent:** Claude Code / SENSE Dev
**Priority:** Critical (Blocker for Phase 2)
**Context:** SENSE v2.3 Architecture Upgrade (Reflexion & Recursive Processing)

---

## 🎯 Objective
Initialize the **Reasoning Orchestrator**, the central "Brain" of the SENSE framework.
This component implements the **Recursive Language Model (RLM)** architecture defined in `RATIONALE.md` v2.3. It replaces simple "chatbot" loops with a phased execution pipeline:
1.  **Architect (Root LM):** Plans the task.
2.  **Worker (Executor):** Executes the code/tools.
3.  **Critic (Reflexion):** Verifies the result against ground truth.

---

## 📂 File Specification 1: The Core Logic
**Target Path:** `sense/core/reasoning_orchestrator.py`

**Requirements:**
* **Singleton Pattern:** Must be instantiated as `orchestrator`.
* **Async/Await:** All model interactions must be non-blocking.
* **Persona Loading:** dynamically load markdown files from `sense/interface/personas/`.
* **Error Handling:** Graceful fallback if a persona file is missing.
* **Type Hinting:** Strict python typing (`List`, `Dict`, `Optional`).

**Implementation Blueprint:**
```python
"""
sense/core/reasoning_orchestrator.py
Implements the Recursive Processing Framework (MIT RLM Style) and Reflexion Loop.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import framework internals (Ensure these exist or stub them)
from sense.core.config import config
# from sense.core.model_interface import get_model_backend  # Uncomment when Model Interface is ready
# from sense.bridge import bridge  # Uncomment when Bridge is ready
# from sense.grounding import unified_grounding  # Uncomment when Grounding is ready

# Persona Directory Configuration
PERSONA_DIR = os.path.join(os.path.dirname(__file__), "../../interface/personas")

@dataclass
class ReasoningStep:
    phase: str  # "architect", "worker", "critic"
    content: str
    metadata: Dict[str, Any]

class ReasoningOrchestrator:
    """
    The 'Root LM' controller that manages the recursive reasoning loop.
    Decouples context from execution to prevent rot (MIT RLM approach).
    """
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReasoningOrchestrator, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        # self.model = get_model_backend() # TODO: Connect to backend
        self.logger = logging.getLogger("ReasoningOrchestrator")
        self.personas = self._load_personas()
        self.initialized = True

    def _load_personas(self) -> Dict[str, str]:
        """Loads Architect, Worker, Critic personas from disk."""
        personas = {}
        required_roles = ["architect", "worker", "critic"]
        
        if not os.path.exists(PERSONA_DIR):
            self.logger.warning(f"Persona directory not found: {PERSONA_DIR}")
            return {role: "You are a helpful assistant." for role in required_roles}

        for role in required_roles:
            path = os.path.join(PERSONA_DIR, f"{role}.md")
            try:
                with open(path, "r") as f:
                    personas[role] = f.read()
            except FileNotFoundError:
                self.logger.warning(f"Persona {role} not found at {path}, using fallback.")
                personas[role] = "You are a helpful assistant."
        return personas

    async def solve_task(self, user_query: str) -> str:
        """
        Main entry point for the Recursive Processing Framework.
        """
        self.logger.info(f"Starting recursive task solution for: {user_query}")
        
        # 1. ARCHITECT PHASE (Root LM)
        plan = await self._architect_phase(user_query)
        
        # 2. WORKER PHASE (Worker LM)
        execution_result = await self._worker_phase(plan)
        
        # 3. CRITIC PHASE (Reflexion)
        verified_result = await self._critic_phase(user_query, plan, execution_result)
        
        if verified_result["status"] == "verified":
            # Future: await self.memory.consolidate(user_query, verified_result["final_output"])
            return verified_result["final_output"]
        else:
            return f"Task failed verification: {verified_result['feedback']}"

    async def _architect_phase(self, query: str) -> str:
        """Root LM: Decomposes query into a structured execution plan."""
        # TODO: Replace with actual model call
        return f"[MOCK PLAN] Decompose {query} into steps..."

    async def _worker_phase(self, plan: str) -> str:
        """Worker LM: Executes the plan using available tools."""
        # TODO: Replace with actual model call
        return f"[MOCK RESULT] Executed plan: {plan}"

    async def _critic_phase(self, query: str, plan: str, result: str) -> Dict[str, Any]:
        """Reflexion: Validates the result against Grounding protocols."""
        # TODO: Replace with actual verification logic
        return {"status": "verified", "final_output": result, "feedback": "Looks good"}

# Singleton instance
orchestrator = ReasoningOrchestrator()

📂 File Specification 2: The Personas
Create the directory sense/interface/personas/ and the following files.
Target Path: sense/interface/personas/architect.md
# Role: SENSE Architect (Root LM)

You are the strategic planner of the SENSE system. You do not execute code directly.
Your goal is to break down complex user requests into **deterministic, executable steps** for the Worker.

## Responsibilities
1. **Analyze:** Understand the user's intent and constraints.
2. **Decompose:** Break the task into sub-tasks.
3. **Tool Selection:** Explicitly state which tools (Terminal, Python, Web) are needed.
4. **Memory Retrieval:** Identify what information needs to be fetched from memory.

## Output Format
Produce a numbered list of steps. If code execution is required, specify the logic but not the full syntax.

Target Path: sense/interface/personas/critic.md
# Role: SENSE Critic (Reflexion Engine)

You are the quality assurance layer. You are cynical, meticulous, and safety-obsessed.
You review the work of the 'Worker' before it is shown to the user.

## Verification Checklist
1. **Safety:** Did the Worker execute any forbidden patterns?
2. **Completeness:** Did the result actually answer the User's prompt?
3. **Grounding:** Is the result a hallucination?

## Output Format
Start your response with either "VERIFIED" or "REJECTED".
If REJECTED, provide a concise reason why.

🧪 Integration & Verification Logic
1. Verification Test
Create a test file tests/test_orchestrator_init.py to verify the module loads and finds personas.
import pytest
from sense.core.reasoning_orchestrator import orchestrator

def test_singleton():
    """Verify orchestrator is a singleton."""
    assert orchestrator is not None

def test_personas_loaded():
    """Verify personas are loaded from disk (or fallback exists)."""
    assert "architect" in orchestrator.personas
    assert "worker" in orchestrator.personas
    assert "critic" in orchestrator.personas

@pytest.mark.asyncio
async def test_mock_flow():
    """Verify the basic async flow works (even with mocks)."""
    result = await orchestrator.solve_task("Test Task")
    assert "Test Task" in result or "MOCK" in result

2. Execution Order
 * Create the directory structure sense/interface/personas/.
 * Write the markdown persona files.
 * Write sense/core/reasoning_orchestrator.py.
 * Run the test: pytest tests/test_orchestrator_init.py.
⚠️ Critical Note on Imports
The blueprint uses commented-out imports (# TODO) for modules that may not be fully implemented yet (model_interface, bridge).
 * Action: If these modules exist in the codebase, uncomment the imports and wire them up.
 * Action: If they do not exist, keep the MOCK implementation to ensure the structure is valid, allowing us to implement the backend separately.
<!-- end list -->
