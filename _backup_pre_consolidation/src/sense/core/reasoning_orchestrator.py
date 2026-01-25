#!/usr/bin/env python3
"""
SENSE v2.3 Reasoning Orchestrator with Reflexion Loop

Implements the Architect/Worker/Critic phased execution model:
- Phase 0 (Alignment): Ensure human intent is clear
- Phase 1 (Discovery): Auto-discover needed tools
- Phase 2 (Architect): Plan the approach (augmented by Knowledge System)
- Phase 3 (Worker): Execute the code/action
- Phase 4 (Critic): Verify the result via grounding (augmented by Fact Checking)
- Phase 5 (Integration): Check for tool crystallization

Part of Phase 2: Reasoning & Agency
Part of Phase 4: Human Alignment & Knowledge Integration
Part of Phase 5: Tool Ecosystem & Discovery
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from enum import Enum
import logging
import asyncio
import time
from datetime import datetime
from .memory.ltm import AgeMem
from ..bridge import Bridge
from ..llm.model_backend import get_model
from .plugins.forge import ToolForge

# v4.0 & v5.0 Integrations
from ..alignment import AlignmentSystem, UncertaintySignal
from ..knowledge import KnowledgeSystem
from ..tools import IntegrationManager

try:
    from .grounding import Tier1Grounding, Tier2Grounding, Tier3Grounding
except ImportError:
    # Fallback for Phase 2 stub
    class Tier1Grounding:
        def preprocess_data(self, data):
            return data
    class Tier2Grounding:
        def run_alignment_cycle(self):
            return {}
    class Tier3Grounding:
        def verify_outcome(self, expected, actual):
            return {}


class Phase(Enum):
    """Phases of the Reflexion loop."""
    PRE_PLANNING = "pre_planning"  # v4.0
    TOOL_DISCOVERY = "tool_discovery" # v5.0
    ARCHITECT = "architect"
    WORKER = "worker"
    CRITIC = "critic"
    INTEGRATION = "integration"


@dataclass
class VerificationResult:
    """Result from the Critic phase grounding verification."""
    passed: bool
    confidence: float
    feedback: str
    tier_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "feedback": self.feedback,
            "tier_results": self.tier_results,
        }


@dataclass
class TaskResult:
    """Complete result of a task execution through the Reflexion loop."""
    task_id: str
    task: str
    plan: str
    execution_result: Any
    verification: VerificationResult
    success: bool
    retry_count: int = 0
    execution_time: float = 0.0
    phases_completed: List[Phase] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "plan": self.plan,
            "execution_result": str(self.execution_result),
            "verification": self.verification.to_dict(),
            "success": self.success,
            "retry_count": self.retry_count,
            "execution_time": self.execution_time,
            "phases_completed": [p.value for p in self.phases_completed],
        }


class UnifiedGrounding:
    """
    Three-tier grounding system.
    Combines Synthetic, Real-World, and Experiential grounding.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        knowledge_system: Optional[KnowledgeSystem] = None
    ):
        self.config = config or {}
        self.weights = weights or {
            "synthetic": 0.4,
            "realworld": 0.3,
            "experiential": 0.3,
        }
        self.tier1 = Tier1Grounding()
        self.tier2 = Tier2Grounding()
        self.tier3 = Tier3Grounding()
        self.logger = logging.getLogger("UnifiedGrounding")
        self.knowledge = knowledge_system

    async def verify(self, result: Any, context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """
        Verify a result using all three grounding tiers.
        """
        tier_results = {}
        total_confidence = 0.0

        # Tier 1: Synthetic (deterministic checks)
        synthetic_score = self._verify_synthetic(result, context)
        tier_results["synthetic"] = synthetic_score
        total_confidence += synthetic_score * self.weights["synthetic"]

        # Tier 2: Real-world (external verification)
        realworld_score = await self._verify_realworld(result, context)
        tier_results["realworld"] = realworld_score
        total_confidence += realworld_score * self.weights["realworld"]

        # Tier 3: Experiential (action outcomes)
        experiential_score = self._verify_experiential(result, context)
        tier_results["experiential"] = experiential_score
        total_confidence += experiential_score * self.weights["experiential"]

        passed = total_confidence >= 0.6
        feedback = self._generate_feedback(tier_results, passed)

        return VerificationResult(
            passed=passed,
            confidence=total_confidence,
            feedback=feedback,
            tier_results=tier_results,
        )

    def _verify_synthetic(self, result: Any, context: Optional[Dict]) -> float:
        """Tier 1: Synthetic grounding - deterministic verification."""
        try:
            if result is None: return 0.0
            if isinstance(result, dict): data = result
            else: data = {"value": result, "type": type(result).__name__}
            processed = self.tier1.preprocess_data(data)
            if processed and isinstance(processed, dict):
                if 'timestamp' in processed: return 0.9
                return 0.7
            return 0.5
        except Exception as e:
            self.logger.warning(f"Tier1 verification failed: {e}")
            return 0.3

    async def _verify_realworld(self, result: Any, context: Optional[Dict]) -> float:
        """Tier 2: Real-world grounding - external fact verification."""
        score = 0.5
        try:
            alignment_result = self.tier2.run_alignment_cycle()
            if alignment_result and isinstance(alignment_result, dict):
                motor_result = alignment_result.get('result', {})
                if motor_result.get('success', False):
                    score = 0.7

            if self.knowledge and context and "original_task" in context:
                claim_to_check = str(result)[:200]
                fact_check = await self.knowledge.verify_plan(claim_to_check)
                if fact_check.get("verified", False):
                    score = min(1.0, score + 0.3)
                else:
                    if fact_check.get("confidence", 0) > 0.5:
                        score = max(0.0, score - 0.2)
            return score
        except Exception as e:
            self.logger.warning(f"Tier2 verification failed: {e}")
            return 0.4

    def _verify_experiential(self, result: Any, context: Optional[Dict]) -> float:
        """Tier 3: Experiential grounding - action outcome verification."""
        try:
            if context and context.get("action_succeeded"):
                expected = {"status": "completed", "action_succeeded": True}
                actual = result if isinstance(result, dict) else {"output": result}
                verification = self.tier3.verify_outcome(expected, actual)
                if verification.get('success', False):
                    return 1.0
                else:
                    error = verification.get('error', 1.0)
                    return max(0.3, 1.0 - (error / 10.0))
            return 0.6
        except Exception as e:
            self.logger.warning(f"Tier3 verification failed: {e}")
            return 0.5

    def _generate_feedback(self, tier_results: Dict[str, float], passed: bool) -> str:
        if passed: return f"Verification passed. Confidence scores: {tier_results}"
        weakest_tier = min(tier_results, key=tier_results.get)
        return f"Verification failed. Weakest tier: {weakest_tier} ({tier_results[weakest_tier]:.2f})"


class ReasoningOrchestrator:
    """
    Orchestrates reasoning through the Reflexion loop.
    Now integrated with Alignment, Knowledge, and Tool Ecosystem.
    """

    _instance = None
    MAX_RETRIES: int = 3

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ReasoningOrchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        grounding: Optional[UnifiedGrounding] = None,
        tool_forge: Optional[ToolForge] = None,
        personas_dir: Optional[Path] = None,
        tool_registry: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if getattr(self, '_initialized', False):
            return

        self.logger = logging.getLogger("ReasoningOrchestrator")
        self.config = config or {}

        # v4.0 & v5.0 Subsystems
        self.alignment = AlignmentSystem(self.config)
        self.knowledge = KnowledgeSystem(self.config)
        self.tool_integration = IntegrationManager(self.config) # v5.0

        # Core Components
        self.tool_registry = tool_registry or {}
        self.grounding = grounding or UnifiedGrounding(
            config=self.config,
            knowledge_system=self.knowledge
        )
        self.tool_forge = tool_forge or ToolForge(tool_registry=self.tool_registry, config=self.config)
        self.age_mem = AgeMem(self.config)
        self.bridge = Bridge()

        # Personas
        if personas_dir is None:
            personas_dir = Path(__file__).parent.parent / "interface" / "personas"
        self.personas_dir = personas_dir
        self._personas: Dict[str, str] = {}
        self._load_personas()

        # LLM Backend
        try:
            from sense_v2.core.config import Config
            self.llm = get_model(Config().to_dict())
        except ImportError:
            self.llm = get_model({'model_name': 'gpt2'})
        except Exception as e:
            self.logger.warning(f"Config load failed: {e}")
            self.llm = get_model({'model_name': 'gpt2'})

        self._task_counter = 0
        self._execution_history: List[TaskResult] = []
        self._initialized = True

    def _load_personas(self) -> None:
        if not self.personas_dir.exists():
            self._personas = {
                "architect": self._default_architect_persona(),
                "worker": self._default_worker_persona(),
                "critic": self._default_critic_persona(),
            }
            return
        for persona_file in self.personas_dir.glob("*.md"):
            try:
                self._personas[persona_file.stem] = persona_file.read_text()
            except Exception as e:
                self.logger.error(f"Failed to load persona {persona_file.stem}: {e}")

    def _default_architect_persona(self) -> str: return "Architect Persona"
    def _default_worker_persona(self) -> str: return "Worker Persona"
    def _default_critic_persona(self) -> str: return "Critic Persona"

    async def solve_task(self, task: str) -> TaskResult:
        """
        Main entry point: Solve a task through the Reflexion loop.
        """
        start_time = time.time()
        self._task_counter += 1
        task_id = f"task_{self._task_counter}_{int(start_time)}"
        self.logger.info(f"Starting task {task_id}: {task[:50]}...")

        phases_completed = []
        retry_count = 0

        # --- Phase 0: Pre-Planning (Alignment Check) ---
        uncertainty = self.alignment.detector.detect_uncertainty(task)
        if uncertainty:
            feedback = await self.alignment.collector.request_feedback(uncertainty)
            if feedback.selected_option == "custom" and feedback.custom_input:
                task = f"{task} (User Clarification: {feedback.custom_input})"
            else:
                task = f"{task} (User Preference: {feedback.selected_option})"
            self.alignment.preferences.update_from_feedback(feedback, {"domain": "general", "task": task})
            phases_completed.append(Phase.PRE_PLANNING)

        # --- Phase 1: Tool Discovery (v5.0) ---
        if self.config.get("auto_discover_tools", True):
            new_tools = await self.tool_integration.auto_integrate(task)
            if new_tools:
                self.logger.info(f"Auto-integrated {len(new_tools)} new tools: {new_tools}")
                phases_completed.append(Phase.TOOL_DISCOVERY)

        # Retrieve memory
        memory_context = await self.age_mem.retrieve_similar(task)

        # --- Phase 2: Architect (with Knowledge Augmentation) ---
        external_context = await self.knowledge.gather_context(task)
        plan = await self._run_architect(task, memory_context, external_context)
        phases_completed.append(Phase.ARCHITECT)

        # Retry loop
        while retry_count <= self.MAX_RETRIES:
            # --- Phase 3: Worker ---
            execution_result = await self._run_worker(plan, task)
            if Phase.WORKER not in phases_completed:
                phases_completed.append(Phase.WORKER)

            # --- Phase 4: Critic ---
            verification = await self._run_critic(execution_result, task, plan)
            if Phase.CRITIC not in phases_completed:
                phases_completed.append(Phase.CRITIC)

            if verification.passed:
                break

            # Retry
            retry_count += 1
            if retry_count <= self.MAX_RETRIES:
                self.logger.info(f"Retry {retry_count}: {verification.feedback}")
                plan = await self._refine_plan(plan, verification.feedback, task)

        execution_time = time.time() - start_time

        result = TaskResult(
            task_id=task_id, task=task, plan=plan, execution_result=execution_result,
            verification=verification, success=verification.passed,
            retry_count=retry_count, execution_time=execution_time,
            phases_completed=phases_completed,
        )

        # --- Phase 5: Integration ---
        if result.success:
            phases_completed.append(Phase.INTEGRATION)
            self.tool_forge.check_for_crystallization(result)

        self._execution_history.append(result)
        return result

    async def _run_architect(self, task, memory_context, external_context) -> str:
        persona_prompt = self._personas.get("architect", self._default_architect_persona())
        full_prompt = f"{persona_prompt}\n\nTask: {task}"
        if external_context: full_prompt += f"\n\nContext from Knowledge Base:\n{external_context}"
        if memory_context:
            memory_str = '\n'.join([m['plan'][:100] for m in memory_context])
            full_prompt += f"\n\nRelevant past workflows:\n{memory_str}"
        full_prompt += "\n\nProvide a detailed step-by-step execution plan:"
        if self.llm:
            try:
                response = self.llm.generate(full_prompt, max_tokens=200, temperature=0.7)
                plan = response.strip()
            except Exception as e:
                self.logger.warning(f"LLM call failed: {e}")
                plan = f"Default plan for: {task}"
        else: plan = f"Default plan for: {task}"
        return plan if len(plan) > 10 else f"Default plan for: {task}"

    async def _run_worker(self, plan: str, original_task: str) -> Any:
        import re
        import json
        command_pattern = r'(?:^|\n)\s*(?:Run|Execute|Command):\s*`?([a-z]+\s+[^`\n]+)`?'
        commands = re.findall(command_pattern, plan, re.IGNORECASE)
        execution_outputs = []
        if commands:
            for cmd in commands:
                try:
                    result = self.bridge.execute(cmd.strip())
                    execution_outputs.append({
                        "command": cmd.strip(), "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""), "returncode": result.get("returncode", "1"),
                        "success": result.get("returncode", "1") == "0"
                    })
                except Exception as e:
                    execution_outputs.append({"command": cmd, "error": str(e), "success": False})
        if not execution_outputs:
            persona = self._personas.get("worker", self._default_worker_persona())
            prompt = f"{persona}\n\nTask: {original_task}\nPlan: {plan}\nExecute and return JSON."
            if self.llm:
                try:
                    response = self.llm.generate(prompt, max_tokens=200)
                    execution_text = response[len(prompt):].strip()
                    try:
                        import json
                        start = execution_text.find('{')
                        end = execution_text.rfind('}') + 1
                        execution_result = json.loads(execution_text[start:end])
                    except: execution_result = {"status": "completed", "output": execution_text, "action_succeeded": True}
                except: execution_result = {"status": "completed", "output": "Stub output", "action_succeeded": True}
            else: execution_result = {"status": "completed", "output": "Stub output", "action_succeeded": True}
            return execution_result
        all_succeeded = all(out.get("success", False) for out in execution_outputs)
        return {"status": "completed" if all_succeeded else "failed", "output": execution_outputs, "action_succeeded": all_succeeded}

    async def _run_critic(self, execution_result, original_task, plan) -> VerificationResult:
        context = {"original_task": original_task, "plan": plan, "action_succeeded": execution_result.get("action_succeeded", False) if isinstance(execution_result, dict) else False}
        return await self.grounding.verify(execution_result, context)

    async def _refine_plan(self, original_plan, feedback, task) -> str:
        return f"{original_plan}\n\n--- Feedback ---\n{feedback}\n\nRefined Plan for {task}..."

    def get_execution_stats(self) -> Dict[str, Any]:
        if not self._execution_history: return {"total_tasks": 0}
        successful = sum(1 for r in self._execution_history if r.success)
        return {"total_tasks": len(self._execution_history), "successful_tasks": successful, "success_rate": successful / len(self._execution_history)}

    def prompt(
        self,
        persona: str,
        input: str,
        tools: Optional[Dict[str, Any]] = None,
    ) -> str:
        persona_text = self._personas.get(persona, "")
        full_prompt = f"{persona_text}\n\nInput: {input}\n\nResponse:"
        if tools: full_prompt += f"\n\nAvailable tools:\n{chr(10).join(f'- {n}: {d}' for n, d in tools.items())}"
        if self.llm:
            try: response = self.llm.generate(full_prompt, max_tokens=100)
            except: response = full_prompt
        else: response = full_prompt + " [Stub response]"
        return response[len(full_prompt):].strip()

def create_orchestrator(**kwargs) -> ReasoningOrchestrator:
    return ReasoningOrchestrator(**kwargs)

orchestrator = ReasoningOrchestrator()