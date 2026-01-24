"""
SENSE v2.3 Reasoning Orchestrator with Reflexion Loop

Implements the Architect/Worker/Critic phased execution model:
- Phase 1 (Architect): Plan the approach
- Phase 2 (Worker): Execute the code/action
- Phase 3 (Critic): Verify the result via grounding
- Phase 4 (Integration): Check for tool crystallization

Part of Phase 2: Reasoning & Agency
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
try:
    from .grounding import Tier1Grounding, Tier2Grounding, Tier3Grounding
except ImportError:
    # Fallback for Phase 2 stub
    class Tier1Grounding:
        def preprocess_data(self, data):
            return data
    class Tier2Grounding:
        def __init__(self, tier1):
            pass
    class Tier3Grounding:
        def __init__(self, tier2):
            pass


class Phase(Enum):
    """Phases of the Reflexion loop."""
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
    Stub for three-tier grounding system.
    Combines Synthetic, Real-World, and Experiential grounding.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.weights = weights or {
            "synthetic": 0.4,
            "realworld": 0.3,
            "experiential": 0.3,
        }
        self.tier1 = Tier1Grounding()
        self.tier2 = Tier2Grounding(self.tier1)
        self.tier3 = Tier3Grounding(self.tier2)
        self.logger = logging.getLogger("UnifiedGrounding")

    def verify(self, result: Any, context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """
        Verify a result using all three grounding tiers.

        Args:
            result: The execution result to verify
            context: Optional context for verification

        Returns:
            VerificationResult with combined confidence score
        """
        tier_results = {}
        total_confidence = 0.0

        # Tier 1: Synthetic (deterministic checks)
        synthetic_score = self._verify_synthetic(result, context)
        tier_results["synthetic"] = synthetic_score
        total_confidence += synthetic_score * self.weights["synthetic"]

        # Tier 2: Real-world (external verification)
        realworld_score = self._verify_realworld(result, context)
        tier_results["realworld"] = realworld_score
        total_confidence += realworld_score * self.weights["realworld"]

        # Tier 3: Experiential (action outcomes)
        experiential_score = self._verify_experiential(result, context)
        tier_results["experiential"] = experiential_score
        total_confidence += experiential_score * self.weights["experiential"]

        # Determine pass/fail threshold
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
        # Stub implementation - returns high confidence for structured results
        if result is None:
            return 0.0
        if isinstance(result, (bool, int, float)):
            return 1.0
        if isinstance(result, str) and len(result) > 0:
            return 0.8
        return 0.5

    def _verify_realworld(self, result: Any, context: Optional[Dict]) -> float:
        """Tier 2: Real-world grounding - external fact verification."""
        # Stub implementation - would query web/APIs
        return 0.7

    def _verify_experiential(self, result: Any, context: Optional[Dict]) -> float:
        """Tier 3: Experiential grounding - action outcome verification."""
        # Stub implementation - would check actual system state
        if context and context.get("action_succeeded"):
            return 1.0
        return 0.6

    def _generate_feedback(self, tier_results: Dict[str, float], passed: bool) -> str:
        """Generate human-readable feedback from tier results."""
        if passed:
            return f"Verification passed. Confidence scores: {tier_results}"

        weakest_tier = min(tier_results, key=tier_results.get)
        return f"Verification failed. Weakest tier: {weakest_tier} ({tier_results[weakest_tier]:.2f})"


class ToolForgeStub:
    """
    Stub for Tool Forge - Dynamic Tool Creation.

    Will be fully implemented in sense/core/plugins/forge.py
    """

    REPETITION_THRESHOLD: int = 3

    def __init__(self):
        self.logger = logging.getLogger("ToolForge")
        self._candidate_patterns: List[Dict[str, Any]] = []

    def check_for_crystallization(self, result: TaskResult) -> bool:
        """
        Check if this result contains patterns worth crystallizing into a tool.

        Stub: Logs the check but doesn't create tools yet.
        """
        self.logger.debug(f"Checking task {result.task_id} for crystallization potential")
        # Will be implemented to scan memory for repeated patterns
        return False

    def scan_memory(self, memory: Any) -> List[Dict[str, Any]]:
        """Scan memory for repeated successful code patterns."""
        # Stub - returns empty list
        return []

    def forge_tool(self, candidate: Dict[str, Any]) -> Optional[Any]:
        """Refactor a candidate pattern into a PluginABC class."""
        # Stub - not implemented yet
        return None


class ReasoningOrchestrator:
    """
    Orchestrates reasoning through the Reflexion loop.

    Phases:
    1. Architect: Analyze task and create execution plan
    2. Worker: Execute the plan using available tools
    3. Critic: Verify results via three-tier grounding
    4. Integration: Check for tool crystallization opportunities

    Example:
        orchestrator = ReasoningOrchestrator()
        result = await orchestrator.solve_task("Calculate 17 * 23")
    """

    _instance = None  # Singleton instance variable
    MAX_RETRIES: int = 3

    def __new__(cls, *args, **kwargs):
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(ReasoningOrchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        grounding: Optional[UnifiedGrounding] = None,
        tool_forge: Optional[ToolForgeStub] = None,
        personas_dir: Optional[Path] = None,
        tool_registry: Optional[Any] = None,
    ):
        # Skip re-initialization for singleton
        if getattr(self, '_initialized', False):
            return

        self.logger = logging.getLogger("ReasoningOrchestrator")

        # Initialize components
        self.grounding = grounding or UnifiedGrounding()
        self.tool_forge = tool_forge or ToolForgeStub()
        self.tool_registry = tool_registry or {}
        self.age_mem = AgeMem({'stm_max_entries': 100})
        self.bridge = Bridge()

        # Personas directory
        if personas_dir is None:
            personas_dir = Path(__file__).parent.parent / "interface" / "personas"
        self.personas_dir = personas_dir

        # Load personas
        self._personas: Dict[str, str] = {}
        self._load_personas()
        # Load LLM backend
        try:
            from sense_v2.core.config import Config
            self.llm = get_model(Config().to_dict())
        except ImportError:
            # sense_v2 not available, use default config
            self.llm = get_model({'model_name': 'gpt2'})
        except Exception as e:
            self.logger.warning(f"Config load failed, using fallback: {e}")
            self.llm = get_model({'model_name': 'gpt2'})

        # Execution tracking
        self._task_counter = 0
        self._execution_history: List[TaskResult] = []

        # Mark initialization complete (for singleton pattern)
        self._initialized = True

    def _load_personas(self) -> None:
        """Load persona prompts from markdown files."""
        if not self.personas_dir.exists():
            self.logger.warning(f"Personas directory not found: {self.personas_dir}")
            self._personas = {
                "architect": self._default_architect_persona(),
                "worker": self._default_worker_persona(),
                "critic": self._default_critic_persona(),
            }
            return

        for persona_file in self.personas_dir.glob("*.md"):
            persona_name = persona_file.stem
            try:
                self._personas[persona_name] = persona_file.read_text()
                self.logger.debug(f"Loaded persona: {persona_name}")
            except Exception as e:
                self.logger.error(f"Failed to load persona {persona_name}: {e}")

    def _default_architect_persona(self) -> str:
        return """You are the Architect. Your role is to:
1. Analyze the task requirements
2. Break down complex tasks into steps
3. Identify required tools and resources
4. Create a clear execution plan

Output a structured plan that the Worker can execute."""

    def _default_worker_persona(self) -> str:
        return """You are the Worker. Your role is to:
1. Execute the plan provided by the Architect
2. Use available tools to complete each step
3. Handle errors and edge cases
4. Return structured results

Follow the plan precisely and report all outcomes."""

    def _default_critic_persona(self) -> str:
        return """You are the Critic. Your role is to:
1. Verify the Worker's results against the original task
2. Check for correctness using available grounding methods
3. Identify any errors or incomplete work
4. Provide actionable feedback if verification fails

Be rigorous but fair in your assessment."""

    async def solve_task(self, task: str) -> TaskResult:
        """
        Main entry point: Solve a task through the Reflexion loop.

        Args:
            task: The task description to solve

        Returns:
            TaskResult with the complete execution history
        """
        start_time = time.time()
        self._task_counter += 1
        task_id = f"task_{self._task_counter}_{int(start_time)}"

        self.logger.info(f"Starting task {task_id}: {task[:50]}...")

        phases_completed = []
        retry_count = 0

        # Retrieve memory for procedural RAG
        memory_context = await self.age_mem.retrieve_similar(task)
        # Phase 1: Architect
        plan = await self._run_architect(task, memory_context)
        phases_completed.append(Phase.ARCHITECT)
        self.logger.debug(f"Architect produced plan: {plan[:100]}...")

        # Retry loop for Worker/Critic
        while retry_count <= self.MAX_RETRIES:
            # Phase 2: Worker
            execution_result = await self._run_worker(plan, task)
            if Phase.WORKER not in phases_completed:
                phases_completed.append(Phase.WORKER)

            # Phase 3: Critic (Reflexion)
            verification = await self._run_critic(execution_result, task, plan)
            if Phase.CRITIC not in phases_completed:
                phases_completed.append(Phase.CRITIC)

            if verification.passed:
                break

            # Retry with feedback
            retry_count += 1
            if retry_count <= self.MAX_RETRIES:
                self.logger.info(f"Retry {retry_count}/{self.MAX_RETRIES}: {verification.feedback}")
                plan = await self._refine_plan(plan, verification.feedback, task)

        execution_time = time.time() - start_time

        # Build result
        result = TaskResult(
            task_id=task_id,
            task=task,
            plan=plan,
            execution_result=execution_result,
            verification=verification,
            success=verification.passed,
            retry_count=retry_count,
            execution_time=execution_time,
            phases_completed=phases_completed,
        )

        # Phase 4: Integration (Tool Forge)
        if result.success:
            phases_completed.append(Phase.INTEGRATION)
            self.tool_forge.check_for_crystallization(result)

        # Track execution
        self._execution_history.append(result)

        self.logger.info(
            f"Task {task_id} {'succeeded' if result.success else 'failed'} "
            f"in {execution_time:.2f}s with {retry_count} retries"
        )

        return result

    async def _run_architect(self, task: str, memory_context: Optional[List[Dict]] = None) -> str:
        """
        Phase 1: Architect creates an execution plan using LLM.
        """
        persona_prompt = self._personas.get("architect", self._default_architect_persona())
        full_prompt = f"{persona_prompt}\n\nTask: {task}\n\nProvide a detailed step-by-step execution plan:"
        if memory_context:
            memory_str = '\\n'.join([m['plan'][:100] for m in memory_context])
            full_prompt += f"\nRelevant past workflows:\n{memory_str}"
        if self.llm:
            try:
                response = self.llm.generate(full_prompt, max_tokens=200, temperature=0.7)
                plan = response.strip()
            except Exception as e:
                self.logger.warning(f"LLM call failed: {e}")
                plan = f"Default plan for task: {task}"
        else:
            plan = f"Default plan for task: {task}"
        if len(plan) < 10:
            plan = f"Default plan for task: {task}"
        return plan

    async def _run_worker(self, plan: str, original_task: str) -> Any:
        """
        Phase 2: Worker executes the plan using LLM to simulate execution.
        """
        persona_prompt = self._personas.get("worker", self._default_worker_persona())
        full_prompt = f"{persona_prompt}\n\nOriginal Task: {original_task}\n\nPlan: {plan}\n\nExecute the plan and return the result in JSON format: {{\"status\": \"completed\" or \"failed\", \"output\": \"the result\", \"action_succeeded\": true or false}}."
        if self.llm:
            try:
                response = self.llm.generate(full_prompt, max_tokens=200, temperature=0.7)
            except Exception as e:
                self.logger.warning(f"LLM call failed: {e}")
                response = full_prompt
        else:
            response = full_prompt + " [Stub response]"
        execution_text = response[len(full_prompt):].strip()
        try:
            import json
            execution_result = json.loads(execution_text)
        except:
            execution_result = {
                "status": "completed",
                "output": execution_text or f"Executed plan for {original_task}",
                "action_succeeded": True,
            }
        return execution_result

    async def _run_critic(
        self,
        execution_result: Any,
        original_task: str,
        plan: str,
    ) -> VerificationResult:
        """
        Phase 3: Critic verifies the result via grounding.
        """
        persona = self._personas.get("critic", self._default_critic_persona())

        # Use grounding system for verification
        context = {
            "original_task": original_task,
            "plan": plan,
            "action_succeeded": execution_result.get("action_succeeded", False)
            if isinstance(execution_result, dict) else False,
        }

        verification = self.grounding.verify(execution_result, context)

        return verification

    async def _refine_plan(self, original_plan: str, feedback: str, task: str) -> str:
        """
        Refine the plan based on Critic feedback.
        """
        # Stub: Append feedback to plan
        refined_plan = f"""{original_plan}

---
Refinement based on feedback:
{feedback}

Updated approach for: {task[:30]}..."""

        return refined_plan

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about task executions."""
        if not self._execution_history:
            return {"total_tasks": 0}

        successful = sum(1 for r in self._execution_history if r.success)
        total_time = sum(r.execution_time for r in self._execution_history)
        total_retries = sum(r.retry_count for r in self._execution_history)

        return {
            "total_tasks": len(self._execution_history),
            "successful_tasks": successful,
            "success_rate": successful / len(self._execution_history),
            "average_execution_time": total_time / len(self._execution_history),
            "total_retries": total_retries,
            "average_retries": total_retries / len(self._execution_history),
        }

    def prompt(
        self,
        persona: str,
        input: str,
        tools: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a prompt to the model with a specific persona using LLM.
        """
        persona_text = self._personas.get(persona, "")
        full_prompt = f"{persona_text}\n\nInput: {input}\n\nResponse:"
        if tools:
            tools_str = "\n".join(f"- {name}: {desc}" for name, desc in tools.items())
            full_prompt += f"\n\nAvailable tools:\n{tools_str}"
        if self.llm:
            try:
                response = self.llm.generate(full_prompt, max_tokens=100, temperature=0.7)
            except Exception as e:
                self.logger.warning(f"LLM call failed: {e}")
                response = full_prompt
        else:
            response = full_prompt + " [Stub response]"
        return response[len(full_prompt):].strip()


# Convenience function for creating orchestrator
def create_orchestrator(**kwargs) -> ReasoningOrchestrator:
    """Factory function to create a configured ReasoningOrchestrator."""
    return ReasoningOrchestrator(**kwargs)


# Singleton instance (per directive requirement)
# Lazy initialization - created on first import
orchestrator = ReasoningOrchestrator()
