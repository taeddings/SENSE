"""
SENSE-v2 Executor Agent (Student)
Executes curriculum tasks and learns through tool-verified success.
Part of Agent 0 - The School co-evolutionary system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import asyncio
from datetime import datetime
import traceback

from sense.core.base import BaseAgent, BaseTool, AgentState
from sense.core.schemas import (
    AgentMessage,
    MessageRole,
    ToolResult,
    ToolResultStatus,
    RewardSignal,
)
from sense.core.config import EvolutionConfig
from sense.agents.agent_0.curriculum import CurriculumTask


@dataclass
class ExecutionTrace:
    """Record of a single execution attempt."""
    task_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reward: Optional[RewardSignal] = None
    total_time_ms: int = 0
    success: bool = False
    error: Optional[str] = None

    def add_step(self, action: str, result: Any, time_ms: int = 0) -> None:
        self.steps.append({
            "action": action,
            "result": str(result)[:500],
            "time_ms": time_ms,
            "timestamp": datetime.now().isoformat(),
        })

    def add_tool_call(self, tool_name: str, args: Dict, result: ToolResult) -> None:
        self.tool_calls.append({
            "tool": tool_name,
            "args": args,
            "success": result.is_success,
            "output": str(result.output)[:500] if result.output else None,
            "error": result.error,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "reward": self.reward.value if self.reward else None,
            "total_time_ms": self.total_time_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ExecutorState:
    """State tracking for the executor agent."""
    generation: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    current_fitness: float = 0.0
    best_fitness: float = 0.0
    exploration_rate: float = 0.3
    learning_rate: float = 0.1

    # Strategy weights learned over time
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "direct_execution": 0.5,
        "plan_then_execute": 0.3,
        "iterative_refinement": 0.2,
    })

    def update_from_reward(self, reward: RewardSignal, strategy: str) -> None:
        """Update state based on reward signal."""
        self.total_executions += 1

        if reward.value >= 0.5:
            self.successful_executions += 1

        # Update fitness
        self.current_fitness = (
            self.current_fitness * 0.9 + reward.value * 0.1
        )
        self.best_fitness = max(self.best_fitness, self.current_fitness)

        # Update strategy weights
        if strategy in self.strategy_weights:
            delta = self.learning_rate * (reward.value - 0.5)
            self.strategy_weights[strategy] += delta

            # Normalize weights
            total = sum(self.strategy_weights.values())
            self.strategy_weights = {
                k: v / total for k, v in self.strategy_weights.items()
            }

        # Decay exploration rate
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions


class ExecutorAgent(BaseAgent):
    """
    Executor Agent (Student) for Agent 0 - The School.

    Responsibilities:
    - Execute tasks assigned by CurriculumAgent
    - Learn from success/failure via reward signals
    - Evolve execution strategies over time
    - Self-correct based on stderr/error analysis

    Per SYSTEM_PROMPT requirements:
    - Distinct from CurriculumAgent
    - Reward based on Unit Test Success or Terminal Exit Codes
    - Implements self-correction loop
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        super().__init__(name="ExecutorAgent", config=config, tools=tools)
        self.config = config or EvolutionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Executor state
        self.state = ExecutorState()

        # Execution history
        self.execution_traces: List[ExecutionTrace] = []
        self.max_traces = 1000

        # Current execution context
        self._current_task: Optional[CurriculumTask] = None
        self._current_trace: Optional[ExecutionTrace] = None

    async def execute_task(self, task: CurriculumTask) -> Tuple[RewardSignal, ExecutionTrace]:
        """
        Execute a curriculum task and return reward signal.

        Args:
            task: The task to execute

        Returns:
            Tuple of (RewardSignal, ExecutionTrace)
        """
        self._current_task = task
        self._current_trace = ExecutionTrace(task_id=task.task_id)

        start_time = datetime.now()

        try:
            # Select execution strategy
            strategy = self._select_strategy()
            self._current_trace.add_step(f"Selected strategy: {strategy}", None)

            # Execute based on strategy
            if strategy == "direct_execution":
                result = await self._direct_execute(task)
            elif strategy == "plan_then_execute":
                result = await self._plan_and_execute(task)
            else:  # iterative_refinement
                result = await self._iterative_execute(task)

            # Calculate reward
            reward = self._calculate_reward(task, result)

            # Update state
            self.state.update_from_reward(reward, strategy)

            # Finalize trace
            self._current_trace.reward = reward
            self._current_trace.success = reward.value >= 0.5
            self._current_trace.total_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            # Store trace
            self._store_trace(self._current_trace)

            return reward, self._current_trace

        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            self._current_trace.error = str(e)
            self._current_trace.add_step("Exception", traceback.format_exc())

            reward = RewardSignal.from_exit_code(1, task_id=task.task_id)
            self._current_trace.reward = reward
            self._store_trace(self._current_trace)

            return reward, self._current_trace

        finally:
            self._current_task = None
            self._current_trace = None

    def _select_strategy(self) -> str:
        """Select execution strategy based on learned weights and exploration."""
        import random

        if random.random() < self.state.exploration_rate:
            # Explore: random strategy
            return random.choice(list(self.state.strategy_weights.keys()))

        # Exploit: weighted selection
        strategies = list(self.state.strategy_weights.keys())
        weights = list(self.state.strategy_weights.values())
        return random.choices(strategies, weights=weights)[0]

    async def _direct_execute(self, task: CurriculumTask) -> Dict[str, Any]:
        """Direct execution strategy - attempt task immediately."""
        results = []

        for tool_name in task.expected_tools:
            tool = self.get_tool(tool_name)
            if tool:
                result = await self.call_tool(tool_name)
                self._current_trace.add_tool_call(tool_name, {}, result)
                results.append(result)

        return {
            "strategy": "direct_execution",
            "tool_results": results,
            "success": all(r.is_success for r in results) if results else False,
        }

    async def _plan_and_execute(self, task: CurriculumTask) -> Dict[str, Any]:
        """Plan-then-execute strategy - analyze task before execution."""
        self._current_trace.add_step("Planning phase", task.description)

        # Simple planning: order tools by dependency
        plan = []
        for tool_name in task.expected_tools:
            plan.append({"tool": tool_name, "purpose": "Execute task step"})

        self._current_trace.add_step("Generated plan", plan)

        # Execute plan
        results = []
        for step in plan:
            tool = self.get_tool(step["tool"])
            if tool:
                result = await self.call_tool(step["tool"])
                self._current_trace.add_tool_call(step["tool"], {}, result)
                results.append(result)

                # Abort on failure
                if not result.is_success:
                    break

        return {
            "strategy": "plan_then_execute",
            "plan": plan,
            "tool_results": results,
            "success": all(r.is_success for r in results) if results else False,
        }

    async def _iterative_execute(self, task: CurriculumTask) -> Dict[str, Any]:
        """Iterative refinement strategy - retry with corrections."""
        max_iterations = 3
        results = []

        for iteration in range(max_iterations):
            self._current_trace.add_step(f"Iteration {iteration + 1}", None)

            iteration_results = []
            for tool_name in task.expected_tools:
                tool = self.get_tool(tool_name)
                if tool:
                    result = await tool.execute_with_retry(max_retries=2)
                    self._current_trace.add_tool_call(tool_name, {}, result)
                    iteration_results.append(result)

            results.extend(iteration_results)

            # Check if successful
            if all(r.is_success for r in iteration_results):
                break

            # Analyze errors for next iteration
            errors = [r.error for r in iteration_results if r.error]
            self._current_trace.add_step("Error analysis", errors)

        return {
            "strategy": "iterative_refinement",
            "iterations": iteration + 1,
            "tool_results": results,
            "success": any(r.is_success for r in results) if results else False,
        }

    def _calculate_reward(self, task: CurriculumTask, result: Dict[str, Any]) -> RewardSignal:
        """
        Calculate reward signal from execution result.
        Per SYSTEM_PROMPT: Binary or scalar based on Unit Test Success or Exit Codes.
        """
        tool_results = result.get("tool_results", [])

        if not tool_results:
            return RewardSignal.from_exit_code(1, task_id=task.task_id)

        # Count successes
        successes = sum(1 for r in tool_results if r.is_success)
        total = len(tool_results)

        if self.config.reward_binary:
            # Binary reward: all must succeed
            exit_code = 0 if successes == total else 1
            return RewardSignal.from_exit_code(
                exit_code,
                task_id=task.task_id,
                difficulty_level=task.difficulty,
            )
        else:
            # Scalar reward based on success ratio
            return RewardSignal.from_unit_tests(
                passed=successes,
                total=total,
                task_id=task.task_id,
                difficulty_level=task.difficulty,
            )

    def _store_trace(self, trace: ExecutionTrace) -> None:
        """Store execution trace with size limit."""
        self.execution_traces.append(trace)

        # Prune old traces
        if len(self.execution_traces) > self.max_traces:
            self.execution_traces = self.execution_traces[-self.max_traces:]

    async def self_correct(self, error: str) -> Optional[str]:
        """
        Analyze error and suggest correction.
        Implements self-correction loop per SYSTEM_PROMPT.
        """
        error_lower = error.lower()

        corrections = {
            "permission denied": "Try with elevated permissions or check file ownership",
            "file not found": "Verify the path exists and is spelled correctly",
            "command not found": "Ensure the required tool is installed",
            "timeout": "Increase timeout or simplify the operation",
            "memory": "Reduce batch size or free up memory",
            "connection refused": "Check if the service is running",
        }

        for pattern, correction in corrections.items():
            if pattern in error_lower:
                return correction

        return None

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "generation": self.state.generation,
            "total_executions": self.state.total_executions,
            "successful_executions": self.state.successful_executions,
            "success_rate": self.state.success_rate,
            "current_fitness": self.state.current_fitness,
            "best_fitness": self.state.best_fitness,
            "exploration_rate": self.state.exploration_rate,
            "strategy_weights": self.state.strategy_weights,
            "traces_stored": len(self.execution_traces),
        }

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message (interface compliance)."""
        return AgentMessage.assistant(
            f"ExecutorAgent ready. Stats: {self.get_execution_stats()}"
        )

    async def run(self) -> None:
        """Main agent loop (interface compliance)."""
        self._is_running = True
        self.logger.info("ExecutorAgent started")

        while self._is_running:
            await asyncio.sleep(1)
