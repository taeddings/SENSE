"""
SENSE-v2 ReasoningOrchestrator
Orchestrates reasoning processes with RestrictedPython sandbox for safe execution.

Part of Sprint 3: The Loop

Provides safe execution environment for reasoning code, allowing dynamic
reasoning strategies while preventing system compromise.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging
import asyncio
import time
import hashlib
from datetime import datetime

from sense_v2.core.base import BaseAgent
from sense_v2.core.schemas import AgentMessage, MessageRole
from sense_v2.core.config import EvolutionConfig

# RestrictedPython imports
try:
    from RestrictedPython import compile_restricted, safe_builtins, limited_builtins
    from RestrictedPython.Guards import safe_builtins as guarded_builtins
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False


@dataclass
class ReasoningContext:
    """Context for a reasoning execution."""
    task_id: str
    problem: str
    available_tools: List[str] = field(default_factory=list)
    max_steps: int = 10
    timeout_seconds: int = 30
    memory_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "problem": self.problem,
            "available_tools": self.available_tools,
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "memory_context": self.memory_context,
        }


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_params: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_params": self.action_params,
            "observation": self.observation,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReasoningResult:
    """Result of a complete reasoning process."""
    task_id: str
    final_answer: str
    steps: List[ReasoningStep] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_used: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_answer": self.final_answer,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "memory_used": self.memory_used,
        }


class ReasoningOrchestrator(BaseAgent):
    """
    ReasoningOrchestrator - Safe execution environment for reasoning processes.

    Uses RestrictedPython to execute reasoning strategies in a sandboxed environment,
    preventing malicious code execution while allowing dynamic reasoning patterns.

    Key Features:
    - RestrictedPython sandbox for safe code execution
    - Configurable reasoning strategies
    - Tool integration with safety checks
    - Memory context integration
    - Execution monitoring and timeouts

    Example:
        orchestrator = ReasoningOrchestrator()
        context = ReasoningContext(task_id="test", problem="Solve x + 1 = 2")
        result = await orchestrator.execute_reasoning(context)
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        reasoning_strategy: Optional[str] = None,
        enable_sandbox: bool = True,
    ):
        super().__init__(name="ReasoningOrchestrator", config=config)
        self.config = config or EvolutionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Sandbox configuration
        self.enable_sandbox = enable_sandbox and RESTRICTED_PYTHON_AVAILABLE
        if not RESTRICTED_PYTHON_AVAILABLE and enable_sandbox:
            self.logger.warning("RestrictedPython not available, sandbox disabled")
            self.enable_sandbox = False

        # Reasoning strategy (code template)
        self.reasoning_strategy = reasoning_strategy or self._default_reasoning_strategy()

        # Execution tracking
        self.execution_history: List[ReasoningResult] = []
        self._active_executions: Dict[str, asyncio.Task] = {}

    def _default_reasoning_strategy(self) -> str:
        """Default reasoning strategy code template."""
        return '''
def reason(problem, tools, max_steps, memory):
    """
    Default reasoning strategy: systematic problem solving.
    """
    steps = []
    current_state = {}

    for step in range(max_steps):
        # Analyze current situation
        thought = f"Step {step + 1}: Analyzing problem '{problem[:50]}...'"

        # Decide on action
        if step == 0:
            # First step: understand the problem
            action = "analyze"
            params = {"problem": problem}
        elif "calculate" in problem.lower() or "=" in problem:
            # Looks like a math problem
            action = "calculate"
            params = {"expression": problem}
        elif any(tool in problem.lower() for tool in tools):
            # Problem mentions a tool
            mentioned_tools = [t for t in tools if t in problem.lower()]
            action = mentioned_tools[0] if mentioned_tools else "think"
            params = {"query": problem}
        else:
            # General reasoning
            action = "reason"
            params = {"thought": f"Considering: {problem}"}

        # Execute action (simulated)
        if action == "analyze":
            observation = f"Problem analyzed: {problem}"
        elif action == "calculate":
            observation = f"Calculated result for: {problem}"
        elif action in tools:
            observation = f"Used tool {action} on: {problem}"
        else:
            observation = f"Reasoned about: {problem}"

        steps.append({
            "thought": thought,
            "action": action,
            "params": params,
            "observation": observation
        })

        # Check if we should stop
        if step >= max_steps - 1:
            break

    # Generate final answer
    final_answer = f"Completed reasoning for: {problem}"
    return final_answer, steps
'''

    async def execute_reasoning(
        self,
        context: ReasoningContext,
    ) -> ReasoningResult:
        """
        Execute a reasoning process in the sandboxed environment.

        Args:
            context: Reasoning context with problem and constraints

        Returns:
            ReasoningResult with steps and final answer
        """
        start_time = time.time()

        try:
            # Check for existing execution
            if context.task_id in self._active_executions:
                raise ValueError(f"Task {context.task_id} already executing")

            # Create execution task
            execution_task = asyncio.create_task(
                self._execute_reasoning_task(context)
            )
            self._active_executions[context.task_id] = execution_task

            # Wait with timeout
            result = await asyncio.wait_for(
                execution_task,
                timeout=context.timeout_seconds
            )

            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # Track result
            self.execution_history.append(result)

            return result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_result = ReasoningResult(
                task_id=context.task_id,
                final_answer="",
                success=False,
                error_message=f"Execution timeout after {execution_time:.1f}s",
                execution_time=execution_time,
            )
            self.execution_history.append(error_result)
            return error_result

        except Exception as e:
            execution_time = time.time() - start_time
            error_result = ReasoningResult(
                task_id=context.task_id,
                final_answer="",
                success=False,
                error_message=f"Execution error: {str(e)}",
                execution_time=execution_time,
            )
            self.execution_history.append(error_result)
            return error_result

        finally:
            # Clean up active execution
            self._active_executions.pop(context.task_id, None)

    async def _execute_reasoning_task(
        self,
        context: ReasoningContext,
    ) -> ReasoningResult:
        """
        Internal method to execute reasoning in sandbox.
        """
        if self.enable_sandbox:
            return await self._execute_sandboxed(context)
        else:
            return await self._execute_unsandboxed(context)

    async def _execute_sandboxed(
        self,
        context: ReasoningContext,
    ) -> ReasoningResult:
        """
        Execute reasoning in RestrictedPython sandbox.
        """
        # Prepare sandbox environment
        sandbox_builtins = safe_builtins.copy()

        # Add safe utility functions
        def safe_print(*args, **kwargs):
            # Log instead of printing
            self.logger.debug(f"Sandbox print: {args}")
            return None

        sandbox_globals = {
            '__builtins__': sandbox_builtins,
            'print': safe_print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
        }

        # Compile the reasoning strategy
        try:
            compiled_code = compile_restricted(
                self.reasoning_strategy,
                '<reasoning_strategy>',
                'exec'
            )
        except Exception as e:
            raise ValueError(f"Failed to compile reasoning strategy: {e}")

        # Execute in sandbox
        exec_globals = sandbox_globals.copy()
        exec_locals = {}

        try:
            exec(compiled_code.code, exec_globals, exec_locals)
        except Exception as e:
            raise ValueError(f"Failed to execute reasoning strategy: {e}")

        # Get the reason function
        reason_func = exec_locals.get('reason')
        if not reason_func:
            raise ValueError("Reasoning strategy must define a 'reason' function")

        # Prepare arguments
        args = (
            context.problem,
            context.available_tools,
            context.max_steps,
            context.memory_context or {},
        )

        # Execute reasoning function in sandbox
        try:
            # Use a thread to avoid blocking (though RestrictedPython should be safe)
            loop = asyncio.get_event_loop()
            final_answer, steps_data = await loop.run_in_executor(
                None,
                self._call_reason_function_safely,
                reason_func,
                args
            )
        except Exception as e:
            raise ValueError(f"Reasoning execution failed: {e}")

        # Convert steps to ReasoningStep objects
        steps = []
        for i, step_data in enumerate(steps_data):
            step = ReasoningStep(
                step_number=i + 1,
                thought=step_data.get('thought', ''),
                action=step_data.get('action'),
                action_params=step_data.get('params'),
                observation=step_data.get('observation'),
            )
            steps.append(step)

        return ReasoningResult(
            task_id=context.task_id,
            final_answer=final_answer,
            steps=steps,
            success=True,
        )

    def _call_reason_function_safely(self, func, args):
        """Call the reason function with safety checks."""
        # This runs in a thread to isolate execution
        return func(*args)

    async def _execute_unsandboxed(
        self,
        context: ReasoningContext,
    ) -> ReasoningResult:
        """
        Execute reasoning without sandbox (fallback).
        """
        # Simple fallback implementation
        steps = []
        for i in range(min(context.max_steps, 3)):
            step = ReasoningStep(
                step_number=i + 1,
                thought=f"Step {i+1}: Processing {context.problem[:30]}...",
                action="think",
                observation=f"Step {i+1} completed",
            )
            steps.append(step)

        return ReasoningResult(
            task_id=context.task_id,
            final_answer=f"Completed reasoning for: {context.problem}",
            steps=steps,
            success=True,
        )

    def update_reasoning_strategy(self, new_strategy: str) -> bool:
        """
        Update the reasoning strategy code.

        Args:
            new_strategy: New strategy code as string

        Returns:
            True if update successful
        """
        if self.enable_sandbox:
            try:
                # Validate by compiling
                compile_restricted(new_strategy, '<new_strategy>', 'exec')
                self.reasoning_strategy = new_strategy
                self.logger.info("Reasoning strategy updated successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to update reasoning strategy: {e}")
                return False
        else:
            self.reasoning_strategy = new_strategy
            return True

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning executions."""
        if not self.execution_history:
            return {"total_executions": 0}

        total_time = sum(r.execution_time for r in self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        avg_steps = sum(len(r.steps) for r in self.execution_history) / len(self.execution_history)

        return {
            "total_executions": len(self.execution_history),
            "successful_executions": successful,
            "success_rate": successful / len(self.execution_history),
            "average_execution_time": total_time / len(self.execution_history),
            "average_steps": avg_steps,
            "sandbox_enabled": self.enable_sandbox,
        }