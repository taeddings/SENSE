"""
SENSE-v2 Master Agent
Hierarchical orchestration layer for Agent Zero - The Workplace.
Delegates to specialized sub-agents for OS-level task execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from enum import Enum
import logging
import asyncio
from datetime import datetime

from sense_v2.core.base import BaseAgent, BaseTool, AgentState
from sense_v2.core.schemas import AgentMessage, MessageRole, ToolResult
from sense_v2.core.config import OrchestrationConfig


class TaskType(Enum):
    """Types of tasks that can be delegated."""
    TERMINAL = "terminal"
    FILESYSTEM = "filesystem"
    BROWSER = "browser"
    REASONING = "reasoning"
    COMPOSITE = "composite"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class DelegatedTask:
    """A task delegated to a sub-agent."""
    task_id: str
    task_type: TaskType
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    parent_task_id: Optional[str] = None
    sub_agent_name: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "priority": self.priority.value,
            "parent_task_id": self.parent_task_id,
            "sub_agent_name": self.sub_agent_name,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
        }


class MasterAgent(BaseAgent):
    """
    Master Agent for Agent Zero - The Workplace.

    Per SYSTEM_PROMPT requirements:
    - Never performs heavy computation (delegates to sub-agents)
    - Aggregates results to keep context window lean
    - Implements hierarchical delegation
    - Max delegation depth enforced

    Responsibilities:
    - Parse high-level tasks into sub-tasks
    - Route tasks to appropriate sub-agents
    - Aggregate and summarize results
    - Handle failures with retries
    """

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        sub_agents: Optional[Dict[str, BaseAgent]] = None,
    ):
        super().__init__(name="MasterAgent", config=config)
        self.config = config or OrchestrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Sub-agents registry
        self.sub_agents: Dict[str, BaseAgent] = sub_agents or {}

        # Task management
        self.task_queue: List[DelegatedTask] = []
        self.active_tasks: Dict[str, DelegatedTask] = {}
        self.completed_tasks: List[DelegatedTask] = []
        self.max_completed_history = 100

        # Delegation tracking
        self._current_delegation_depth = 0
        self._task_counter = 0

        # Context management
        self._context_tokens = 0

    def register_sub_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a sub-agent for delegation."""
        self.sub_agents[name] = agent
        self.logger.debug(f"Registered sub-agent: {name}")

    def get_sub_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a registered sub-agent by name."""
        return self.sub_agents.get(name)

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{int(datetime.now().timestamp())}_{self._task_counter}"

    def _classify_task(self, description: str) -> TaskType:
        """
        Classify task to determine appropriate sub-agent.
        Simple keyword-based classification (upgrade to LLM in production).
        """
        desc_lower = description.lower()

        terminal_keywords = ["run", "execute", "command", "shell", "terminal", "install", "build"]
        filesystem_keywords = ["file", "read", "write", "create", "delete", "directory", "path"]
        browser_keywords = ["browse", "web", "url", "fetch", "download", "http", "api"]

        if any(kw in desc_lower for kw in terminal_keywords):
            return TaskType.TERMINAL
        elif any(kw in desc_lower for kw in filesystem_keywords):
            return TaskType.FILESYSTEM
        elif any(kw in desc_lower for kw in browser_keywords):
            return TaskType.BROWSER
        else:
            return TaskType.REASONING

    def _get_agent_for_task_type(self, task_type: TaskType) -> Optional[str]:
        """Get appropriate sub-agent name for task type."""
        mapping = {
            TaskType.TERMINAL: "terminal",
            TaskType.FILESYSTEM: "filesystem",
            TaskType.BROWSER: "browser",
        }
        return mapping.get(task_type)

    async def delegate_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        parent_task_id: Optional[str] = None,
    ) -> DelegatedTask:
        """
        Delegate a task to an appropriate sub-agent.

        Args:
            description: Task description
            priority: Task priority
            parent_task_id: Parent task for composite tasks

        Returns:
            DelegatedTask with result or error
        """
        # Check delegation depth
        if self._current_delegation_depth >= self.config.max_delegation_depth:
            task = DelegatedTask(
                task_id=self._generate_task_id(),
                task_type=TaskType.REASONING,
                description=description,
                priority=priority,
                parent_task_id=parent_task_id,
                status="failed",
                error="Max delegation depth exceeded",
            )
            return task

        # Classify and create task
        task_type = self._classify_task(description)
        task = DelegatedTask(
            task_id=self._generate_task_id(),
            task_type=task_type,
            description=description,
            priority=priority,
            parent_task_id=parent_task_id,
        )

        # Find appropriate sub-agent
        agent_name = self._get_agent_for_task_type(task_type)
        if agent_name and agent_name in self.sub_agents:
            task.sub_agent_name = agent_name
        else:
            # Handle locally if no sub-agent available
            task.sub_agent_name = "master"

        self.active_tasks[task.task_id] = task
        self.logger.info(f"Delegating task {task.task_id} to {task.sub_agent_name}")

        # Execute delegation
        try:
            self._current_delegation_depth += 1
            result = await self._execute_delegation(task)
            task.result = result
            task.status = "completed"
        except asyncio.TimeoutError:
            task.status = "timeout"
            task.error = "Task execution timed out"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)

            # Retry logic
            if task.retries < self.config.max_retries:
                task.retries += 1
                self.logger.warning(f"Retrying task {task.task_id} (attempt {task.retries})")
                return await self.delegate_task(description, priority, parent_task_id)
        finally:
            self._current_delegation_depth -= 1
            task.completed_at = datetime.now()

            # Move to completed
            del self.active_tasks[task.task_id]
            self._store_completed_task(task)

        return task

    async def _execute_delegation(self, task: DelegatedTask) -> Any:
        """Execute task delegation to sub-agent."""
        if task.sub_agent_name == "master":
            # Handle locally
            return await self._handle_locally(task)

        sub_agent = self.sub_agents.get(task.sub_agent_name)
        if not sub_agent:
            raise ValueError(f"Sub-agent {task.sub_agent_name} not found")

        # Create message for sub-agent
        message = AgentMessage.user(task.description)

        # Execute with timeout
        result = await asyncio.wait_for(
            sub_agent.process_message(message),
            timeout=self.config.task_timeout_seconds,
        )

        return result.content

    async def _handle_locally(self, task: DelegatedTask) -> str:
        """Handle task locally when no sub-agent is available."""
        # Simple local handling for reasoning tasks
        return f"Analyzed: {task.description}"

    def _store_completed_task(self, task: DelegatedTask) -> None:
        """Store completed task with history limit."""
        self.completed_tasks.append(task)

        # Prune history
        if len(self.completed_tasks) > self.max_completed_history:
            self.completed_tasks = self.completed_tasks[-self.max_completed_history:]

    async def execute_composite_task(
        self,
        subtasks: List[str],
        parallel: bool = False,
    ) -> List[DelegatedTask]:
        """
        Execute a composite task with multiple subtasks.

        Args:
            subtasks: List of subtask descriptions
            parallel: Execute in parallel if True

        Returns:
            List of completed DelegatedTasks
        """
        parent_id = self._generate_task_id()
        results = []

        if parallel:
            # Execute all subtasks concurrently
            tasks = [
                self.delegate_task(desc, parent_task_id=parent_id)
                for desc in subtasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to failed tasks
            results = [
                r if isinstance(r, DelegatedTask) else DelegatedTask(
                    task_id=self._generate_task_id(),
                    task_type=TaskType.REASONING,
                    description="Unknown",
                    status="failed",
                    error=str(r),
                )
                for r in results
            ]
        else:
            # Execute sequentially
            for desc in subtasks:
                result = await self.delegate_task(desc, parent_task_id=parent_id)
                results.append(result)

                # Stop on critical failure
                if result.status == "failed" and result.error:
                    break

        return results

    def aggregate_results(self, tasks: List[DelegatedTask]) -> Dict[str, Any]:
        """
        Aggregate results from multiple tasks.
        Keeps master agent context lean per SYSTEM_PROMPT.
        """
        successful = [t for t in tasks if t.status == "completed"]
        failed = [t for t in tasks if t.status == "failed"]

        # Summarize results (truncate long outputs)
        summaries = []
        for task in successful:
            result_str = str(task.result)[:200] if task.result else ""
            summaries.append({
                "task_id": task.task_id,
                "type": task.task_type.value,
                "summary": result_str,
            })

        return {
            "total_tasks": len(tasks),
            "successful": len(successful),
            "failed": len(failed),
            "summaries": summaries,
            "errors": [{"task_id": t.task_id, "error": t.error} for t in failed],
        }

    def get_context_usage(self) -> Dict[str, Any]:
        """Get context window usage statistics."""
        # Estimate context from active tasks and history
        active_context = sum(
            len(t.description) // 4 for t in self.active_tasks.values()
        )
        history_context = sum(
            len(str(t.result or "")) // 4 for t in self.completed_tasks[-10:]
        )

        total = active_context + history_context

        return {
            "active_tasks_tokens": active_context,
            "history_tokens": history_context,
            "total_tokens": total,
            "limit": self.config.master_context_limit,
            "utilization": total / self.config.master_context_limit,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get master agent status."""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "sub_agents": list(self.sub_agents.keys()),
            "delegation_depth": self._current_delegation_depth,
            "context_usage": self.get_context_usage(),
        }

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process incoming message by delegating to appropriate sub-agent.
        """
        # Parse for task delegation
        task = await self.delegate_task(message.content)

        if task.status == "completed":
            return AgentMessage.assistant(str(task.result))
        else:
            return AgentMessage.assistant(f"Task failed: {task.error}")

    async def run(self) -> None:
        """Main agent loop - process task queue."""
        self._is_running = True
        self.logger.info("MasterAgent started")

        while self._is_running:
            # Process queued tasks
            if self.task_queue:
                task = self.task_queue.pop(0)
                await self.delegate_task(task.description, task.priority)

            await asyncio.sleep(0.1)
