"""
SENSE-v2 Base Classes
Abstract base classes for Agents, Tools, and Memory systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable
import logging
import asyncio
from datetime import datetime
import traceback

from sense_v2.core.schemas import (
    ToolSchema,
    ToolResult,
    ToolResultStatus,
    AgentMessage,
    MessageRole,
    RewardSignal,
)


class BaseTool(ABC):
    """
    Abstract base class for all SENSE-v2 tools.

    Per SYSTEM_PROMPT requirements:
    - Every high-level function must be exposed as a Schema-based Python Tool
    - Includes feedback mechanism for self-correction (stderr parsing)
    - Agents interact via tools, not direct script execution
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._execution_count = 0
        self._success_count = 0
        self._last_result: Optional[ToolResult] = None

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the tool's schema definition."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    async def execute_with_retry(
        self,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute tool with automatic retry on recoverable errors.
        Implements self-correction loop by parsing stderr.
        """
        retries = max_retries or self.schema.max_retries
        last_result = None

        for attempt in range(retries + 1):
            try:
                result = await self.execute(**kwargs)
                result.retry_count = attempt
                self._last_result = result
                self._execution_count += 1

                if result.is_success:
                    self._success_count += 1
                    return result

                # Check if we should retry based on stderr analysis
                if result.should_retry and attempt < retries:
                    self.logger.warning(
                        f"Tool {self.schema.name} failed (attempt {attempt + 1}/{retries + 1}), "
                        f"retrying... Error: {result.error}"
                    )
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    last_result = result
                    continue

                return result

            except Exception as e:
                self.logger.error(f"Tool {self.schema.name} exception: {e}")
                last_result = ToolResult.error(
                    str(e),
                    stderr=traceback.format_exc(),
                    retry_count=attempt
                )
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return last_result

        return last_result or ToolResult.error("Max retries exceeded")

    def validate_input(self, **kwargs) -> List[str]:
        """Validate input parameters against schema."""
        return self.schema.validate_input(kwargs)

    @property
    def success_rate(self) -> float:
        """Calculate tool success rate."""
        if self._execution_count == 0:
            return 0.0
        return self._success_count / self._execution_count


class BaseAgent(ABC):
    """
    Abstract base class for all SENSE-v2 agents.

    Per SYSTEM_PROMPT requirements:
    - Agents must interact with system via tools, not direct script execution
    - MasterAgent must never perform heavy computation (delegates to sub-agents)
    - Includes self-correction loop
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        self.name = name
        self.config = config
        self.tools: Dict[str, BaseTool] = {}
        self.logger = logging.getLogger(f"Agent.{name}")
        self.message_history: List[AgentMessage] = []
        self._is_running = False

        # Register tools
        if tools:
            for tool in tools:
                self.register_tool(tool)

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool for this agent to use."""
        self.tools[tool.schema.name] = tool
        self.logger.debug(f"Registered tool: {tool.schema.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a registered tool by name."""
        return self.tools.get(name)

    def get_available_tools(self) -> List[ToolSchema]:
        """Get schemas for all available tools."""
        return [tool.schema for tool in self.tools.values()]

    async def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Call a registered tool by name.
        Implements self-correction loop on failure.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult.error(f"Tool '{tool_name}' not found")

        # Validate input
        errors = tool.validate_input(**kwargs)
        if errors:
            return ToolResult.error(f"Validation errors: {', '.join(errors)}")

        # Execute with retry
        return await tool.execute_with_retry(**kwargs)

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process an incoming message and generate a response."""
        pass

    @abstractmethod
    async def run(self) -> None:
        """Main agent execution loop."""
        pass

    async def think(self, context: str) -> str:
        """
        Generate agent's internal reasoning/planning.
        Override in subclasses for specific thinking patterns.
        """
        return ""

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the conversation history."""
        self.message_history.append(message)

    def get_context_tokens(self) -> int:
        """Estimate total tokens in conversation context."""
        return sum(msg.token_estimate() for msg in self.message_history)

    def should_prune_context(self, threshold_ratio: float = 0.8) -> bool:
        """
        Check if context should be pruned.
        Per SYSTEM_PROMPT: Trigger summarize-and-prune at 80% of model limit.
        """
        from sense_v2.core.config import Config
        if self.config and hasattr(self.config, 'orchestration'):
            limit = self.config.orchestration.master_context_limit
        else:
            limit = 4096

        current = self.get_context_tokens()
        return current > (limit * threshold_ratio)

    def clear_history(self) -> None:
        """Clear message history."""
        self.message_history = []


class BaseMemory(ABC):
    """
    Abstract base class for SENSE-v2 memory systems.

    Per SYSTEM_PROMPT requirements (AgeMem - The Filing Cabinet):
    - Structured knowledge persistence
    - Vector database for LTM
    - Summarize-and-Prune hook at 80% context limit
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def store(self, key: str, value: Any, metadata: Optional[Dict] = None) -> bool:
        """Store a value in memory."""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory."""
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search memory using semantic similarity."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        pass

    @abstractmethod
    async def summarize_and_prune(self, content: str, target_ratio: float = 0.3) -> str:
        """
        Summarize content and prune to target ratio.
        Triggered when active chat context exceeds 80% of model's limit.
        """
        pass

    @abstractmethod
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        pass


@dataclass
class AgentState:
    """
    State tracking for agents.
    Used for dev_log.json to track evolutionary progress and system health.
    """
    agent_name: str
    generation: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    current_fitness: float = 0.0
    best_fitness: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.total_tasks_completed + self.total_tasks_failed
        if total == 0:
            return 0.0
        return self.total_tasks_completed / total

    def update_fitness(self, reward: RewardSignal) -> None:
        """Update fitness based on reward signal."""
        self.current_fitness = reward.value
        if reward.value > self.best_fitness:
            self.best_fitness = reward.value
        self.last_activity = datetime.now()

    def record_task(self, success: bool) -> None:
        """Record a task completion."""
        if success:
            self.total_tasks_completed += 1
        else:
            self.total_tasks_failed += 1
        self.last_activity = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "agent_name": self.agent_name,
            "generation": self.generation,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "current_fitness": self.current_fitness,
            "best_fitness": self.best_fitness,
            "success_rate": self.success_rate,
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Deserialize state from dictionary."""
        last_activity = data.get("last_activity")
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)
        else:
            last_activity = datetime.now()

        return cls(
            agent_name=data["agent_name"],
            generation=data.get("generation", 0),
            total_tasks_completed=data.get("total_tasks_completed", 0),
            total_tasks_failed=data.get("total_tasks_failed", 0),
            current_fitness=data.get("current_fitness", 0.0),
            best_fitness=data.get("best_fitness", 0.0),
            last_activity=last_activity,
            metadata=data.get("metadata", {}),
        )


class ToolRegistry:
    """
    Global registry for all SENSE-v2 tools.
    Ensures tools are schema-based and properly registered.
    """

    _instance = None
    _tools: Dict[str, Type[BaseTool]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """
        Decorator to register a tool class.
        Usage: @ToolRegistry.register
        """
        instance = tool_class()
        cls._tools[instance.schema.name] = tool_class
        return tool_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseTool]]:
        """Get a registered tool class by name."""
        return cls._tools.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered tool names."""
        return list(cls._tools.keys())

    @classmethod
    def create_instance(cls, name: str, config: Optional[Any] = None) -> Optional[BaseTool]:
        """Create an instance of a registered tool."""
        tool_class = cls.get(name)
        if tool_class:
            return tool_class(config)
        return None
