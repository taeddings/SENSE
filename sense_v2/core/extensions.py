"""
SENSE-v2 Extension System
Lifecycle hooks for agent customization and monitoring.

Based on agent-zero extension patterns:
- agent_init: Agent initialization hook
- monologue_start/end: Conversation loop hooks
- message_loop_start/end: Per-message processing hooks
- before_tool_call/after_tool_call: Tool execution hooks
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable, Awaitable, TYPE_CHECKING
from enum import Enum
import logging
import asyncio
import importlib
import os
from pathlib import Path

if TYPE_CHECKING:
    from sense_v2.core.base import BaseAgent
    from sense_v2.core.schemas import AgentMessage, ToolResult


# =============================================================================
# Extension Point Definitions
# =============================================================================

class ExtensionPoint(Enum):
    """Available extension points in the agent lifecycle."""
    AGENT_INIT = "agent_init"
    AGENT_SHUTDOWN = "agent_shutdown"
    MONOLOGUE_START = "monologue_start"
    MONOLOGUE_END = "monologue_end"
    MESSAGE_LOOP_START = "message_loop_start"
    MESSAGE_LOOP_END = "message_loop_end"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"


@dataclass
class ExtensionContext:
    """Context passed to extensions during execution."""
    agent: Optional["BaseAgent"] = None
    message: Optional["AgentMessage"] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional["ToolResult"] = None
    llm_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_message(self, message: "AgentMessage") -> "ExtensionContext":
        """Create a new context with the given message."""
        return ExtensionContext(
            agent=self.agent,
            message=message,
            tool_name=self.tool_name,
            tool_args=self.tool_args,
            tool_result=self.tool_result,
            llm_response=self.llm_response,
            metadata=self.metadata.copy(),
        )

    def with_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Optional["ToolResult"] = None
    ) -> "ExtensionContext":
        """Create a new context with tool information."""
        return ExtensionContext(
            agent=self.agent,
            message=self.message,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=result,
            llm_response=self.llm_response,
            metadata=self.metadata.copy(),
        )


# =============================================================================
# Extension Base Class
# =============================================================================

class Extension:
    """
    Base class for all extensions.

    Extensions are loaded from extension folders and called at specific
    lifecycle points. Each extension can implement one or more hooks.
    """

    # Priority for execution order (lower = earlier)
    priority: int = 50

    def __init__(self, agent: Optional["BaseAgent"] = None, **kwargs):
        self.agent = agent
        self.kwargs = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute(self, context: ExtensionContext, **kwargs) -> Any:
        """
        Execute the extension logic.

        Args:
            context: Extension context with relevant data
            **kwargs: Additional arguments

        Returns:
            Extension-specific result (may be ignored)
        """
        pass

    def supports_point(self, point: ExtensionPoint) -> bool:
        """
        Check if this extension supports a given extension point.

        Override this to specify which points the extension handles.
        By default, extensions support all points.
        """
        return True


# =============================================================================
# Extension Registry and Manager
# =============================================================================

class ExtensionRegistry:
    """
    Registry for discovered extensions.

    Manages extension loading, caching, and lookup.
    """

    _instance: Optional["ExtensionRegistry"] = None
    _extensions: Dict[str, List[Type[Extension]]] = {}
    _cache: Dict[str, List[Type[Extension]]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._extensions = {}
            cls._instance._cache = {}
        return cls._instance

    @classmethod
    def register(cls, extension_point: str):
        """Decorator to register an extension for a specific point."""
        def decorator(ext_class: Type[Extension]):
            if extension_point not in cls._extensions:
                cls._extensions[extension_point] = []
            cls._extensions[extension_point].append(ext_class)
            return ext_class
        return decorator

    @classmethod
    def get_extensions(cls, extension_point: str) -> List[Type[Extension]]:
        """Get all registered extensions for a point."""
        return cls._extensions.get(extension_point, [])

    @classmethod
    def clear(cls):
        """Clear all registered extensions."""
        cls._extensions = {}
        cls._cache = {}


class ExtensionManager:
    """
    Manages extension lifecycle and execution.

    Handles:
    - Loading extensions from folders
    - Executing extensions at lifecycle points
    - Caching loaded extensions
    - Ordering extensions by priority
    """

    def __init__(
        self,
        agent: Optional["BaseAgent"] = None,
        extension_paths: Optional[List[str]] = None,
    ):
        self.agent = agent
        self.extension_paths = extension_paths or []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._loaded_extensions: Dict[str, List[Extension]] = {}

    async def call_extensions(
        self,
        point: ExtensionPoint,
        context: Optional[ExtensionContext] = None,
        **kwargs
    ) -> List[Any]:
        """
        Call all extensions registered for a lifecycle point.

        Args:
            point: The extension point to trigger
            context: Extension context (created if not provided)
            **kwargs: Additional arguments for extensions

        Returns:
            List of results from extensions
        """
        if context is None:
            context = ExtensionContext(agent=self.agent)

        extensions = await self._get_extensions_for_point(point)
        results = []

        for ext in extensions:
            try:
                if ext.supports_point(point):
                    result = await ext.execute(context, **kwargs)
                    results.append(result)
            except Exception as e:
                self.logger.error(
                    f"Extension {ext.__class__.__name__} failed at {point.value}: {e}"
                )

        return results

    async def _get_extensions_for_point(
        self,
        point: ExtensionPoint
    ) -> List[Extension]:
        """Get instantiated extensions for a point."""
        point_name = point.value

        if point_name not in self._loaded_extensions:
            # Load from registry
            ext_classes = ExtensionRegistry.get_extensions(point_name)

            # Load from paths
            for path in self.extension_paths:
                folder = os.path.join(path, point_name)
                if os.path.exists(folder):
                    loaded = await self._load_extensions_from_folder(folder)
                    ext_classes.extend(loaded)

            # Instantiate and sort by priority
            instances = [cls(agent=self.agent) for cls in ext_classes]
            instances.sort(key=lambda e: e.priority)

            self._loaded_extensions[point_name] = instances

        return self._loaded_extensions[point_name]

    async def _load_extensions_from_folder(
        self,
        folder: str
    ) -> List[Type[Extension]]:
        """Load extension classes from a folder."""
        classes = []
        folder_path = Path(folder)

        if not folder_path.exists():
            return classes

        for file in folder_path.glob("*.py"):
            if file.name.startswith("_"):
                continue

            try:
                # Determine module name
                module_name = f"extensions.{folder_path.name}.{file.stem}"

                # Import module
                spec = importlib.util.spec_from_file_location(
                    module_name, file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find Extension subclasses
                    for name, obj in vars(module).items():
                        if (
                            isinstance(obj, type) and
                            issubclass(obj, Extension) and
                            obj is not Extension
                        ):
                            classes.append(obj)

            except Exception as e:
                self.logger.error(f"Failed to load extension from {file}: {e}")

        return classes

    def clear_cache(self):
        """Clear loaded extensions cache."""
        self._loaded_extensions = {}


# =============================================================================
# Built-in Extensions
# =============================================================================

@ExtensionRegistry.register("agent_init")
class LoggingInitExtension(Extension):
    """Extension to log agent initialization."""

    priority = 10

    async def execute(self, context: ExtensionContext, **kwargs) -> None:
        if context.agent:
            self.logger.info(f"Agent {context.agent.name} initialized")


@ExtensionRegistry.register("before_tool_call")
class ToolCallLoggingExtension(Extension):
    """Extension to log tool calls."""

    priority = 10

    async def execute(self, context: ExtensionContext, **kwargs) -> None:
        if context.tool_name:
            self.logger.debug(
                f"Calling tool: {context.tool_name} with args: {context.tool_args}"
            )


@ExtensionRegistry.register("after_tool_call")
class ToolResultLoggingExtension(Extension):
    """Extension to log tool results."""

    priority = 90

    async def execute(self, context: ExtensionContext, **kwargs) -> None:
        if context.tool_result:
            self.logger.debug(
                f"Tool {context.tool_name} returned: {context.tool_result.status}"
            )


@ExtensionRegistry.register("monologue_start")
class MonologueTimingExtension(Extension):
    """Extension to track monologue timing."""

    priority = 5

    async def execute(self, context: ExtensionContext, **kwargs) -> Dict[str, Any]:
        import time
        context.metadata["monologue_start_time"] = time.time()
        return {"started": True}


@ExtensionRegistry.register("monologue_end")
class MonologueEndTimingExtension(Extension):
    """Extension to calculate monologue duration."""

    priority = 95

    async def execute(self, context: ExtensionContext, **kwargs) -> Dict[str, Any]:
        import time
        start_time = context.metadata.get("monologue_start_time")
        if start_time:
            duration = time.time() - start_time
            self.logger.debug(f"Monologue completed in {duration:.2f}s")
            return {"duration_seconds": duration}
        return {}


# =============================================================================
# Convenience Functions
# =============================================================================

async def call_extensions(
    extension_point: str,
    agent: Optional["BaseAgent"] = None,
    **kwargs
) -> Any:
    """
    Convenience function to call extensions at a given point.

    This mimics the agent-zero pattern for easy extension calls.

    Args:
        extension_point: Name of the extension point
        agent: Optional agent instance
        **kwargs: Additional arguments for extensions
    """
    manager = ExtensionManager(agent=agent)

    try:
        point = ExtensionPoint(extension_point)
    except ValueError:
        logging.warning(f"Unknown extension point: {extension_point}")
        return

    context = ExtensionContext(agent=agent, metadata=kwargs)
    return await manager.call_extensions(point, context, **kwargs)


def register_extension(
    extension_point: str,
    extension_class: Type[Extension]
) -> None:
    """
    Register an extension class for a given point.

    Args:
        extension_point: Name of the extension point
        extension_class: Extension class to register
    """
    if extension_point not in ExtensionRegistry._extensions:
        ExtensionRegistry._extensions[extension_point] = []
    ExtensionRegistry._extensions[extension_point].append(extension_class)


# =============================================================================
# Agent Mixin for Extension Support
# =============================================================================

class ExtensionMixin:
    """
    Mixin class to add extension support to agents.

    Add this mixin to an agent class to enable lifecycle hooks.
    """

    _extension_manager: Optional[ExtensionManager] = None

    def init_extensions(
        self,
        extension_paths: Optional[List[str]] = None
    ) -> None:
        """Initialize the extension manager."""
        self._extension_manager = ExtensionManager(
            agent=self,  # type: ignore
            extension_paths=extension_paths or [],
        )

    async def call_extension(
        self,
        point: ExtensionPoint,
        **kwargs
    ) -> List[Any]:
        """Call extensions at a lifecycle point."""
        if self._extension_manager is None:
            self.init_extensions()

        context = ExtensionContext(
            agent=self,  # type: ignore
            metadata=kwargs,
        )

        return await self._extension_manager.call_extensions(point, context, **kwargs)

    async def on_agent_init(self) -> None:
        """Hook called during agent initialization."""
        await self.call_extension(ExtensionPoint.AGENT_INIT)

    async def on_agent_shutdown(self) -> None:
        """Hook called during agent shutdown."""
        await self.call_extension(ExtensionPoint.AGENT_SHUTDOWN)

    async def on_monologue_start(self) -> None:
        """Hook called at start of monologue/conversation loop."""
        await self.call_extension(ExtensionPoint.MONOLOGUE_START)

    async def on_monologue_end(self) -> None:
        """Hook called at end of monologue/conversation loop."""
        await self.call_extension(ExtensionPoint.MONOLOGUE_END)

    async def on_message_loop_start(self, message: "AgentMessage") -> None:
        """Hook called at start of message processing."""
        await self.call_extension(
            ExtensionPoint.MESSAGE_LOOP_START,
            message=message,
        )

    async def on_message_loop_end(self, message: "AgentMessage") -> None:
        """Hook called at end of message processing."""
        await self.call_extension(
            ExtensionPoint.MESSAGE_LOOP_END,
            message=message,
        )

    async def on_before_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> None:
        """Hook called before tool execution."""
        await self.call_extension(
            ExtensionPoint.BEFORE_TOOL_CALL,
            tool_name=tool_name,
            tool_args=tool_args,
        )

    async def on_after_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: "ToolResult"
    ) -> None:
        """Hook called after tool execution."""
        await self.call_extension(
            ExtensionPoint.AFTER_TOOL_CALL,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=result,
        )
