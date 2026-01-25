"""
SENSE-v2 Memory Tools
Schema-based tools for AgeMem memory operations and system memory profiling.
"""

from typing import Any, Dict, List, Optional
import logging

from sense.core.base import BaseTool, ToolRegistry
from sense.core.schemas import ToolSchema, ToolParameter, ToolResult

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@ToolRegistry.register
class MemoryStoreTool(BaseTool):
    """Tool for storing information in memory."""

    def __init__(self, config: Optional[Any] = None, memory_system: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._memory = memory_system

    def set_memory_system(self, memory_system: Any) -> None:
        """Set the memory system to use."""
        self._memory = memory_system

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_store",
            description="Store information in the agent's memory system",
            parameters=[
                ToolParameter(
                    name="key",
                    param_type="string",
                    description="Unique key for the memory entry",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    param_type="string",
                    description="Content to store",
                    required=True,
                ),
                ToolParameter(
                    name="memory_type",
                    param_type="string",
                    description="Type of memory: stm, ltm, or auto",
                    required=False,
                    default="auto",
                    enum=["stm", "ltm", "auto"],
                ),
                ToolParameter(
                    name="priority",
                    param_type="float",
                    description="Priority for retention (0.0-1.0)",
                    required=False,
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ToolParameter(
                    name="persist",
                    param_type="boolean",
                    description="Force persistence to long-term memory",
                    required=False,
                    default=False,
                ),
            ],
            returns="object",
            returns_description="Store result with key and status",
            category="memory",
            max_retries=1,
        )

    async def execute(
        self,
        key: str,
        content: str,
        memory_type: str = "auto",
        priority: float = 0.5,
        persist: bool = False,
        **kwargs
    ) -> ToolResult:
        """Store content in memory."""
        if self._memory is None:
            return ToolResult.error("Memory system not initialized")

        try:
            from sense.memory.agemem import MemoryType

            mem_type_map = {
                "stm": MemoryType.STM,
                "ltm": MemoryType.LTM,
                "auto": MemoryType.AUTO,
            }

            success = await self._memory.store(
                key=key,
                value=content,
                memory_type=mem_type_map.get(memory_type, MemoryType.AUTO),
                priority=priority,
                persist=persist,
            )

            if success:
                return ToolResult.success({
                    "key": key,
                    "stored": True,
                    "memory_type": memory_type,
                    "priority": priority,
                })
            else:
                return ToolResult.error("Failed to store in memory")

        except Exception as e:
            self.logger.error(f"Memory store failed: {e}")
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemorySearchTool(BaseTool):
    """Tool for searching memory."""

    def __init__(self, config: Optional[Any] = None, memory_system: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._memory = memory_system

    def set_memory_system(self, memory_system: Any) -> None:
        """Set the memory system to use."""
        self._memory = memory_system

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_search",
            description="Search the agent's memory using semantic similarity",
            parameters=[
                ToolParameter(
                    name="query",
                    param_type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    param_type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=20,
                ),
                ToolParameter(
                    name="memory_type",
                    param_type="string",
                    description="Type of memory to search: stm, ltm, or auto",
                    required=False,
                    default="auto",
                    enum=["stm", "ltm", "auto"],
                ),
                ToolParameter(
                    name="threshold",
                    param_type="float",
                    description="Minimum similarity threshold",
                    required=False,
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            returns="array",
            returns_description="List of matching memory entries with similarity scores",
            category="memory",
            max_retries=1,
        )

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str = "auto",
        threshold: float = 0.5,
        **kwargs
    ) -> ToolResult:
        """Search memory."""
        if self._memory is None:
            return ToolResult.error("Memory system not initialized")

        try:
            from sense.memory.agemem import MemoryType

            mem_type_map = {
                "stm": MemoryType.STM,
                "ltm": MemoryType.LTM,
                "auto": MemoryType.AUTO,
            }

            results = await self._memory.search(
                query=query,
                top_k=top_k,
                memory_type=mem_type_map.get(memory_type, MemoryType.AUTO),
                threshold=threshold,
            )

            return ToolResult.success(
                results,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "memory_type": memory_type,
                },
            )

        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemoryRetrieveTool(BaseTool):
    """Tool for retrieving specific memory entry by key."""

    def __init__(self, config: Optional[Any] = None, memory_system: Optional[Any] = None):
        super().__init__(config)
        self._memory = memory_system

    def set_memory_system(self, memory_system: Any) -> None:
        self._memory = memory_system

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_retrieve",
            description="Retrieve a specific memory entry by key",
            parameters=[
                ToolParameter(
                    name="key",
                    param_type="string",
                    description="Key of the memory entry to retrieve",
                    required=True,
                ),
            ],
            returns="object",
            returns_description="Memory entry content or null",
            category="memory",
            max_retries=0,
        )

    async def execute(self, key: str, **kwargs) -> ToolResult:
        """Retrieve memory by key."""
        if self._memory is None:
            return ToolResult.error("Memory system not initialized")

        try:
            content = await self._memory.retrieve(key)

            if content is not None:
                return ToolResult.success({
                    "key": key,
                    "content": content,
                    "found": True,
                })
            else:
                return ToolResult.success({
                    "key": key,
                    "content": None,
                    "found": False,
                })

        except Exception as e:
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemoryStatsTool(BaseTool):
    """Tool for getting memory usage statistics."""

    def __init__(self, config: Optional[Any] = None, memory_system: Optional[Any] = None):
        super().__init__(config)
        self._memory = memory_system

    def set_memory_system(self, memory_system: Any) -> None:
        self._memory = memory_system

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_stats",
            description="Get memory system usage statistics",
            parameters=[],
            returns="object",
            returns_description="Memory usage statistics",
            category="memory",
            max_retries=0,
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Get memory stats."""
        if self._memory is None:
            return ToolResult.error("Memory system not initialized")

        try:
            stats = self._memory.get_usage_stats()
            return ToolResult.success(stats)
        except Exception as e:
            return ToolResult.error(str(e))


@ToolRegistry.register
class MemoryProfileTool(BaseTool):
    """
    Tool for profiling VRAM/RAM usage and providing optimization recommendations.

    Returns comprehensive memory statistics including:
    - System RAM usage and availability
    - GPU VRAM usage (if available)
    - Memory pressure indicators
    - Recommendations for enabling memory-efficient features
    """

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_profile",
            description="Profile system memory (RAM/VRAM) and get optimization recommendations",
            parameters=[
                ToolParameter(
                    name="include_gpu",
                    param_type="boolean",
                    description="Include GPU memory statistics",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_recommendations",
                    param_type="boolean",
                    description="Include optimization recommendations",
                    required=False,
                    default=True,
                ),
            ],
            returns="object",
            returns_description="Memory profile with usage stats and recommendations",
            category="memory",
            max_retries=0,
        )

    async def execute(
        self,
        include_gpu: bool = True,
        include_recommendations: bool = True,
        **kwargs
    ) -> ToolResult:
        """Profile system memory and provide recommendations."""
        try:
            profile = {}

            # System RAM stats
            if PSUTIL_AVAILABLE:
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()

                profile["ram"] = {
                    "total_mb": mem.total // (1024 * 1024),
                    "available_mb": mem.available // (1024 * 1024),
                    "used_mb": mem.used // (1024 * 1024),
                    "percent_used": mem.percent,
                    "swap_total_mb": swap.total // (1024 * 1024),
                    "swap_used_mb": swap.used // (1024 * 1024),
                    "swap_percent": swap.percent,
                }

                # Memory pressure level
                ram_percent = mem.percent / 100.0
                if ram_percent < 0.60:
                    profile["memory_pressure"] = "low"
                elif ram_percent < 0.75:
                    profile["memory_pressure"] = "moderate"
                elif ram_percent < 0.90:
                    profile["memory_pressure"] = "high"
                else:
                    profile["memory_pressure"] = "critical"
            else:
                profile["ram"] = {"error": "psutil not available"}
                profile["memory_pressure"] = "unknown"

            # GPU stats
            if include_gpu and TORCH_AVAILABLE:
                profile["gpu"] = {}

                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    profile["gpu"]["device_count"] = device_count
                    profile["gpu"]["devices"] = []

                    for i in range(device_count):
                        device_props = torch.cuda.get_device_properties(i)
                        mem_allocated = torch.cuda.memory_allocated(i)
                        mem_reserved = torch.cuda.memory_reserved(i)
                        mem_total = device_props.total_memory

                        profile["gpu"]["devices"].append({
                            "index": i,
                            "name": device_props.name,
                            "total_mb": mem_total // (1024 * 1024),
                            "allocated_mb": mem_allocated // (1024 * 1024),
                            "reserved_mb": mem_reserved // (1024 * 1024),
                            "free_mb": (mem_total - mem_reserved) // (1024 * 1024),
                            "utilization_percent": (mem_allocated / mem_total) * 100,
                        })

                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    profile["gpu"]["backend"] = "mps"
                    profile["gpu"]["available"] = True
                    # MPS doesn't expose detailed memory stats
                    profile["gpu"]["note"] = "MPS memory stats not available"
                else:
                    profile["gpu"]["available"] = False
                    profile["gpu"]["backend"] = "none"
            elif include_gpu:
                profile["gpu"] = {"error": "torch not available"}

            # Recommendations
            if include_recommendations:
                recommendations = []

                pressure = profile.get("memory_pressure", "unknown")

                if pressure == "moderate":
                    recommendations.append({
                        "action": "activate_engram",
                        "reason": "Memory usage above 60%, Engram can reduce compute by caching",
                        "priority": "medium",
                    })
                elif pressure == "high":
                    recommendations.append({
                        "action": "activate_engram",
                        "reason": "Memory usage above 75%, Engram recommended",
                        "priority": "high",
                    })
                    recommendations.append({
                        "action": "use_fused_kernels",
                        "reason": "Memory pressure high, fused kernels can save ~50-84% memory",
                        "priority": "high",
                    })
                elif pressure == "critical":
                    recommendations.append({
                        "action": "activate_engram",
                        "reason": "Critical memory pressure, Engram required",
                        "priority": "critical",
                    })
                    recommendations.append({
                        "action": "use_fused_kernels",
                        "reason": "Critical memory pressure, fused kernels required",
                        "priority": "critical",
                    })
                    recommendations.append({
                        "action": "reduce_batch_size",
                        "reason": "Consider reducing batch size to avoid OOM",
                        "priority": "critical",
                    })

                # GPU-specific recommendations
                if "gpu" in profile and "devices" in profile["gpu"]:
                    for device in profile["gpu"]["devices"]:
                        if device.get("utilization_percent", 0) > 80:
                            recommendations.append({
                                "action": "enable_gradient_checkpointing",
                                "reason": f"GPU {device['index']} utilization > 80%",
                                "priority": "high",
                            })

                profile["recommendations"] = recommendations

            return ToolResult.success(profile)

        except Exception as e:
            self.logger.error(f"Memory profiling failed: {e}")
            return ToolResult.error(str(e))
