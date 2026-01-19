"""
SENSE-v2 Adaptive Reasoning Budget
Dynamic token budget allocation based on task complexity and resource constraints.

Part of Sprint 2: The Brain

Features:
- Task complexity-based allocation
- VRAM monitoring and guards
- Sensor status compensation
- Memory-based efficiency optimization
- Reasoning mode selection
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ReasoningMode(Enum):
    """
    Reasoning modes based on drift level and resource availability.

    EFFICIENT: Low drift, high memory similarity - minimal reasoning
    BALANCED: Moderate conditions - standard reasoning
    EXPLORATORY: High drift, novel problems - extended reasoning
    COMPENSATORY: Sensor offline - doubled verification
    """
    EFFICIENT = "efficient"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"
    COMPENSATORY = "compensatory"


@dataclass
class BudgetAllocation:
    """Result of budget allocation decision."""
    tokens: int
    mode: ReasoningMode
    verification_depth: int
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "mode": self.mode.value,
            "verification_depth": self.verification_depth,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


@dataclass
class ResourceStatus:
    """Current resource availability status."""
    vram_percent: float = 0.0
    vram_available_mb: int = 0
    ram_percent: float = 0.0
    ram_available_mb: int = 0
    sensor_status: str = "ONLINE"
    psutil_available: bool = False

    @classmethod
    def get_current(cls) -> "ResourceStatus":
        """Get current resource status."""
        if not PSUTIL_AVAILABLE:
            return cls(psutil_available=False, sensor_status="UNKNOWN")

        try:
            mem = psutil.virtual_memory()
            return cls(
                vram_percent=mem.percent,  # Using RAM as proxy for VRAM
                vram_available_mb=mem.available // (1024 * 1024),
                ram_percent=mem.percent,
                ram_available_mb=mem.available // (1024 * 1024),
                sensor_status="ONLINE",
                psutil_available=True,
            )
        except Exception:
            return cls(psutil_available=False, sensor_status="ERROR")


class AdaptiveReasoningBudget:
    """
    Adaptive reasoning budget allocator.

    Dynamically adjusts the token budget for reasoning ("thinking" steps)
    based on:
    - Task complexity
    - Available VRAM/memory
    - Sensor status
    - Memory retrieval results

    Compensatory Check:
        If sensor_status == "OFFLINE":
        - Double verification depth (min base_tokens * 2.0, capped by VRAM)

    Resource Guard:
        If VRAM > 90%:
        - Hard-cap at MIN_THRESHOLD or trigger CPU offload

    Memory Integration:
        If high-similarity solution exists (>0.95):
        - Reduce budget by 30%

    Example:
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate_tokens(
            task_complexity=0.7,
            available_vram=8192,
            sensor_status="ONLINE",
        )
        print(f"Allocated {allocation.tokens} tokens in {allocation.mode.value} mode")
    """

    # Configuration constants
    DEFAULT_BASE_BUDGET: int = 1024
    MIN_THRESHOLD: int = 256
    MAX_THRESHOLD: int = 8192

    # Resource thresholds
    VRAM_WARNING_THRESHOLD: float = 0.75
    VRAM_CRITICAL_THRESHOLD: float = 0.90

    # Efficiency thresholds
    HIGH_SIMILARITY_THRESHOLD: float = 0.95
    EFFICIENCY_REDUCTION: float = 0.30

    # Compensatory multiplier for sensor offline
    COMPENSATORY_MULTIPLIER: float = 2.0

    def __init__(
        self,
        base_budget: int = DEFAULT_BASE_BUDGET,
        min_threshold: int = MIN_THRESHOLD,
        max_threshold: int = MAX_THRESHOLD,
        compensatory_multiplier: float = COMPENSATORY_MULTIPLIER,
        efficiency_reduction: float = EFFICIENCY_REDUCTION,
    ):
        """
        Initialize the adaptive reasoning budget allocator.

        Args:
            base_budget: Default token budget
            min_threshold: Minimum allowed budget
            max_threshold: Maximum allowed budget
            compensatory_multiplier: Multiplier when sensor is offline
            efficiency_reduction: Reduction ratio when high similarity found
        """
        self.base_budget = base_budget
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.compensatory_multiplier = compensatory_multiplier
        self.efficiency_reduction = efficiency_reduction
        self.logger = logging.getLogger(self.__class__.__name__)

        # Tracking
        self._allocation_history: List[BudgetAllocation] = []
        self._total_tokens_allocated: int = 0

    def allocate_tokens(
        self,
        task_complexity: float,
        available_vram: Optional[int] = None,
        sensor_status: str = "ONLINE",
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> BudgetAllocation:
        """
        Allocate tokens for reasoning based on current conditions.

        Args:
            task_complexity: Task complexity score (0.0 to 1.0)
            available_vram: Available VRAM in MB (auto-detected if None)
            sensor_status: Sensor status ("ONLINE", "OFFLINE", "DEGRADED")
            memory_context: Optional context from memory retrieval

        Returns:
            BudgetAllocation with tokens, mode, and rationale
        """
        # Get resource status
        if available_vram is None:
            resources = ResourceStatus.get_current()
            available_vram = resources.vram_available_mb
            vram_percent = resources.vram_percent
        else:
            # Estimate percent from available (assume 16GB total for estimation)
            vram_percent = max(0, (1 - available_vram / 16384)) * 100

        # Start with base budget scaled by complexity
        budget = int(self.base_budget * (0.5 + task_complexity))
        rationale_parts = [f"Base budget: {self.base_budget}, complexity: {task_complexity:.2f}"]

        # Determine reasoning mode
        drift_level = self._get_drift_level(memory_context)
        mode = self.get_reasoning_mode(drift_level)

        verification_depth = 1  # Default

        # Compensatory Check: Sensor offline
        if sensor_status.upper() == "OFFLINE":
            mode = ReasoningMode.COMPENSATORY
            budget = int(budget * self.compensatory_multiplier)
            verification_depth = max(2, verification_depth * 2)
            rationale_parts.append(
                f"Sensor OFFLINE: doubled budget ({self.compensatory_multiplier}x), "
                f"verification depth: {verification_depth}"
            )

        # Resource Guard: VRAM critical
        if vram_percent >= self.VRAM_CRITICAL_THRESHOLD * 100:
            original_budget = budget
            budget = self.min_threshold
            rationale_parts.append(
                f"VRAM CRITICAL ({vram_percent:.1f}%): capped at MIN_THRESHOLD ({budget})"
            )
            self.logger.warning(
                f"VRAM critical: reduced budget from {original_budget} to {budget}"
            )

        elif vram_percent >= self.VRAM_WARNING_THRESHOLD * 100:
            reduction = 0.5
            original_budget = budget
            budget = int(budget * reduction)
            rationale_parts.append(
                f"VRAM warning ({vram_percent:.1f}%): reduced by {int((1-reduction)*100)}%"
            )

        # Memory Integration: High similarity reduction
        if memory_context:
            max_similarity = memory_context.get("max_similarity", 0.0)
            if max_similarity >= self.HIGH_SIMILARITY_THRESHOLD:
                original_budget = budget
                budget = int(budget * (1 - self.efficiency_reduction))
                rationale_parts.append(
                    f"High similarity ({max_similarity:.2f}): reduced by "
                    f"{int(self.efficiency_reduction * 100)}%"
                )

        # Apply mode-specific adjustments
        if mode == ReasoningMode.EFFICIENT:
            budget = int(budget * 0.7)
            rationale_parts.append("EFFICIENT mode: -30%")
        elif mode == ReasoningMode.EXPLORATORY:
            budget = int(budget * 1.3)
            verification_depth = max(2, verification_depth)
            rationale_parts.append("EXPLORATORY mode: +30%, min 2 verification loops")

        # Clamp to bounds
        budget = max(self.min_threshold, min(self.max_threshold, budget))

        allocation = BudgetAllocation(
            tokens=budget,
            mode=mode,
            verification_depth=verification_depth,
            rationale=" | ".join(rationale_parts),
            metadata={
                "task_complexity": task_complexity,
                "vram_percent": vram_percent,
                "sensor_status": sensor_status,
                "drift_level": drift_level,
                "high_similarity_found": (
                    memory_context.get("max_similarity", 0) >= self.HIGH_SIMILARITY_THRESHOLD
                    if memory_context else False
                ),
            },
        )

        # Track allocation
        self._allocation_history.append(allocation)
        self._total_tokens_allocated += budget

        return allocation

    def get_reasoning_mode(self, drift_level: float) -> ReasoningMode:
        """
        Determine reasoning mode based on drift level.

        Args:
            drift_level: Current concept drift level (0.0 to 1.0)

        Returns:
            Appropriate ReasoningMode
        """
        if drift_level < 0.3:
            return ReasoningMode.EFFICIENT
        elif drift_level < 0.7:
            return ReasoningMode.BALANCED
        else:
            return ReasoningMode.EXPLORATORY

    def _get_drift_level(self, memory_context: Optional[Dict[str, Any]]) -> float:
        """
        Extract or estimate drift level from memory context.

        Args:
            memory_context: Memory retrieval context

        Returns:
            Drift level (0.0 to 1.0)
        """
        if not memory_context:
            return 0.5  # Default: balanced

        # Check for explicit drift metric
        if "drift_level" in memory_context:
            return memory_context["drift_level"]

        # Estimate from similarity
        max_similarity = memory_context.get("max_similarity", 0.5)
        # Low similarity = high drift (novel problem)
        return 1.0 - max_similarity

    def get_vram_status(self) -> Dict[str, Any]:
        """
        Get current VRAM/memory status.

        Returns:
            Dictionary with memory statistics
        """
        resources = ResourceStatus.get_current()

        status = "NORMAL"
        if resources.vram_percent >= self.VRAM_CRITICAL_THRESHOLD * 100:
            status = "CRITICAL"
        elif resources.vram_percent >= self.VRAM_WARNING_THRESHOLD * 100:
            status = "WARNING"

        return {
            "status": status,
            "vram_percent": resources.vram_percent,
            "vram_available_mb": resources.vram_available_mb,
            "ram_percent": resources.ram_percent,
            "ram_available_mb": resources.ram_available_mb,
            "psutil_available": resources.psutil_available,
            "warning_threshold": self.VRAM_WARNING_THRESHOLD,
            "critical_threshold": self.VRAM_CRITICAL_THRESHOLD,
        }

    def should_offload_to_cpu(self) -> Tuple[bool, str]:
        """
        Determine if computation should be offloaded to CPU.

        Returns:
            Tuple of (should_offload, reason)
        """
        resources = ResourceStatus.get_current()

        if resources.vram_percent >= self.VRAM_CRITICAL_THRESHOLD * 100:
            return True, f"VRAM critical: {resources.vram_percent:.1f}%"

        return False, "VRAM OK"

    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get statistics about budget allocations."""
        if not self._allocation_history:
            return {"status": "no_allocations"}

        allocations = self._allocation_history
        tokens = [a.tokens for a in allocations]
        modes = [a.mode.value for a in allocations]

        from collections import Counter
        mode_counts = Counter(modes)

        return {
            "total_allocations": len(allocations),
            "total_tokens_allocated": self._total_tokens_allocated,
            "average_tokens": sum(tokens) / len(tokens),
            "min_tokens": min(tokens),
            "max_tokens": max(tokens),
            "mode_distribution": dict(mode_counts),
            "recent_allocations": [a.to_dict() for a in allocations[-5:]],
        }

    def reset_tracking(self) -> None:
        """Reset allocation tracking."""
        self._allocation_history = []
        self._total_tokens_allocated = 0


# Convenience functions
def allocate_reasoning_budget(
    task_complexity: float,
    sensor_status: str = "ONLINE",
    memory_context: Optional[Dict[str, Any]] = None,
) -> BudgetAllocation:
    """
    Convenience function to allocate reasoning budget.

    Args:
        task_complexity: Task complexity (0-1)
        sensor_status: Sensor status
        memory_context: Memory retrieval context

    Returns:
        BudgetAllocation
    """
    allocator = AdaptiveReasoningBudget()
    return allocator.allocate_tokens(
        task_complexity=task_complexity,
        sensor_status=sensor_status,
        memory_context=memory_context,
    )
