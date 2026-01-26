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
- Automatic complexity estimation (Context Engineering integration)
- Adaptive retrieval depth gating

Updates:
- Added estimate_complexity() for automatic prompt analysis
- Added calculate_retrieval_depth() for memory query optimization
- Added allocate() method as primary entry point
- BudgetAllocation now includes memory_k and complexity_estimate
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import re
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
    memory_k: int = 3  # Retrieval depth for memory queries
    complexity_estimate: float = 0.5  # Stored complexity score
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "mode": self.mode.value,
            "verification_depth": self.verification_depth,
            "memory_k": self.memory_k,
            "complexity_estimate": self.complexity_estimate,
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


def estimate_complexity(prompt: str) -> float:
    """
    Estimate task complexity using cheap heuristics (no LLM call).
    
    Factors considered:
    - Entity density (proper nouns, technical terms)
    - Question structure (analytical vs. factual)
    - Length as proxy for scope
    - Multi-step indicators
    - Code/technical content markers
    
    Args:
        prompt: The task prompt or query
        
    Returns:
        Complexity score from 0.0 (trivial) to 1.0 (highly complex)
    """
    if not prompt or not prompt.strip():
        return 0.1
    
    prompt_lower = prompt.lower()
    words = prompt.split()
    word_count = len(words)
    
    score = 0.0
    
    # --- Entity density (proper nouns, capitalized terms) ---
    # Simple heuristic: words starting with uppercase that aren't sentence starters
    capitalized = [w for w in words[1:] if w and w[0].isupper()]
    entity_ratio = len(capitalized) / max(word_count, 1)
    score += min(entity_ratio * 1.5, 0.25)
    
    # --- Question structure ---
    # Analytical questions are more complex
    analytical_keywords = {
        "compare", "contrast", "analyze", "evaluate", "assess",
        "synthesize", "critique", "differentiate", "justify",
        "implications", "consequences", "tradeoffs", "trade-offs",
    }
    if any(kw in prompt_lower for kw in analytical_keywords):
        score += 0.20
    
    # Explanatory questions are medium complexity
    explanatory_keywords = {"why", "how", "explain", "describe", "elaborate"}
    if any(kw in prompt_lower for kw in explanatory_keywords):
        score += 0.12
    
    # Factual questions are lower complexity
    factual_keywords = {"what is", "who is", "when did", "where is", "define"}
    if any(kw in prompt_lower for kw in factual_keywords):
        score += 0.05
    
    # --- Length proxy for scope ---
    # Longer prompts typically indicate more complex tasks
    length_score = min(word_count / 150, 0.20)
    score += length_score
    
    # --- Multi-step indicators ---
    multi_step_keywords = {
        "then", "after", "first", "second", "finally", "next",
        "steps", "step by step", "process", "procedure", "sequence",
        "followed by", "before", "once", "subsequently",
    }
    multi_step_count = sum(1 for kw in multi_step_keywords if kw in prompt_lower)
    score += min(multi_step_count * 0.08, 0.20)
    
    # --- Code/technical markers ---
    code_indicators = {
        "```", "def ", "class ", "function", "import ", "return ",
        "error", "bug", "debug", "traceback", "exception",
        "implement", "refactor", "optimize", "algorithm",
    }
    if any(ind in prompt_lower or ind in prompt for ind in code_indicators):
        score += 0.15
    
    # --- Negation/constraint complexity ---
    constraint_keywords = {
        "without", "except", "unless", "but not", "excluding",
        "must not", "cannot", "shouldn't", "avoid", "constraint",
    }
    if any(kw in prompt_lower for kw in constraint_keywords):
        score += 0.10
    
    # --- List/enumeration requests ---
    list_patterns = [
        r'\d+\s+(things|items|points|reasons|ways|examples)',
        r'list\s+(all|the|some)',
        r'give me\s+\d+',
    ]
    for pattern in list_patterns:
        if re.search(pattern, prompt_lower):
            score += 0.08
            break
    
    # --- Domain complexity markers ---
    domain_markers = {
        # Technical domains
        "architecture", "infrastructure", "distributed", "concurrent",
        "async", "synchronization", "latency", "throughput",
        # Analytical domains  
        "hypothesis", "correlation", "regression", "statistical",
        "probability", "variance", "confidence interval",
        # Business domains
        "stakeholder", "requirements", "specification", "deliverable",
    }
    domain_hits = sum(1 for m in domain_markers if m in prompt_lower)
    score += min(domain_hits * 0.05, 0.15)
    
    return min(score, 1.0)


def calculate_retrieval_depth(
    complexity: float,
    base_k: int = 3,
    max_k: int = 10,
    min_k: int = 1,
) -> int:
    """
    Calculate adaptive retrieval depth based on query complexity.
    
    Simple tasks get fewer memories (faster, less noise).
    Complex tasks get more memories (richer context).
    
    Args:
        complexity: Complexity score (0.0 to 1.0)
        base_k: Base retrieval count
        max_k: Maximum retrieval count
        min_k: Minimum retrieval count
        
    Returns:
        Number of memories to retrieve
    """
    # Linear scaling with complexity
    # complexity 0.0 -> min_k
    # complexity 0.5 -> base_k  
    # complexity 1.0 -> max_k
    
    if complexity < 0.3:
        k = min_k + int((base_k - min_k) * (complexity / 0.3))
    elif complexity < 0.7:
        k = base_k
    else:
        k = base_k + int((max_k - base_k) * ((complexity - 0.7) / 0.3))
    
    return max(min_k, min(max_k, k))


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
        
        # New primary interface with auto-estimation
        allocation = budget.allocate(
            prompt="Compare the tradeoffs between LoRA and full fine-tuning",
        )
        print(f"Complexity: {allocation.complexity_estimate:.2f}")
        print(f"Memory retrieval depth: {allocation.memory_k}")
        print(f"Allocated {allocation.tokens} tokens in {allocation.mode.value} mode")
        
        # Legacy interface with explicit complexity
        allocation = budget.allocate_tokens(
            task_complexity=0.7,
            available_vram=8192,
            sensor_status="ONLINE",
        )
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

    # Retrieval depth bounds
    BASE_RETRIEVAL_K: int = 3
    MIN_RETRIEVAL_K: int = 1
    MAX_RETRIEVAL_K: int = 10

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

    def allocate(
        self,
        prompt: str,
        sensor_status: str = "ONLINE",
        memory_context: Optional[Dict[str, Any]] = None,
        available_vram: Optional[int] = None,
    ) -> BudgetAllocation:
        """
        Full allocation with automatic complexity estimation.
        
        This is the primary entry point that:
        1. Estimates task complexity from the prompt
        2. Calculates appropriate retrieval depth
        3. Allocates token budget
        
        Args:
            prompt: The task prompt or query
            sensor_status: Sensor status
            memory_context: Optional pre-existing memory context
            available_vram: Optional VRAM override
            
        Returns:
            BudgetAllocation with tokens, memory_k, and mode
        """
        # Step 1: Estimate complexity
        complexity = estimate_complexity(prompt)
        
        # Step 2: Calculate retrieval depth
        memory_k = calculate_retrieval_depth(
            complexity,
            base_k=self.BASE_RETRIEVAL_K,
            max_k=self.MAX_RETRIEVAL_K,
            min_k=self.MIN_RETRIEVAL_K,
        )
        
        # Step 3: Get token allocation using existing method
        allocation = self.allocate_tokens(
            task_complexity=complexity,
            available_vram=available_vram,
            sensor_status=sensor_status,
            memory_context=memory_context,
        )
        
        # Step 4: Inject retrieval depth and complexity into allocation
        allocation.memory_k = memory_k
        allocation.complexity_estimate = complexity
        allocation.metadata["auto_estimated"] = True
        allocation.metadata["prompt_word_count"] = len(prompt.split())
        
        return allocation

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
        complexities = [a.complexity_estimate for a in allocations]

        from collections import Counter
        mode_counts = Counter(modes)

        return {
            "total_allocations": len(allocations),
            "total_tokens_allocated": self._total_tokens_allocated,
            "average_tokens": sum(tokens) / len(tokens),
            "min_tokens": min(tokens),
            "max_tokens": max(tokens),
            "average_complexity": sum(complexities) / len(complexities),
            "mode_distribution": dict(mode_counts),
            "recent_allocations": [a.to_dict() for a in allocations[-5:]],
        }

    def reset_tracking(self) -> None:
        """Reset allocation tracking."""
        self._allocation_history = []
        self._total_tokens_allocated = 0


# Convenience functions
def allocate_reasoning_budget(
    prompt: str,
    sensor_status: str = "ONLINE",
    memory_context: Optional[Dict[str, Any]] = None,
    task_complexity: Optional[float] = None,
) -> BudgetAllocation:
    """
    Convenience function to allocate reasoning budget.
    
    If task_complexity is provided, uses it directly.
    Otherwise, estimates complexity from the prompt.

    Args:
        prompt: The task prompt or query
        sensor_status: Sensor status
        memory_context: Memory retrieval context
        task_complexity: Optional explicit complexity (skips estimation)

    Returns:
        BudgetAllocation
    """
    allocator = AdaptiveReasoningBudget()
    
    if task_complexity is not None:
        # Use explicit complexity
        allocation = allocator.allocate_tokens(
            task_complexity=task_complexity,
            sensor_status=sensor_status,
            memory_context=memory_context,
        )
        allocation.memory_k = calculate_retrieval_depth(task_complexity)
        allocation.complexity_estimate = task_complexity
        return allocation
    else:
        # Auto-estimate from prompt
        return allocator.allocate(
            prompt=prompt,
            sensor_status=sensor_status,
            memory_context=memory_context,
        )
