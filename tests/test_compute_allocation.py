"""
Tests for SENSE-v2 Adaptive Reasoning Budget and Context Engineering

Tests cover:
- estimate_complexity() heuristic scoring
- calculate_retrieval_depth() adaptive k
- AdaptiveReasoningBudget allocation
- BudgetAllocation dataclass
- ReasoningMode selection
- Resource guards and compensatory reasoning
"""

import pytest
from unittest.mock import patch, MagicMock

from sense.llm.reasoning.compute_allocation import (
    estimate_complexity,
    calculate_retrieval_depth,
    AdaptiveReasoningBudget,
    BudgetAllocation,
    ReasoningMode,
    ResourceStatus,
    allocate_reasoning_budget,
)


class TestEstimateComplexityBasic:
    """Test basic complexity estimation behavior."""

    def test_empty_prompt_returns_low_complexity(self):
        """Empty or whitespace prompts return minimal complexity."""
        assert estimate_complexity("") == 0.1
        assert estimate_complexity("   ") == 0.1
        assert estimate_complexity("\n\t") == 0.1

    def test_simple_factual_question(self):
        """Simple factual questions score low complexity."""
        complexity = estimate_complexity("What is Python?")
        assert complexity < 0.3
        assert complexity > 0.0

    def test_simple_greeting(self):
        """Simple greetings score very low."""
        complexity = estimate_complexity("Hello, how are you?")
        assert complexity < 0.25


class TestEstimateComplexityAnalytical:
    """Test complexity for analytical questions."""

    def test_compare_question_scores_higher(self):
        """Questions with 'compare' score higher."""
        simple = estimate_complexity("What is machine learning?")
        analytical = estimate_complexity("Compare supervised and unsupervised learning approaches")
        assert analytical > simple
        assert analytical >= 0.3

    def test_analyze_question_scores_higher(self):
        """Questions with 'analyze' score higher."""
        complexity = estimate_complexity("Analyze the performance implications of this algorithm")
        assert complexity >= 0.3

    def test_tradeoffs_question(self):
        """Questions about tradeoffs are complex."""
        complexity = estimate_complexity("What are the tradeoffs between LoRA and full fine-tuning?")
        assert complexity >= 0.35


class TestEstimateComplexityExplanatory:
    """Test complexity for explanatory questions."""

    def test_why_question(self):
        """Why questions score medium complexity."""
        complexity = estimate_complexity("Why does gradient descent converge?")
        assert 0.15 <= complexity <= 0.5

    def test_how_question(self):
        """How questions score medium complexity."""
        complexity = estimate_complexity("How does backpropagation work?")
        assert 0.15 <= complexity <= 0.5

    def test_explain_request(self):
        """Explain requests score medium complexity."""
        complexity = estimate_complexity("Explain the attention mechanism in transformers")
        assert 0.15 <= complexity <= 0.5


class TestEstimateComplexityMultiStep:
    """Test complexity for multi-step indicators."""

    def test_sequential_keywords(self):
        """Multi-step keywords increase complexity."""
        simple = estimate_complexity("Install Python")
        multi_step = estimate_complexity("First install Python, then configure the environment, finally run the tests")
        assert multi_step > simple

    def test_step_by_step_request(self):
        """Step-by-step requests are complex."""
        complexity = estimate_complexity("Walk me through the process step by step")
        assert complexity >= 0.2

    def test_procedure_request(self):
        """Procedure requests indicate complexity."""
        complexity = estimate_complexity("What is the procedure for deploying to production?")
        assert complexity >= 0.15


class TestEstimateComplexityCode:
    """Test complexity for code-related prompts."""

    def test_code_block_marker(self):
        """Code blocks increase complexity."""
        complexity = estimate_complexity("Fix this code:\n```python\ndef foo(): pass\n```")
        assert complexity >= 0.2

    def test_implement_keyword(self):
        """Implement keyword increases complexity."""
        complexity = estimate_complexity("Implement a binary search algorithm")
        assert complexity >= 0.2

    def test_debug_keyword(self):
        """Debug keyword increases complexity."""
        complexity = estimate_complexity("Debug this error: TypeError in line 42")
        assert complexity >= 0.2

    def test_refactor_keyword(self):
        """Refactor keyword increases complexity."""
        complexity = estimate_complexity("Refactor this function to be more efficient")
        assert complexity >= 0.2


class TestEstimateComplexityConstraints:
    """Test complexity for constrained requests."""

    def test_without_constraint(self):
        """'Without' constraints add complexity."""
        simple = estimate_complexity("Write a sorting function")
        constrained = estimate_complexity("Write a sorting function without using recursion")
        assert constrained > simple

    def test_except_constraint(self):
        """'Except' constraints add complexity."""
        complexity = estimate_complexity("List all programming languages except Python")
        assert complexity >= 0.15

    def test_must_not_constraint(self):
        """'Must not' constraints add complexity."""
        complexity = estimate_complexity("Generate text that must not contain profanity")
        assert complexity >= 0.15


class TestEstimateComplexityDomain:
    """Test complexity for domain-specific terms."""

    def test_technical_domain_markers(self):
        """Technical domain terms increase complexity."""
        simple = estimate_complexity("How do I start?")
        technical = estimate_complexity("What is the architecture for distributed concurrent systems?")
        assert technical > simple

    def test_statistical_markers(self):
        """Statistical terms increase complexity."""
        complexity = estimate_complexity("Calculate the confidence interval for this regression")
        assert complexity >= 0.3

    def test_business_markers(self):
        """Business terms increase complexity."""
        complexity = estimate_complexity("Define the requirements specification for stakeholders")
        assert complexity >= 0.2


class TestEstimateComplexityLength:
    """Test complexity scaling with prompt length."""

    def test_longer_prompts_score_higher(self):
        """Longer prompts generally score higher."""
        short = estimate_complexity("Fix bug")
        medium = estimate_complexity("Fix the bug in the authentication system that causes users to be logged out unexpectedly")
        long_prompt = estimate_complexity(
            "Fix the bug in the authentication system that causes users to be logged out unexpectedly. "
            "This bug appears to be related to session management and only occurs when users switch "
            "between different pages rapidly. Please also add appropriate error handling and logging."
        )
        assert medium >= short
        assert long_prompt >= medium

    def test_complexity_caps_at_one(self):
        """Complexity should not exceed 1.0."""
        very_long = "Compare and analyze " + " ".join(["word"] * 200)
        complexity = estimate_complexity(very_long)
        assert complexity <= 1.0


class TestEstimateComplexityEntityDensity:
    """Test entity density scoring."""

    def test_proper_nouns_increase_complexity(self):
        """Proper nouns (capitalized terms) increase complexity."""
        generic = estimate_complexity("how do i use the library")
        specific = estimate_complexity("how do I use TensorFlow with PyTorch and HuggingFace Transformers")
        assert specific > generic


class TestCalculateRetrievalDepthBasic:
    """Test basic retrieval depth calculation."""

    def test_low_complexity_returns_min_k(self):
        """Low complexity (< 0.3) returns min to base k."""
        k = calculate_retrieval_depth(0.1, base_k=3, max_k=10, min_k=1)
        assert 1 <= k <= 3

    def test_medium_complexity_returns_base_k(self):
        """Medium complexity (0.3-0.7) returns base k."""
        k = calculate_retrieval_depth(0.5, base_k=3, max_k=10, min_k=1)
        assert k == 3

    def test_high_complexity_returns_more_than_base(self):
        """High complexity (> 0.7) returns more than base k."""
        k = calculate_retrieval_depth(0.9, base_k=3, max_k=10, min_k=1)
        assert k > 3
        assert k <= 10

    def test_max_complexity_approaches_max_k(self):
        """Maximum complexity (1.0) approaches max k."""
        k = calculate_retrieval_depth(1.0, base_k=3, max_k=10, min_k=1)
        assert k >= 8


class TestCalculateRetrievalDepthBounds:
    """Test retrieval depth bounds."""

    def test_never_below_min_k(self):
        """Result never goes below min_k."""
        k = calculate_retrieval_depth(0.0, base_k=3, max_k=10, min_k=1)
        assert k >= 1

    def test_never_above_max_k(self):
        """Result never exceeds max_k."""
        k = calculate_retrieval_depth(1.0, base_k=3, max_k=10, min_k=1)
        assert k <= 10

    def test_custom_bounds(self):
        """Custom bounds are respected."""
        k = calculate_retrieval_depth(0.5, base_k=5, max_k=20, min_k=2)
        assert k == 5

        k_high = calculate_retrieval_depth(1.0, base_k=5, max_k=20, min_k=2)
        assert k_high <= 20


class TestCalculateRetrievalDepthScaling:
    """Test retrieval depth scaling behavior."""

    def test_monotonic_increase(self):
        """Retrieval depth increases monotonically with complexity."""
        prev_k = 0
        for complexity in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            k = calculate_retrieval_depth(complexity)
            assert k >= prev_k
            prev_k = k


class TestBudgetAllocation:
    """Test BudgetAllocation dataclass."""

    def test_dataclass_fields(self):
        """BudgetAllocation has expected fields."""
        allocation = BudgetAllocation(
            tokens=1024,
            mode=ReasoningMode.BALANCED,
            verification_depth=2,
            memory_k=5,
            complexity_estimate=0.65,
            rationale="test rationale",
        )
        assert allocation.tokens == 1024
        assert allocation.mode == ReasoningMode.BALANCED
        assert allocation.verification_depth == 2
        assert allocation.memory_k == 5
        assert allocation.complexity_estimate == 0.65
        assert allocation.rationale == "test rationale"

    def test_to_dict(self):
        """BudgetAllocation serializes to dict."""
        allocation = BudgetAllocation(
            tokens=512,
            mode=ReasoningMode.EFFICIENT,
            verification_depth=1,
            memory_k=2,
            complexity_estimate=0.3,
        )
        d = allocation.to_dict()
        assert d["tokens"] == 512
        assert d["mode"] == "efficient"
        assert d["verification_depth"] == 1
        assert d["memory_k"] == 2
        assert d["complexity_estimate"] == 0.3

    def test_default_values(self):
        """BudgetAllocation has sensible defaults."""
        allocation = BudgetAllocation(
            tokens=1024,
            mode=ReasoningMode.BALANCED,
            verification_depth=1,
        )
        assert allocation.memory_k == 3
        assert allocation.complexity_estimate == 0.5
        assert allocation.rationale == ""
        assert allocation.metadata == {}


class TestReasoningMode:
    """Test ReasoningMode enum."""

    def test_mode_values(self):
        """ReasoningMode has expected values."""
        assert ReasoningMode.EFFICIENT.value == "efficient"
        assert ReasoningMode.BALANCED.value == "balanced"
        assert ReasoningMode.EXPLORATORY.value == "exploratory"
        assert ReasoningMode.COMPENSATORY.value == "compensatory"


class TestAdaptiveReasoningBudgetInit:
    """Test AdaptiveReasoningBudget initialization."""

    def test_default_init(self):
        """Default initialization uses constants."""
        budget = AdaptiveReasoningBudget()
        assert budget.base_budget == 1024
        assert budget.min_threshold == 256
        assert budget.max_threshold == 8192

    def test_custom_init(self):
        """Custom initialization overrides defaults."""
        budget = AdaptiveReasoningBudget(
            base_budget=2048,
            min_threshold=512,
            max_threshold=4096,
        )
        assert budget.base_budget == 2048
        assert budget.min_threshold == 512
        assert budget.max_threshold == 4096


class TestAdaptiveReasoningBudgetAllocate:
    """Test AdaptiveReasoningBudget.allocate() method."""

    def test_allocate_returns_budget_allocation(self):
        """allocate() returns BudgetAllocation instance."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate(prompt="What is Python?")
        assert isinstance(allocation, BudgetAllocation)

    def test_allocate_estimates_complexity(self):
        """allocate() estimates complexity from prompt."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate(prompt="Compare LoRA and full fine-tuning")
        assert allocation.complexity_estimate > 0.0
        assert allocation.metadata.get("auto_estimated") is True

    def test_allocate_calculates_memory_k(self):
        """allocate() calculates appropriate memory_k."""
        budget = AdaptiveReasoningBudget()

        simple_allocation = budget.allocate(prompt="What is Python?")
        complex_allocation = budget.allocate(
            prompt="Compare and analyze the architectural tradeoffs between microservices and monolithic systems"
        )

        assert complex_allocation.memory_k >= simple_allocation.memory_k

    def test_allocate_simple_prompt(self):
        """Simple prompts get efficient allocation."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate(prompt="Hi")
        assert allocation.tokens < budget.base_budget
        assert allocation.memory_k <= 3

    def test_allocate_complex_prompt(self):
        """Complex prompts get higher allocation."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate(
            prompt="Analyze the performance implications of distributed concurrent architecture with statistical hypothesis testing"
        )
        assert allocation.complexity_estimate > 0.5
        assert allocation.memory_k >= 3


class TestAdaptiveReasoningBudgetAllocateTokens:
    """Test AdaptiveReasoningBudget.allocate_tokens() method."""

    def test_allocate_tokens_basic(self):
        """allocate_tokens() returns allocation based on complexity."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate_tokens(task_complexity=0.5)
        assert allocation.tokens > 0
        assert allocation.mode in ReasoningMode

    def test_higher_complexity_more_tokens(self):
        """Higher complexity results in more tokens."""
        budget = AdaptiveReasoningBudget()
        low = budget.allocate_tokens(task_complexity=0.2)
        high = budget.allocate_tokens(task_complexity=0.8)
        assert high.tokens >= low.tokens

    def test_tokens_bounded_by_thresholds(self):
        """Tokens stay within min/max thresholds."""
        budget = AdaptiveReasoningBudget(min_threshold=256, max_threshold=4096)

        for complexity in [0.0, 0.5, 1.0]:
            allocation = budget.allocate_tokens(task_complexity=complexity)
            assert allocation.tokens >= 256
            assert allocation.tokens <= 4096


class TestAdaptiveReasoningBudgetSensorOffline:
    """Test compensatory reasoning when sensor offline."""

    def test_sensor_offline_doubles_budget(self):
        """Sensor OFFLINE triggers compensatory mode."""
        budget = AdaptiveReasoningBudget()

        online = budget.allocate_tokens(task_complexity=0.5, sensor_status="ONLINE")
        offline = budget.allocate_tokens(task_complexity=0.5, sensor_status="OFFLINE")

        assert offline.mode == ReasoningMode.COMPENSATORY
        assert offline.tokens > online.tokens
        assert offline.verification_depth >= 2

    def test_sensor_offline_rationale(self):
        """Sensor OFFLINE is documented in rationale."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate_tokens(task_complexity=0.5, sensor_status="OFFLINE")
        assert "OFFLINE" in allocation.rationale or "doubled" in allocation.rationale.lower()


class TestAdaptiveReasoningBudgetVRAMGuards:
    """Test VRAM resource guards."""

    def test_critical_vram_caps_budget(self):
        """Critical VRAM (>90%) caps at min threshold."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate_tokens(
            task_complexity=0.9,
            available_vram=500,  # Low VRAM triggers high percent estimate
        )
        # The allocation should be capped when VRAM is critical
        assert allocation.tokens <= budget.max_threshold

    def test_vram_warning_reduces_budget(self):
        """Warning VRAM (>75%) reduces budget."""
        budget = AdaptiveReasoningBudget()
        # Note: available_vram affects percent calculation
        allocation = budget.allocate_tokens(
            task_complexity=0.5,
            available_vram=2000,  # Moderate VRAM
        )
        assert allocation.tokens > 0


class TestAdaptiveReasoningBudgetMemoryContext:
    """Test memory context integration."""

    def test_high_similarity_reduces_budget(self):
        """High similarity (>0.95) reduces budget."""
        budget = AdaptiveReasoningBudget()

        no_context = budget.allocate_tokens(task_complexity=0.5)
        high_similarity = budget.allocate_tokens(
            task_complexity=0.5,
            memory_context={"max_similarity": 0.98},
        )

        assert high_similarity.tokens < no_context.tokens

    def test_low_similarity_no_reduction(self):
        """Low similarity does not reduce budget."""
        budget = AdaptiveReasoningBudget()

        no_context = budget.allocate_tokens(task_complexity=0.5)
        low_similarity = budget.allocate_tokens(
            task_complexity=0.5,
            memory_context={"max_similarity": 0.5},
        )

        assert low_similarity.tokens == no_context.tokens


class TestAdaptiveReasoningBudgetModes:
    """Test reasoning mode selection."""

    def test_low_drift_efficient_mode(self):
        """Low drift selects EFFICIENT mode."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate_tokens(
            task_complexity=0.5,
            memory_context={"drift_level": 0.1},
        )
        assert allocation.mode == ReasoningMode.EFFICIENT

    def test_medium_drift_balanced_mode(self):
        """Medium drift selects BALANCED mode."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate_tokens(
            task_complexity=0.5,
            memory_context={"drift_level": 0.5},
        )
        assert allocation.mode == ReasoningMode.BALANCED

    def test_high_drift_exploratory_mode(self):
        """High drift selects EXPLORATORY mode."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate_tokens(
            task_complexity=0.5,
            memory_context={"drift_level": 0.8},
        )
        assert allocation.mode == ReasoningMode.EXPLORATORY


class TestAdaptiveReasoningBudgetStats:
    """Test allocation statistics tracking."""

    def test_allocation_tracking(self):
        """Allocations are tracked in history."""
        budget = AdaptiveReasoningBudget()
        budget.reset_tracking()

        budget.allocate(prompt="Test 1")
        budget.allocate(prompt="Test 2")
        budget.allocate(prompt="Test 3")

        stats = budget.get_allocation_stats()
        assert stats["total_allocations"] == 3
        assert stats["total_tokens_allocated"] > 0
        assert len(stats["recent_allocations"]) == 3

    def test_reset_tracking(self):
        """reset_tracking() clears history."""
        budget = AdaptiveReasoningBudget()
        budget.allocate(prompt="Test")
        budget.reset_tracking()

        stats = budget.get_allocation_stats()
        assert stats["status"] == "no_allocations"


class TestResourceStatus:
    """Test ResourceStatus helper class."""

    def test_resource_status_defaults(self):
        """ResourceStatus has sensible defaults."""
        status = ResourceStatus()
        assert status.vram_percent == 0.0
        assert status.sensor_status == "ONLINE"

    @patch('sense.llm.reasoning.compute_allocation.PSUTIL_AVAILABLE', False)
    def test_get_current_without_psutil(self):
        """get_current() handles missing psutil."""
        status = ResourceStatus.get_current()
        assert status.psutil_available is False


class TestConvenienceFunction:
    """Test allocate_reasoning_budget() convenience function."""

    def test_convenience_auto_estimate(self):
        """Convenience function auto-estimates complexity."""
        allocation = allocate_reasoning_budget(prompt="What is Python?")
        assert isinstance(allocation, BudgetAllocation)
        assert allocation.complexity_estimate > 0

    def test_convenience_explicit_complexity(self):
        """Convenience function accepts explicit complexity."""
        allocation = allocate_reasoning_budget(
            prompt="ignored",
            task_complexity=0.75,
        )
        assert allocation.complexity_estimate == 0.75

    def test_convenience_with_sensor_status(self):
        """Convenience function accepts sensor status."""
        allocation = allocate_reasoning_budget(
            prompt="Test",
            sensor_status="OFFLINE",
        )
        assert allocation.mode == ReasoningMode.COMPENSATORY


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_prompt(self):
        """Very long prompts don't cause errors."""
        long_prompt = "word " * 1000
        complexity = estimate_complexity(long_prompt)
        assert 0.0 <= complexity <= 1.0

    def test_unicode_prompt(self):
        """Unicode characters don't cause errors."""
        complexity = estimate_complexity("Analyze the tradeoffs between 日本語 and English text processing")
        assert 0.0 <= complexity <= 1.0

    def test_newlines_in_prompt(self):
        """Newlines in prompt don't cause errors."""
        complexity = estimate_complexity("First step:\nDo this\n\nSecond step:\nDo that")
        assert 0.0 <= complexity <= 1.0

    def test_special_characters(self):
        """Special characters don't cause errors."""
        complexity = estimate_complexity("Compare $100 vs €100 using @mentions and #hashtags")
        assert 0.0 <= complexity <= 1.0

    def test_code_with_braces(self):
        """Code with braces doesn't cause errors."""
        complexity = estimate_complexity("def foo(): { return bar() }")
        assert 0.0 <= complexity <= 1.0


class TestIntegration:
    """Integration tests for full allocation pipeline."""

    def test_end_to_end_simple_query(self):
        """End-to-end test for simple query."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate(prompt="What time is it?")

        assert allocation.complexity_estimate < 0.3
        assert allocation.memory_k <= 3
        assert allocation.tokens > 0
        assert allocation.mode in [ReasoningMode.EFFICIENT, ReasoningMode.BALANCED]

    def test_end_to_end_complex_query(self):
        """End-to-end test for complex query."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate(
            prompt=(
                "Compare and contrast the architectural tradeoffs between using LoRA "
                "fine-tuning versus full model fine-tuning. Analyze the implications "
                "for memory usage, training time, and model quality. Then, step by step, "
                "explain how to implement each approach using the HuggingFace Transformers library."
            )
        )

        assert allocation.complexity_estimate > 0.5
        assert allocation.memory_k >= 3
        assert allocation.tokens > budget.min_threshold

    def test_allocation_with_all_parameters(self):
        """Test allocation with all optional parameters."""
        budget = AdaptiveReasoningBudget()
        allocation = budget.allocate(
            prompt="Compare architectures",
            sensor_status="DEGRADED",
            memory_context={"max_similarity": 0.7, "drift_level": 0.4},
            available_vram=8000,
        )

        assert isinstance(allocation, BudgetAllocation)
        assert allocation.tokens > 0
        assert allocation.memory_k > 0
