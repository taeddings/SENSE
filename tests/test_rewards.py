"""
Tests for enhanced reward system from Agent0 integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from sense.agents.agent_0.trainer import (
    format_reward,
    tool_usage_reward,
    diversity_penalty,
    _bleu_distance_matrix,
    RewardComponents,
    GRPOTrainer,
)
from sense.agents.agent_0.curriculum import (
    extract_boxed_content,
    validate_structured_output,
    format_task_as_structured,
    CurriculumAgent,
    CurriculumTask,
)


class TestFormatReward:
    """Tests for format_reward function."""

    def test_empty_output_returns_zero(self):
        """Empty output should return 0."""
        assert format_reward("") == 0.0
        assert format_reward(None) == 0.0

    def test_thinking_and_answer_returns_full(self):
        """Output with both thinking and answer gets full reward."""
        output = """
        <think>
        Let me analyze this problem step by step.
        First, I'll consider the constraints.
        </think>
        The answer is \\boxed{42}
        """
        assert format_reward(output) == 1.0

    def test_thinking_only_returns_partial(self):
        """Output with only thinking gets partial reward."""
        output = """
        <think>
        I'm analyzing the problem.
        </think>
        The answer is 42.
        """
        assert format_reward(output) == 0.5

    def test_answer_only_returns_partial(self):
        """Output with only boxed answer gets partial reward."""
        output = "The result is \\boxed{123}"
        assert format_reward(output) == 0.5

    def test_code_block_returns_partial(self):
        """Output with code block gets partial reward."""
        output = """
        Here's the solution:
        ```python
        def solve():
            return 42
        ```
        """
        assert format_reward(output) == 0.5

    def test_plain_output_returns_zero(self):
        """Output without structured format gets zero."""
        output = "The answer is 42."
        assert format_reward(output) == 0.0

    def test_custom_format_regex(self):
        """Custom format regex validation."""
        output = "[ANSWER] 42 [/ANSWER]"
        pattern = r"\[ANSWER\].*\[/ANSWER\]"
        assert format_reward(output, expected_format=pattern) == 1.0
        assert format_reward("wrong format", expected_format=pattern) == 0.0


class TestToolUsageReward:
    """Tests for tool_usage_reward function."""

    def test_empty_output_returns_zero(self):
        """Empty output returns 0."""
        assert tool_usage_reward("") == 0.0

    def test_single_tool_call(self):
        """Single tool call gets base reward."""
        output = "```output\nresult: success\n```"
        assert tool_usage_reward(output) == 0.05

    def test_multiple_tool_calls(self):
        """Multiple tool calls get accumulated reward."""
        output = """
        tool_call: execute
        ```output
        result 1
        ```
        tool_call: another
        ```output
        result 2
        ```
        """
        # Should detect tool_call patterns
        reward = tool_usage_reward(output)
        assert reward > 0.05

    def test_cap_enforced(self):
        """Tool reward is capped at maximum."""
        output = """
        tool_call: 1
        tool_call: 2
        tool_call: 3
        tool_call: 4
        tool_call: 5
        tool_call: 6
        """
        reward = tool_usage_reward(output, weight=0.1, cap=4)
        assert reward == 0.4  # 4 * 0.1

    def test_custom_weight(self):
        """Custom weight is applied correctly."""
        output = "tool_call: test"
        assert tool_usage_reward(output, weight=0.2) == 0.2


class TestDiversityPenalty:
    """Tests for diversity_penalty function."""

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        assert diversity_penalty([]) == []

    def test_single_output_no_penalty(self):
        """Single output has no penalty."""
        penalties = diversity_penalty(["single output"])
        assert penalties == [0.0]

    def test_identical_outputs_high_penalty(self):
        """Identical outputs get high penalties."""
        outputs = ["same text", "same text", "same text"]
        penalties = diversity_penalty(outputs)
        # All should have same (high) penalty
        assert len(penalties) == 3
        assert all(p > 0 for p in penalties)

    def test_diverse_outputs_low_penalty(self):
        """Diverse outputs get lower penalties."""
        outputs = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a programming language for data science.",
            "Machine learning models require training data.",
        ]
        penalties = diversity_penalty(outputs)
        assert len(penalties) == 3
        # Diverse outputs should have lower penalties than identical
        assert sum(penalties) / len(penalties) < 1.0


class TestBLEUDistanceMatrix:
    """Tests for BLEU distance matrix calculation."""

    def test_identical_sentences_zero_distance(self):
        """Identical sentences have zero distance."""
        sentences = ["hello world", "hello world"]
        dist = _bleu_distance_matrix(sentences)
        assert dist[0, 0] == 0.0
        assert dist[1, 1] == 0.0
        # Identical sentences should have very low distance
        assert dist[0, 1] < 0.3

    def test_different_sentences_positive_distance(self):
        """Different sentences have positive distance."""
        sentences = [
            "the quick brown fox",
            "machine learning models"
        ]
        dist = _bleu_distance_matrix(sentences)
        assert dist[0, 1] > 0.5
        assert dist[0, 1] == dist[1, 0]  # Symmetric

    def test_matrix_is_symmetric(self):
        """Distance matrix is symmetric."""
        sentences = ["a b c", "d e f", "g h i"]
        dist = _bleu_distance_matrix(sentences)
        assert np.allclose(dist, dist.T)


class TestRewardComponents:
    """Tests for RewardComponents dataclass."""

    def test_to_dict(self):
        """RewardComponents converts to dict correctly."""
        components = RewardComponents(
            base_reward=0.8,
            format_reward=0.1,
            tool_reward=0.05,
            diversity_penalty=0.02,
            total=0.93,
        )
        d = components.to_dict()
        assert d["base_reward"] == 0.8
        assert d["format_reward"] == 0.1
        assert d["tool_reward"] == 0.05
        assert d["diversity_penalty"] == 0.02
        assert d["total"] == 0.93


class TestExtractBoxedContent:
    """Tests for boxed content extraction."""

    def test_simple_boxed(self):
        """Extract simple boxed content."""
        text = r"The answer is \boxed{42}"
        results = extract_boxed_content(text)
        assert results == ["42"]

    def test_multiple_boxed(self):
        """Extract multiple boxed contents."""
        text = r"First: \boxed{1}, Second: \boxed{2}"
        results = extract_boxed_content(text)
        assert results == ["1", "2"]

    def test_nested_braces(self):
        """Handle nested braces correctly."""
        text = r"\boxed{\frac{a}{b}}"
        results = extract_boxed_content(text)
        assert results == [r"\frac{a}{b}"]

    def test_no_boxed(self):
        """Return empty list when no boxed content."""
        text = "Just plain text without boxes"
        results = extract_boxed_content(text)
        assert results == []


class TestValidateStructuredOutput:
    """Tests for structured output validation."""

    def test_valid_structured_output(self):
        """Valid structured output is recognized."""
        output = """
        <question>What is 2+2?</question>
        <think>Let me calculate: 2+2=4</think>
        The answer is \\boxed{4}
        """
        is_valid, data = validate_structured_output(output)
        assert is_valid
        assert data["has_question_tags"]
        assert data["has_boxed_answer"]
        assert data["has_thinking"]

    def test_missing_question_invalid(self):
        """Missing question tag makes output invalid."""
        output = """
        <think>Thinking...</think>
        \\boxed{answer}
        """
        is_valid, data = validate_structured_output(output)
        assert not is_valid
        assert not data["has_question_tags"]

    def test_missing_answer_invalid(self):
        """Missing boxed answer makes output invalid."""
        output = """
        <question>What is the answer?</question>
        The answer is 42.
        """
        is_valid, data = validate_structured_output(output)
        assert not is_valid
        assert not data["has_boxed_answer"]


class TestFormatTaskAsStructured:
    """Tests for structured task formatting."""

    def test_basic_format(self):
        """Basic formatting includes question tags."""
        result = format_task_as_structured("What is 2+2?")
        assert "<question>" in result
        assert "What is 2+2?" in result
        assert "</question>" in result

    def test_with_expected_answer(self):
        """Formatting includes expected answer hint."""
        result = format_task_as_structured("Test question", answer="42")
        assert "\\boxed{42}" in result

    def test_without_thinking_prompt(self):
        """Can disable thinking prompt."""
        result = format_task_as_structured(
            "Test",
            include_thinking_prompt=False
        )
        assert "<think>" not in result


class TestCurriculumTaskEnhancements:
    """Tests for enhanced CurriculumTask."""

    def test_format_compliance_tracking(self):
        """Format compliance is tracked correctly."""
        task = CurriculumTask(
            task_id="test",
            description="Test task",
            difficulty=0.5,
            category="terminal",
        )

        # Record attempts with varying format compliance
        task.record_attempt(success=True, format_valid=True)
        task.record_attempt(success=False, format_valid=True)
        task.record_attempt(success=True, format_valid=False)

        assert task.attempts == 3
        assert task.format_successes == 2
        assert task.format_compliance_rate == pytest.approx(2/3)

    def test_get_structured_description(self):
        """Structured description is formatted correctly."""
        task = CurriculumTask(
            task_id="test",
            description="Count to 10",
            difficulty=0.5,
            category="terminal",
            structured_format=True,
            expected_answer="1,2,3,4,5,6,7,8,9,10",
        )

        desc = task.get_structured_description()
        assert "<question>" in desc
        assert "Count to 10" in desc
        assert "\\boxed" in desc

    def test_domain_in_to_dict(self):
        """Domain is included in dict output."""
        task = CurriculumTask(
            task_id="test",
            description="Test",
            difficulty=0.5,
            category="terminal",
            domain="coding",
        )

        d = task.to_dict()
        assert d["domain"] == "coding"


class TestGRPOTrainerEnhancedRewards:
    """Tests for GRPOTrainer enhanced reward computation."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer with mocked curriculum."""
        with patch('sense.agents.agent_0.trainer.CurriculumAgent'):
            mock_curriculum = Mock()
            trainer = GRPOTrainer(
                curriculum_agent=mock_curriculum,
                format_reward_weight=0.1,
                tool_reward_weight=0.05,
                diversity_penalty_weight=0.1,
                enable_enhanced_rewards=True,
            )
            return trainer

    def test_compute_enhanced_reward(self, trainer):
        """Enhanced reward computation combines all components."""
        output = """
        <think>Analyzing...</think>
        tool_call: execute
        \\boxed{result}
        """
        components = trainer.compute_enhanced_reward(
            base_reward=0.8,
            output=output,
            all_outputs=[output],
            output_index=0,
        )

        assert components.base_reward == 0.8
        assert components.format_reward > 0  # Has thinking and boxed
        assert components.tool_reward > 0  # Has tool call
        assert components.diversity_penalty == 0  # Single output
        assert components.total == pytest.approx(
            components.base_reward +
            components.format_reward +
            components.tool_reward -
            components.diversity_penalty
        )

    def test_enhanced_rewards_disabled(self, trainer):
        """When disabled, only base reward is used."""
        trainer.enable_enhanced_rewards = False

        components = trainer.compute_enhanced_reward(
            base_reward=0.8,
            output="<think>test</think>\\boxed{x}",
        )

        assert components.base_reward == 0.8
        assert components.format_reward == 0.0
        assert components.tool_reward == 0.0
        assert components.total == 0.8

    def test_reward_component_stats_empty(self, trainer):
        """Stats returns no_data when no history."""
        stats = trainer.get_reward_component_stats()
        assert stats["status"] == "no_data"
