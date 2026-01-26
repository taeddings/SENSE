"""
SENSE v4.0: Metacognitive Engine

Self-awareness of reasoning quality and process.
Tracks reasoning traces, evaluates quality, and detects when to backtrack.
"""

import logging
import json
import os
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class StepType(Enum):
    """Types of reasoning steps."""
    MODE_DECISION = "mode_decision"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    TOOL_CALL = "tool_call"
    SYNTHESIS = "synthesis"
    CLARIFICATION = "clarification"
    VALIDATION = "validation"


@dataclass
class ReasoningStep:
    """
    A single step in the reasoning process.

    Attributes:
        step_type: Type of reasoning step
        content: Description of the step
        confidence: Confidence in this step (0.0 - 1.0)
        timestamp: When this step occurred
        metadata: Additional step-specific data
    """
    step_type: str
    content: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_type': self.step_type,
            'content': self.content,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class QualityScore:
    """
    Reasoning quality metrics.

    Attributes:
        coherence: Logical flow between steps (0.0 - 1.0)
        completeness: All relevant factors considered (0.0 - 1.0)
        efficiency: No unnecessary steps (0.0 - 1.0)
        confidence_stability: Confidence consistency (0.0 - 1.0)
        overall: Combined quality score (0.0 - 1.0)
        issues: List of detected quality issues
    """
    coherence: float
    completeness: float
    efficiency: float
    confidence_stability: float
    overall: float
    issues: List[str] = field(default_factory=list)

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if reasoning meets quality threshold."""
        return self.overall >= threshold


@dataclass
class ReasoningTrace:
    """
    Complete trace of a reasoning process.

    Attributes:
        task: The original task
        steps: List of reasoning steps
        start_time: When reasoning started
        end_time: When reasoning completed
        quality_score: Evaluated quality metrics
        metadata: Additional trace-level data
    """
    task: str
    steps: List[ReasoningStep] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    quality_score: Optional[QualityScore] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the trace."""
        self.steps.append(step)

    def complete(self, quality_score: QualityScore) -> None:
        """Mark trace as complete with quality evaluation."""
        self.end_time = datetime.now()
        self.quality_score = quality_score

    def duration_seconds(self) -> float:
        """Get reasoning duration in seconds."""
        if self.end_time is None:
            end = datetime.now()
        else:
            end = self.end_time
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task': self.task,
            'steps': [s.to_dict() for s in self.steps],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'quality_score': asdict(self.quality_score) if self.quality_score else None,
            'duration_seconds': self.duration_seconds(),
            'metadata': self.metadata
        }


# Law 3: OS-Agnostic Workspace
def get_traces_path():
    """Get OS-agnostic path for reasoning traces."""
    if hasattr(sys, 'getandroidapilevel') or os.path.exists('/data/data/com.termux'):
        base = "/sdcard/Download/SENSE_Data"
    elif os.name == 'nt':
        base = os.path.join(os.path.expanduser("~"), "Documents", "SENSE_Data")
    else:
        base = os.path.join(os.path.expanduser("~"), "SENSE_Data")
    return os.path.join(base, "reasoning_traces.jsonl")


class MetacognitiveEngine:
    """
    Self-monitoring layer for reasoning quality.

    Tracks reasoning processes, evaluates quality, and provides
    self-awareness of reasoning capabilities.
    """

    def __init__(self, trace_enabled: bool = True, log_level: str = "info"):
        """
        Initialize metacognitive engine.

        Args:
            trace_enabled: Whether to log reasoning traces
            log_level: Logging level for metacognitive insights
        """
        self.trace_enabled = trace_enabled
        self.current_trace: Optional[ReasoningTrace] = None
        self.quality_history: List[QualityScore] = []
        self.logger = logging.getLogger("Intelligence.Metacognition")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Ensure traces directory exists
        if trace_enabled:
            self._ensure_traces_dir()

    def _ensure_traces_dir(self):
        """Ensure traces directory exists."""
        traces_path = get_traces_path()
        directory = os.path.dirname(traces_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Could not create traces directory: {e}")

    def start_trace(self, task: str, metadata: Optional[Dict] = None) -> ReasoningTrace:
        """
        Begin tracking a reasoning process.

        Args:
            task: The task being reasoned about
            metadata: Optional metadata about the task

        Returns:
            New ReasoningTrace object
        """
        if metadata is None:
            metadata = {}

        self.current_trace = ReasoningTrace(
            task=task,
            metadata=metadata
        )

        self.logger.info(f"🧠 Started reasoning trace for: {task[:50]}...")

        return self.current_trace

    def log_step(
        self,
        step_type: str,
        content: str,
        confidence: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a reasoning step.

        Args:
            step_type: Type of step (from StepType enum)
            content: Description of what happened
            confidence: Confidence in this step (0.0 - 1.0)
            metadata: Optional step-specific metadata
        """
        if not self.current_trace:
            self.logger.warning("No active trace, ignoring step log")
            return

        if metadata is None:
            metadata = {}

        step = ReasoningStep(
            step_type=step_type,
            content=content,
            confidence=confidence,
            metadata=metadata
        )

        self.current_trace.add_step(step)

        self.logger.debug(
            f"Step [{step_type}] confidence={confidence:.2f}: {content[:50]}..."
        )

    def evaluate_trace(self) -> QualityScore:
        """
        Evaluate reasoning quality of current trace.

        Returns:
            QualityScore with detailed metrics
        """
        if not self.current_trace or not self.current_trace.steps:
            return QualityScore(
                coherence=0.0,
                completeness=0.0,
                efficiency=0.0,
                confidence_stability=0.0,
                overall=0.0,
                issues=["no_steps"]
            )

        issues = []

        # 1. Coherence: Steps follow logically
        coherence = self._evaluate_coherence()
        if coherence < 0.6:
            issues.append(f"low_coherence: {coherence:.2f}")

        # 2. Completeness: All aspects addressed
        completeness = self._evaluate_completeness()
        if completeness < 0.7:
            issues.append(f"incomplete: {completeness:.2f}")

        # 3. Efficiency: No unnecessary loops
        efficiency = self._evaluate_efficiency()
        if efficiency < 0.7:
            issues.append(f"inefficient: {efficiency:.2f}")

        # 4. Confidence stability: Not wildly fluctuating
        confidence_stability = self._evaluate_confidence_stability()
        if confidence_stability < 0.6:
            issues.append(f"unstable_confidence: {confidence_stability:.2f}")

        # Overall score (weighted average)
        overall = (
            coherence * 0.3 +
            completeness * 0.3 +
            efficiency * 0.2 +
            confidence_stability * 0.2
        )

        quality_score = QualityScore(
            coherence=coherence,
            completeness=completeness,
            efficiency=efficiency,
            confidence_stability=confidence_stability,
            overall=overall,
            issues=issues
        )

        self.quality_history.append(quality_score)

        return quality_score

    def _evaluate_coherence(self) -> float:
        """
        Evaluate logical flow between steps.

        High coherence = steps follow a logical sequence.
        """
        steps = self.current_trace.steps

        if len(steps) < 2:
            return 1.0  # Single step is trivially coherent

        # Check for expected step sequences
        expected_sequences = [
            (StepType.MODE_DECISION.value, StepType.KNOWLEDGE_RETRIEVAL.value),
            (StepType.KNOWLEDGE_RETRIEVAL.value, StepType.TOOL_CALL.value),
            (StepType.TOOL_CALL.value, StepType.SYNTHESIS.value),
            (StepType.CLARIFICATION.value, StepType.MODE_DECISION.value),
        ]

        coherent_transitions = 0
        total_transitions = len(steps) - 1

        for i in range(len(steps) - 1):
            current_type = steps[i].step_type
            next_type = steps[i + 1].step_type

            # Check if this is an expected transition
            is_coherent = any(
                current_type == seq[0] and next_type == seq[1]
                for seq in expected_sequences
            )

            if is_coherent:
                coherent_transitions += 1

        return coherent_transitions / total_transitions if total_transitions > 0 else 1.0

    def _evaluate_completeness(self) -> float:
        """
        Evaluate if all relevant aspects were considered.

        Checks for presence of key step types.
        """
        steps = self.current_trace.steps
        step_types = {s.step_type for s in steps}

        # Expected step types for complete reasoning
        expected_types = {
            StepType.MODE_DECISION.value,
            StepType.SYNTHESIS.value,
        }

        # Calculate coverage
        covered = len(expected_types.intersection(step_types))
        total = len(expected_types)

        return covered / total if total > 0 else 1.0

    def _evaluate_efficiency(self) -> float:
        """
        Evaluate efficiency (no unnecessary steps/loops).

        Penalizes:
        - Excessive tool calls
        - Repeated similar steps
        - Very long reasoning chains
        """
        steps = self.current_trace.steps

        if len(steps) < 2:
            return 1.0

        # Check for repeated tool calls
        tool_calls = [s for s in steps if s.step_type == StepType.TOOL_CALL.value]
        if len(tool_calls) > 3:
            efficiency = max(0.5, 1.0 - (len(tool_calls) - 3) * 0.1)
        else:
            efficiency = 1.0

        # Penalize very long chains
        if len(steps) > 10:
            efficiency *= max(0.7, 1.0 - (len(steps) - 10) * 0.05)

        return max(0.0, min(1.0, efficiency))

    def _evaluate_confidence_stability(self) -> float:
        """
        Evaluate confidence stability across steps.

        High stability = confidence doesn't fluctuate wildly.
        """
        steps = self.current_trace.steps

        if len(steps) < 2:
            return 1.0

        confidences = [s.confidence for s in steps]

        # Calculate variance
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5

        # Low variance = high stability
        # Normalize to [0, 1] (assuming std_dev in range [0, 0.5])
        stability = max(0.0, 1.0 - (std_dev * 2))

        return stability

    def should_backtrack(self) -> bool:
        """
        Detect when reasoning has gone wrong and should restart.

        Backtracking triggers:
        - Confidence drops significantly
        - Circular reasoning detected
        - Quality score very low
        """
        if not self.current_trace or len(self.current_trace.steps) < 3:
            return False

        steps = self.current_trace.steps

        # Check for confidence drop
        recent_confidences = [s.confidence for s in steps[-3:]]
        if all(c < 0.4 for c in recent_confidences):
            self.logger.warning("⚠️ Backtrack trigger: Low confidence cascade")
            return True

        # Check for circular reasoning (repeated tool calls with same args)
        tool_steps = [s for s in steps if s.step_type == StepType.TOOL_CALL.value]
        if len(tool_steps) >= 2:
            # Check if last two tool calls are identical
            if tool_steps[-1].content == tool_steps[-2].content:
                self.logger.warning("⚠️ Backtrack trigger: Circular tool calls")
                return True

        # Check overall quality if enough steps
        if len(steps) >= 5:
            quality = self.evaluate_trace()
            if quality.overall < 0.3:
                self.logger.warning(f"⚠️ Backtrack trigger: Low quality ({quality.overall:.2f})")
                return True

        return False

    def complete_trace(self) -> Optional[ReasoningTrace]:
        """
        Complete the current trace and persist it.

        Returns:
            Completed ReasoningTrace, or None if no active trace
        """
        if not self.current_trace:
            return None

        # Evaluate quality
        quality = self.evaluate_trace()
        self.current_trace.complete(quality)

        # Persist if enabled
        if self.trace_enabled:
            self._persist_trace(self.current_trace)

        self.logger.info(
            f"✅ Completed trace: quality={quality.overall:.2f}, "
            f"steps={len(self.current_trace.steps)}, "
            f"duration={self.current_trace.duration_seconds():.1f}s"
        )

        completed_trace = self.current_trace
        self.current_trace = None

        return completed_trace

    def _persist_trace(self, trace: ReasoningTrace):
        """
        Persist trace to disk.

        Uses JSONL format for efficient append-only writes.
        """
        traces_path = get_traces_path()

        try:
            with open(traces_path, 'a') as f:
                f.write(json.dumps(trace.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to persist trace: {e}")

    def get_reasoning_summary(self) -> str:
        """
        Generate human-readable reasoning trace summary.

        Returns:
            Formatted summary string
        """
        if not self.current_trace:
            return "No active reasoning trace"

        lines = [
            f"Reasoning Trace: {self.current_trace.task[:60]}...",
            f"Duration: {self.current_trace.duration_seconds():.1f}s",
            f"Steps: {len(self.current_trace.steps)}",
            "",
            "Step-by-step breakdown:"
        ]

        for i, step in enumerate(self.current_trace.steps, 1):
            lines.append(
                f"{i}. [{step.step_type}] (conf={step.confidence:.2f}) {step.content[:50]}..."
            )

        if self.current_trace.quality_score:
            q = self.current_trace.quality_score
            lines.extend([
                "",
                f"Quality: {q.overall:.2f} (coherence={q.coherence:.2f}, "
                f"completeness={q.completeness:.2f}, efficiency={q.efficiency:.2f})"
            ])

        return '\n'.join(lines)

    def get_quality_stats(self) -> Dict[str, Any]:
        """
        Get statistics about reasoning quality over time.

        Returns:
            Dictionary with quality metrics
        """
        if not self.quality_history:
            return {"count": 0}

        recent = self.quality_history[-10:]  # Last 10 traces

        avg_overall = sum(q.overall for q in recent) / len(recent)
        avg_coherence = sum(q.coherence for q in recent) / len(recent)
        avg_completeness = sum(q.completeness for q in recent) / len(recent)
        avg_efficiency = sum(q.efficiency for q in recent) / len(recent)

        return {
            "count": len(self.quality_history),
            "recent_count": len(recent),
            "avg_overall": avg_overall,
            "avg_coherence": avg_coherence,
            "avg_completeness": avg_completeness,
            "avg_efficiency": avg_efficiency,
        }
