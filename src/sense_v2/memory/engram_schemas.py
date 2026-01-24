"""
SENSE-v2 Engram Schemas
Data structures for reasoning traces and memory integration.

Part of Sprint 2: The Brain

Defines schemas for:
- ReasoningTrace: Captures reasoning process for replay/analysis
- GroundingRecord: Tracks sensor alignment
- DriftSnapshot: Records concept drift state
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import hashlib
import numpy as np


@dataclass
class DriftSnapshot:
    """Snapshot of drift metrics at a point in time."""
    drift_level: float  # 0.0 = stable, 1.0 = high drift
    fitness_variance: float
    task_distribution_shift: float
    sensor_accuracy_delta: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_level": self.drift_level,
            "fitness_variance": self.fitness_variance,
            "task_distribution_shift": self.task_distribution_shift,
            "sensor_accuracy_delta": self.sensor_accuracy_delta,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftSnapshot":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        return cls(
            drift_level=data.get("drift_level", 0.0),
            fitness_variance=data.get("fitness_variance", 0.0),
            task_distribution_shift=data.get("task_distribution_shift", 0.0),
            sensor_accuracy_delta=data.get("sensor_accuracy_delta", 0.0),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class GroundingRecord:
    """Record of sensor grounding verification."""
    claim: str  # The claim being verified
    claimed_value: Optional[float]
    ground_truth: Optional[float]
    grounding_score: float  # 0.0 = no match, 1.0 = perfect match
    sensor_name: str
    is_valid: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "claimed_value": self.claimed_value,
            "ground_truth": self.ground_truth,
            "grounding_score": self.grounding_score,
            "sensor_name": self.sensor_name,
            "is_valid": self.is_valid,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReasoningTrace:
    """
    Complete trace of a reasoning process.

    Captures all aspects of an agent's reasoning for:
    - Retrieval-augmented reasoning (find similar past solutions)
    - Training data generation (process supervision)
    - Debugging and analysis
    - Backward transfer detection

    Attributes:
        problem_embedding: Vector for retrieval-augmented reasoning
        thought_chain: Intermediate reasoning steps
        outcome: Success/Failure of the reasoning
        grounding_score: Alignment with sensor data
        reasoning_tokens_used: Actual tokens consumed
        generation_id: Which generation produced this
        drift_context: Snapshot of drift metrics
        timestamp: When this trace was created
    """
    # Identification
    trace_id: str = field(default_factory=lambda: "")
    task_id: str = ""
    genome_id: str = ""
    generation_id: int = 0

    # Problem representation
    problem_description: str = ""
    problem_embedding: Optional[List[float]] = None  # For FAISS retrieval

    # Reasoning process
    thought_chain: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_results: List[Any] = field(default_factory=list)

    # Outcome
    outcome: bool = False  # Success/Failure
    final_answer: str = ""
    error_message: str = ""

    # Grounding
    grounding_score: float = 0.0  # Alignment with sensor data
    grounding_records: List[GroundingRecord] = field(default_factory=list)
    hallucination_detected: bool = False

    # Resource usage
    reasoning_tokens_used: int = 0
    total_tokens_used: int = 0
    execution_time_ms: int = 0
    verification_loops: int = 0

    # Context
    drift_context: Optional[DriftSnapshot] = None
    memory_retrievals: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate trace ID if not provided."""
        if not self.trace_id:
            self.trace_id = self._generate_trace_id()

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        content = f"{self.task_id}_{self.genome_id}_{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @property
    def is_efficient(self) -> bool:
        """Check if this trace represents efficient reasoning."""
        # Consider efficient if:
        # - Successful outcome
        # - Tokens below 2x median
        # - Good grounding score
        return self.outcome and self.grounding_score >= 0.7

    @property
    def should_flag_for_ltm(self) -> bool:
        """
        Determine if this trace should be flagged for LTM storage.

        Store if:
        - grounding_score > 0.8 AND outcome == True
        - OR tokens > 2x average (for efficiency analysis)
        """
        if self.grounding_score > 0.8 and self.outcome:
            return True
        return False

    @property
    def should_flag_for_efficiency_analysis(self) -> bool:
        """Check if tokens usage warrants efficiency analysis."""
        # This would need comparison to historical average
        return self.reasoning_tokens_used > 2000  # Placeholder threshold

    def add_thought(self, thought: str) -> None:
        """Add a thought to the reasoning chain."""
        self.thought_chain.append(thought)

    def add_tool_call(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        output: Any,
        success: bool,
    ) -> None:
        """Record a tool call."""
        self.tool_calls.append({
            "tool_name": tool_name,
            "inputs": inputs,
            "output": str(output)[:1000],  # Truncate large outputs
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

    def add_grounding_check(
        self,
        claim: str,
        claimed_value: Optional[float],
        ground_truth: Optional[float],
        sensor_name: str,
    ) -> GroundingRecord:
        """
        Add a grounding verification record.

        Args:
            claim: The claim being verified
            claimed_value: Value claimed by the model
            ground_truth: Actual value from sensor
            sensor_name: Name of the sensor

        Returns:
            The created GroundingRecord
        """
        # Calculate grounding score
        if ground_truth is None or claimed_value is None:
            score = 0.5  # Unknown
            is_valid = True
        else:
            if abs(ground_truth) < 1e-8:
                score = 1.0 if abs(claimed_value) < 1e-8 else 0.0
            else:
                error = abs(claimed_value - ground_truth) / max(abs(ground_truth), abs(claimed_value))
                score = max(0.0, 1.0 - error)
            is_valid = score >= 0.5

        record = GroundingRecord(
            claim=claim,
            claimed_value=claimed_value,
            ground_truth=ground_truth,
            grounding_score=score,
            sensor_name=sensor_name,
            is_valid=is_valid,
        )

        self.grounding_records.append(record)

        # Update overall grounding score
        if self.grounding_records:
            self.grounding_score = sum(r.grounding_score for r in self.grounding_records) / len(self.grounding_records)

        # Check for hallucination
        if score < 0.5:
            self.hallucination_detected = True

        return record

    def finalize(
        self,
        outcome: bool,
        final_answer: str = "",
        error_message: str = "",
        reasoning_tokens: int = 0,
        total_tokens: int = 0,
        execution_time_ms: int = 0,
    ) -> None:
        """
        Finalize the trace with outcome information.

        Args:
            outcome: Whether reasoning was successful
            final_answer: The final answer produced
            error_message: Error message if failed
            reasoning_tokens: Tokens used for reasoning
            total_tokens: Total tokens used
            execution_time_ms: Execution time in milliseconds
        """
        self.outcome = outcome
        self.final_answer = final_answer
        self.error_message = error_message
        self.reasoning_tokens_used = reasoning_tokens
        self.total_tokens_used = total_tokens
        self.execution_time_ms = execution_time_ms

    def set_drift_context(
        self,
        drift_level: float,
        fitness_variance: float = 0.0,
        task_shift: float = 0.0,
        sensor_delta: float = 0.0,
    ) -> None:
        """Set drift context snapshot."""
        self.drift_context = DriftSnapshot(
            drift_level=drift_level,
            fitness_variance=fitness_variance,
            task_distribution_shift=task_shift,
            sensor_accuracy_delta=sensor_delta,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "genome_id": self.genome_id,
            "generation_id": self.generation_id,
            "problem_description": self.problem_description,
            "problem_embedding": self.problem_embedding,
            "thought_chain": self.thought_chain,
            "tool_calls": self.tool_calls,
            "intermediate_results": [str(r) for r in self.intermediate_results],
            "outcome": self.outcome,
            "final_answer": self.final_answer,
            "error_message": self.error_message,
            "grounding_score": self.grounding_score,
            "grounding_records": [r.to_dict() for r in self.grounding_records],
            "hallucination_detected": self.hallucination_detected,
            "reasoning_tokens_used": self.reasoning_tokens_used,
            "total_tokens_used": self.total_tokens_used,
            "execution_time_ms": self.execution_time_ms,
            "verification_loops": self.verification_loops,
            "drift_context": self.drift_context.to_dict() if self.drift_context else None,
            "memory_retrievals": self.memory_retrievals,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_efficient": self.is_efficient,
            "should_flag_for_ltm": self.should_flag_for_ltm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningTrace":
        """Deserialize trace from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        drift_context = None
        if data.get("drift_context"):
            drift_context = DriftSnapshot.from_dict(data["drift_context"])

        grounding_records = []
        for r in data.get("grounding_records", []):
            grounding_records.append(GroundingRecord(
                claim=r.get("claim", ""),
                claimed_value=r.get("claimed_value"),
                ground_truth=r.get("ground_truth"),
                grounding_score=r.get("grounding_score", 0.0),
                sensor_name=r.get("sensor_name", ""),
                is_valid=r.get("is_valid", True),
            ))

        return cls(
            trace_id=data.get("trace_id", ""),
            task_id=data.get("task_id", ""),
            genome_id=data.get("genome_id", ""),
            generation_id=data.get("generation_id", 0),
            problem_description=data.get("problem_description", ""),
            problem_embedding=data.get("problem_embedding"),
            thought_chain=data.get("thought_chain", []),
            tool_calls=data.get("tool_calls", []),
            intermediate_results=data.get("intermediate_results", []),
            outcome=data.get("outcome", False),
            final_answer=data.get("final_answer", ""),
            error_message=data.get("error_message", ""),
            grounding_score=data.get("grounding_score", 0.0),
            grounding_records=grounding_records,
            hallucination_detected=data.get("hallucination_detected", False),
            reasoning_tokens_used=data.get("reasoning_tokens_used", 0),
            total_tokens_used=data.get("total_tokens_used", 0),
            execution_time_ms=data.get("execution_time_ms", 0),
            verification_loops=data.get("verification_loops", 0),
            drift_context=drift_context,
            memory_retrievals=data.get("memory_retrievals", []),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )

    def get_embedding_for_storage(self) -> List[float]:
        """
        Get embedding vector for FAISS storage.

        Uses problem embedding if available, otherwise generates
        a feature-based embedding.
        """
        if self.problem_embedding:
            return self.problem_embedding

        # Generate feature-based embedding
        features = [
            self.grounding_score,
            1.0 if self.outcome else 0.0,
            min(1.0, self.reasoning_tokens_used / 4096),  # Normalize
            min(1.0, len(self.thought_chain) / 10),
            min(1.0, len(self.tool_calls) / 5),
            1.0 if self.hallucination_detected else 0.0,
            self.drift_context.drift_level if self.drift_context else 0.5,
        ]

        # Pad to fixed size
        while len(features) < 16:
            features.append(0.0)

        return features[:16]


def create_reasoning_trace(
    task_id: str,
    genome_id: str,
    generation_id: int,
    problem_description: str,
) -> ReasoningTrace:
    """
    Factory function to create a new reasoning trace.

    Args:
        task_id: ID of the task being solved
        genome_id: ID of the genome doing the reasoning
        generation_id: Current generation number
        problem_description: Description of the problem

    Returns:
        New ReasoningTrace instance
    """
    return ReasoningTrace(
        task_id=task_id,
        genome_id=genome_id,
        generation_id=generation_id,
        problem_description=problem_description,
    )
