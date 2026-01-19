"""
SENSE-v2 ReasoningGenome
Genome representation for the evolutionary architecture with reasoning optimization.

Part of Sprint 1: The Core
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import random
import copy
import json
from datetime import datetime


@dataclass
class Genome(ABC):
    """
    Abstract base class for all genome types in the SENSE evolutionary system.

    Genomes represent the heritable traits of an agent that can be
    mutated, crossed over, and selected upon.
    """
    generation_id: int = 0
    fitness_history: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    parent_ids: List[str] = field(default_factory=list)

    @property
    @abstractmethod
    def genome_id(self) -> str:
        """Unique identifier for this genome instance."""
        pass

    @abstractmethod
    def mutate(self, mutation_rate: float = 0.05) -> "Genome":
        """
        Apply mutations to create a new genome variant.

        Args:
            mutation_rate: Probability of mutation for each gene

        Returns:
            New mutated genome instance
        """
        pass

    @abstractmethod
    def crossover(self, other: "Genome") -> "Genome":
        """
        Perform crossover with another genome to create offspring.

        Args:
            other: Partner genome for crossover

        Returns:
            New child genome
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Genome":
        """Deserialize genome from dictionary."""
        pass

    @property
    def current_fitness(self) -> float:
        """Get most recent fitness value."""
        if not self.fitness_history:
            return 0.0
        return self.fitness_history[-1]

    @property
    def best_fitness(self) -> float:
        """Get best fitness ever achieved."""
        if not self.fitness_history:
            return 0.0
        return max(self.fitness_history)

    @property
    def fitness_trend(self) -> float:
        """
        Calculate fitness trend over recent history.
        Positive = improving, Negative = degrading.
        """
        if len(self.fitness_history) < 2:
            return 0.0
        recent = self.fitness_history[-5:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)

    def record_fitness(self, fitness: float) -> None:
        """Record a fitness evaluation."""
        self.fitness_history.append(fitness)
        # Keep history bounded
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-100:]


@dataclass
class ReasoningGenome(Genome):
    """
    Genome optimized for reasoning-intensive agents.

    Contains three optimization vectors:
    1. Model Weights: Hyperparameters for LoRA/fine-tuning
    2. Reasoning Depth: Token budget and verification loops
    3. Physical Grounding: Sensor alignment strategies

    Attributes:
        base_model_id: Hash identifier for foundational LLM
        hyperparameters: Training parameters {learning_rate, lora_rank, temperature}
        reasoning_budget: Maximum tokens for "Thinking" steps (mutable)
        thinking_patterns: Successful system prompt fragments
        verification_depth: Mandatory self-correction loops
        generation_id: Lineage tracking
        fitness_history: For backward transfer calculation
    """

    # Model identification
    base_model_id: str = ""

    # Hyperparameters for training/inference
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 1e-4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    })

    # Reasoning configuration
    reasoning_budget: int = 1024  # Max tokens for thinking steps
    thinking_patterns: List[str] = field(default_factory=lambda: [
        "Let me analyze this step by step.",
        "First, I need to understand the problem.",
        "Let me verify my reasoning.",
    ])
    verification_depth: int = 1  # Number of self-correction loops

    # Grounding configuration
    grounding_weight: float = 1.5  # Weight for sensor alignment in fitness
    hallucination_threshold: float = 0.5  # Below this, flag as hallucination
    sensor_trust_level: float = 0.8  # How much to trust sensor data vs model

    # Metadata
    _genome_hash: Optional[str] = field(default=None, repr=False)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.base_model_id:
            self.base_model_id = self._generate_base_model_id()
        self._genome_hash = None

    def _generate_base_model_id(self) -> str:
        """Generate a hash ID for the base model configuration."""
        config_str = json.dumps({
            "hyperparameters": self.hyperparameters,
            "reasoning_budget": self.reasoning_budget,
            "verification_depth": self.verification_depth,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    @property
    def genome_id(self) -> str:
        """Unique identifier combining model ID and generation."""
        if self._genome_hash is None:
            config_str = json.dumps(self.to_dict(), sort_keys=True)
            self._genome_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return f"{self.base_model_id}_gen{self.generation_id}_{self._genome_hash}"

    def mutate(self, mutation_rate: float = 0.05, drift_metric: float = 0.0) -> "ReasoningGenome":
        """
        Apply mutations with adaptive rates based on drift.

        HIGH drift (>0.5) -> Increase reasoning_budget by 10-20%
        LOW drift (<0.2) -> Decrease reasoning_budget for efficiency

        Args:
            mutation_rate: Base mutation probability
            drift_metric: Current concept drift level (0-1)

        Returns:
            New mutated ReasoningGenome
        """
        # Deep copy for mutation
        new_hyperparams = copy.deepcopy(self.hyperparameters)
        new_patterns = copy.deepcopy(self.thinking_patterns)

        # Mutate hyperparameters
        if random.random() < mutation_rate:
            # Learning rate mutation (log scale)
            lr = new_hyperparams.get("learning_rate", 1e-4)
            new_hyperparams["learning_rate"] = lr * random.uniform(0.5, 2.0)
            new_hyperparams["learning_rate"] = max(1e-6, min(1e-2, new_hyperparams["learning_rate"]))

        if random.random() < mutation_rate:
            # LoRA rank mutation
            rank = new_hyperparams.get("lora_rank", 16)
            new_hyperparams["lora_rank"] = max(4, min(128, rank + random.choice([-8, -4, 4, 8])))

        if random.random() < mutation_rate:
            # Temperature mutation
            temp = new_hyperparams.get("temperature", 0.7)
            new_hyperparams["temperature"] = max(0.1, min(2.0, temp + random.gauss(0, 0.1)))

        # Adaptive reasoning budget mutation based on drift
        new_reasoning_budget = self.reasoning_budget
        if drift_metric > 0.5:
            # HIGH drift: Increase reasoning budget (10-20%)
            increase = random.uniform(0.1, 0.2)
            new_reasoning_budget = int(self.reasoning_budget * (1 + increase))
        elif drift_metric < 0.2 and random.random() < mutation_rate:
            # LOW drift: Decrease for efficiency (5-10%)
            decrease = random.uniform(0.05, 0.1)
            new_reasoning_budget = int(self.reasoning_budget * (1 - decrease))
        elif random.random() < mutation_rate:
            # Random mutation
            new_reasoning_budget = int(self.reasoning_budget * random.uniform(0.9, 1.1))

        # Bound reasoning budget
        new_reasoning_budget = max(256, min(8192, new_reasoning_budget))

        # Mutate verification depth
        new_verification_depth = self.verification_depth
        if random.random() < mutation_rate:
            new_verification_depth = max(0, min(5, self.verification_depth + random.choice([-1, 1])))

        # Mutate thinking patterns (rarely)
        if random.random() < mutation_rate * 0.5:
            if new_patterns and random.random() < 0.5:
                # Remove a random pattern
                idx = random.randint(0, len(new_patterns) - 1)
                new_patterns.pop(idx)
            else:
                # Add variation of existing pattern
                if new_patterns:
                    base_pattern = random.choice(new_patterns)
                    variations = [
                        f"Additionally, {base_pattern.lower()}",
                        f"To be thorough, {base_pattern.lower()}",
                        base_pattern.replace("step by step", "methodically"),
                    ]
                    new_patterns.append(random.choice(variations))

        # Mutate grounding parameters
        new_grounding_weight = self.grounding_weight
        new_hallucination_threshold = self.hallucination_threshold

        if random.random() < mutation_rate:
            new_grounding_weight = max(0.5, min(3.0, self.grounding_weight + random.gauss(0, 0.2)))

        if random.random() < mutation_rate:
            new_hallucination_threshold = max(0.2, min(0.8, self.hallucination_threshold + random.gauss(0, 0.1)))

        return ReasoningGenome(
            base_model_id=self.base_model_id,
            hyperparameters=new_hyperparams,
            reasoning_budget=new_reasoning_budget,
            thinking_patterns=new_patterns,
            verification_depth=new_verification_depth,
            grounding_weight=new_grounding_weight,
            hallucination_threshold=new_hallucination_threshold,
            sensor_trust_level=self.sensor_trust_level,
            generation_id=self.generation_id + 1,
            fitness_history=[],
            parent_ids=[self.genome_id],
            tags=copy.deepcopy(self.tags),
        )

    def crossover(self, other: "ReasoningGenome") -> "ReasoningGenome":
        """
        Merge two genomes, weighted by fitness.

        Thinking patterns are merged preferring patterns from
        the higher-fitness parent.

        Args:
            other: Partner genome

        Returns:
            Child genome with mixed traits
        """
        # Determine fitness weights
        self_fitness = self.current_fitness if self.fitness_history else 0.5
        other_fitness = other.current_fitness if other.fitness_history else 0.5
        total_fitness = self_fitness + other_fitness + 1e-8

        self_weight = self_fitness / total_fitness
        other_weight = other_fitness / total_fitness

        # Merge hyperparameters
        merged_hyperparams = {}
        for key in self.hyperparameters:
            if key in other.hyperparameters:
                self_val = self.hyperparameters[key]
                other_val = other.hyperparameters[key]
                if isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                    merged_hyperparams[key] = self_val * self_weight + other_val * other_weight
                else:
                    # Non-numeric: random selection
                    merged_hyperparams[key] = random.choice([self_val, other_val])
            else:
                merged_hyperparams[key] = self.hyperparameters[key]

        # Ensure integer types where needed
        if "lora_rank" in merged_hyperparams:
            merged_hyperparams["lora_rank"] = int(merged_hyperparams["lora_rank"])

        # Merge reasoning budget
        merged_budget = int(self.reasoning_budget * self_weight + other.reasoning_budget * other_weight)

        # Merge verification depth (weighted random)
        if random.random() < self_weight:
            merged_verification = self.verification_depth
        else:
            merged_verification = other.verification_depth

        # Merge thinking patterns - combine and dedupe, prefer high-fitness parent
        all_patterns = []
        if self_weight >= other_weight:
            all_patterns = self.thinking_patterns + other.thinking_patterns
        else:
            all_patterns = other.thinking_patterns + self.thinking_patterns

        # Deduplicate while preserving order
        seen = set()
        merged_patterns = []
        for pattern in all_patterns:
            if pattern not in seen:
                seen.add(pattern)
                merged_patterns.append(pattern)

        # Limit pattern count
        merged_patterns = merged_patterns[:10]

        # Merge grounding parameters
        merged_grounding_weight = self.grounding_weight * self_weight + other.grounding_weight * other_weight
        merged_hallucination_threshold = (
            self.hallucination_threshold * self_weight +
            other.hallucination_threshold * other_weight
        )

        return ReasoningGenome(
            base_model_id=self.base_model_id,  # Keep same base model
            hyperparameters=merged_hyperparams,
            reasoning_budget=merged_budget,
            thinking_patterns=merged_patterns,
            verification_depth=merged_verification,
            grounding_weight=merged_grounding_weight,
            hallucination_threshold=merged_hallucination_threshold,
            sensor_trust_level=(self.sensor_trust_level + other.sensor_trust_level) / 2,
            generation_id=max(self.generation_id, other.generation_id) + 1,
            fitness_history=[],
            parent_ids=[self.genome_id, other.genome_id],
            tags=list(set(self.tags + other.tags)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary for LTM persistence."""
        return {
            "genome_type": "ReasoningGenome",
            "genome_id": self.genome_id,
            "base_model_id": self.base_model_id,
            "hyperparameters": self.hyperparameters,
            "reasoning_budget": self.reasoning_budget,
            "thinking_patterns": self.thinking_patterns,
            "verification_depth": self.verification_depth,
            "grounding_weight": self.grounding_weight,
            "hallucination_threshold": self.hallucination_threshold,
            "sensor_trust_level": self.sensor_trust_level,
            "generation_id": self.generation_id,
            "fitness_history": self.fitness_history,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "notes": self.notes,
            "current_fitness": self.current_fitness,
            "best_fitness": self.best_fitness,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningGenome":
        """Deserialize genome from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        return cls(
            base_model_id=data.get("base_model_id", ""),
            hyperparameters=data.get("hyperparameters", {}),
            reasoning_budget=data.get("reasoning_budget", 1024),
            thinking_patterns=data.get("thinking_patterns", []),
            verification_depth=data.get("verification_depth", 1),
            grounding_weight=data.get("grounding_weight", 1.5),
            hallucination_threshold=data.get("hallucination_threshold", 0.5),
            sensor_trust_level=data.get("sensor_trust_level", 0.8),
            generation_id=data.get("generation_id", 0),
            fitness_history=data.get("fitness_history", []),
            parent_ids=data.get("parent_ids", []),
            created_at=created_at,
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )

    def get_embedding_vector(self) -> List[float]:
        """
        Generate embedding vector for FAISS storage.

        Encodes hyperparameters into a fixed-size vector for
        similarity search in PopulationManager.

        Returns:
            List of float values representing genome configuration
        """
        vector = [
            self.hyperparameters.get("learning_rate", 1e-4) * 10000,  # Scale to reasonable range
            self.hyperparameters.get("lora_rank", 16) / 128,  # Normalize
            self.hyperparameters.get("temperature", 0.7),
            self.hyperparameters.get("top_p", 0.9),
            self.reasoning_budget / 8192,  # Normalize
            self.verification_depth / 5,  # Normalize
            self.grounding_weight / 3,  # Normalize
            self.hallucination_threshold,
            self.sensor_trust_level,
            self.current_fitness,
            self.best_fitness,
            len(self.thinking_patterns) / 10,  # Normalize
            self.generation_id / 100,  # Normalize
        ]
        return vector

    def calculate_backward_transfer(self, historical_tasks: List[Tuple[str, float]]) -> float:
        """
        Calculate backward transfer metric.

        Measures how much performance on previously learned tasks
        has degraded (catastrophic forgetting).

        Args:
            historical_tasks: List of (task_id, original_score) tuples

        Returns:
            Backward transfer score (0 = no forgetting, 1 = complete forgetting)
        """
        if not historical_tasks or not self.fitness_history:
            return 0.0

        # This would need re-evaluation on historical tasks
        # For now, estimate from fitness trend
        if len(self.fitness_history) < 5:
            return 0.0

        # Compare recent performance to historical peak
        recent_avg = sum(self.fitness_history[-5:]) / 5
        historical_peak = max(self.fitness_history[:-5]) if len(self.fitness_history) > 5 else recent_avg

        if historical_peak <= 0:
            return 0.0

        degradation = max(0, (historical_peak - recent_avg) / historical_peak)
        return min(1.0, degradation)

    def get_reasoning_config(self) -> Dict[str, Any]:
        """
        Get configuration for the AdaptiveReasoningBudget system.

        Returns:
            Dictionary with reasoning parameters
        """
        return {
            "base_budget": self.reasoning_budget,
            "verification_depth": self.verification_depth,
            "thinking_patterns": self.thinking_patterns,
            "temperature": self.hyperparameters.get("temperature", 0.7),
            "grounding_weight": self.grounding_weight,
            "hallucination_threshold": self.hallucination_threshold,
        }


def create_random_genome(
    base_model_id: str = "",
    generation_id: int = 0,
) -> ReasoningGenome:
    """
    Create a new random genome for population initialization.

    Args:
        base_model_id: Identifier for the base model
        generation_id: Generation number

    Returns:
        Randomly initialized ReasoningGenome
    """
    return ReasoningGenome(
        base_model_id=base_model_id,
        hyperparameters={
            "learning_rate": random.uniform(1e-5, 1e-3),
            "lora_rank": random.choice([8, 16, 32, 64]),
            "lora_alpha": random.choice([16, 32, 64]),
            "temperature": random.uniform(0.5, 1.0),
            "top_p": random.uniform(0.8, 1.0),
            "top_k": random.choice([20, 50, 100]),
            "repetition_penalty": random.uniform(1.0, 1.3),
        },
        reasoning_budget=random.choice([512, 1024, 2048, 4096]),
        thinking_patterns=[
            "Let me analyze this step by step.",
            random.choice([
                "First, I need to understand the problem.",
                "Let me break this down into components.",
                "I'll start by identifying the key elements.",
            ]),
        ],
        verification_depth=random.choice([0, 1, 2]),
        grounding_weight=random.uniform(1.0, 2.0),
        hallucination_threshold=random.uniform(0.4, 0.6),
        sensor_trust_level=random.uniform(0.7, 0.9),
        generation_id=generation_id,
    )
