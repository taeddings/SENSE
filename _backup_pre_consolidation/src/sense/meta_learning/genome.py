"""
Curriculum Genome.

Encodes the strategy for generating tasks. This genome evolves based on how well
it helps the agent learn.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List
import random
import numpy as np

class DifficultyStrategy(Enum):
    LINEAR = "linear"           # Steady increase
    EXPONENTIAL = "exponential" # Rapid increase
    ADAPTIVE = "adaptive"       # Based on success rate
    SPIRAL = "spiral"           # Revisit topics with higher difficulty

class TopicStrategy(Enum):
    BREADTH = "breadth"         # Wide coverage
    DEPTH = "depth"             # Deep mastery
    WEAKNESS = "weakness"       # Focus on failures

@dataclass
class CurriculumGenome:
    """
    Genes controlling the curriculum generation process.
    """
    # Strategy
    difficulty_strategy: DifficultyStrategy = DifficultyStrategy.LINEAR
    topic_strategy: TopicStrategy = TopicStrategy.BREADTH
    
    # Parameters
    initial_difficulty: float = 0.3
    difficulty_increment: float = 0.05
    task_diversity: float = 0.5  # 0 (repetitive) -> 1 (novel)
    
    # Evolution tracking
    fitness: float = 0.0  # Measured by Agent's Learning Rate
    generation: int = 0

    def mutate(self, rate: float = 0.1):
        """Apply random mutations to the strategy."""
        if random.random() < rate:
            self.difficulty_strategy = random.choice(list(DifficultyStrategy))
        
        if random.random() < rate:
            self.topic_strategy = random.choice(list(TopicStrategy))
            
        if random.random() < rate:
            self.initial_difficulty = np.clip(self.initial_difficulty + random.gauss(0, 0.1), 0.1, 0.9)
            
        if random.random() < rate:
            self.difficulty_increment = np.clip(self.difficulty_increment + random.gauss(0, 0.02), 0.01, 0.2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "difficulty_strategy": self.difficulty_strategy.value,
            "topic_strategy": self.topic_strategy.value,
            "initial_difficulty": self.initial_difficulty,
            "difficulty_increment": self.difficulty_increment,
            "fitness": self.fitness
        }
