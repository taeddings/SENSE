"""Curriculum Agent for SENSE v3.0

Generates adaptive tasks for self-evolution training.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import logging
import asyncio
import random
from ...llm.model_backend import get_model  # LLM for task generation (stubbed)

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class CurriculumAgent:
    """
    Curriculum Agent: Generates tasks with progressive difficulty for evolution training.
    Integrates with PopulationManager to feed tasks to genomes.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.current_stage = 0
        self.task_history = []
        self.logger = logging.getLogger("CurriculumAgent")
        self.llm = get_model(self.config)  # LLM for dynamic task generation

    async def generate_task(self, difficulty: Optional[Difficulty] = None) -> str:
        """Generate a task based on difficulty or current stage."""
        if difficulty is None:
            difficulty = self._get_current_difficulty()
        templates = self._get_templates(difficulty)
        template = random.choice(templates)
        params = self._generate_params(difficulty)
        task = template.format(**params)
        self.task_history.append(task)
        self.logger.info(f"Generated task (stage {self.current_stage}, difficulty {difficulty.value}): {task}")
        return task

    def _get_templates(self, difficulty: Difficulty) -> List[str]:
        templates = {
            Difficulty.EASY: [
                "Calculate {a} + {b}",
                "Is {n} even? (yes/no)",
                "Print 'Hello World' {times} times"
            ],
            Difficulty.MEDIUM: [
                "Find prime factors of {n}",
                "Sort the list {lst}",
                "Reverse the string '{s}' without built-ins"
            ],
            Difficulty.HARD: [
                "Implement binary search for {target} in {arr}",
                "Solve knapsack with capacity {c}, items {items}",
                "Generate Fibonacci up to {n} iteratively"
            ]
        }
        return templates[difficulty]

    def _generate_params(self, difficulty: Difficulty) -> Dict[str, Any]:
        params = {}
        if difficulty == Difficulty.EASY:
            params = {"a": random.randint(1, 10), "b": random.randint(1, 10), "n": random.randint(1, 20), "times": 3}
        elif difficulty == Difficulty.MEDIUM:
            params = {"n": random.randint(20, 100), "lst": list(range(10)), "s": "abcde"}
        else:
            params = {"target": random.randint(1, 50), "arr": list(range(100)), "c": 10, "items": [(2,3), (3,4), (4,5)], "n": 20}
        return params

    def _get_current_difficulty(self) -> Difficulty:
        stage = self.current_stage % 30  # Cycle every 30 stages
        if stage < 10:
            return Difficulty.EASY
        elif stage < 20:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD

    def advance_stage(self):
        """Advance curriculum stage and ramp difficulty."""
        self.current_stage += 1
        if self.current_stage % self.config.get('curriculum_stages', 5) == 0:
            self.config['difficulty_ramp_rate'] += 0.1
            self.logger.info(f"Advanced to stage {self.current_stage}; ramp rate now {self.config['difficulty_ramp_rate']}")

    async def get_next_task(self) -> str:
        """Get the next task for training."""
        task = await self.generate_task()
        self.advance_stage()
        return task

    def reset(self):
        """Reset curriculum for new training run."""
        self.current_stage = 0
        self.task_history = []
        self.logger.info("Curriculum reset")

# Convenience
def create_curriculum(config: Dict[str, Any]) -> CurriculumAgent:
    return CurriculumAgent(config)