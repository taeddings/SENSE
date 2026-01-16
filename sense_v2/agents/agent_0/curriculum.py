"""
SENSE-v2 Curriculum Agent (Teacher)
Generates progressive curriculum for the ExecutorAgent to learn from.
Part of Agent 0 - The School co-evolutionary system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import logging
import asyncio
import random
from datetime import datetime

from sense_v2.core.base import BaseAgent, BaseTool, AgentState
from sense_v2.core.schemas import AgentMessage, MessageRole, RewardSignal
from sense_v2.core.config import EvolutionConfig


class TaskDifficulty(Enum):
    """Difficulty levels for curriculum tasks."""
    TRIVIAL = 0.1
    EASY = 0.3
    MEDIUM = 0.5
    HARD = 0.7
    EXPERT = 0.9


@dataclass
class CurriculumTask:
    """A single task in the curriculum."""
    task_id: str
    description: str
    difficulty: float
    category: str
    expected_tools: List[str] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    timeout_seconds: int = 60
    created_at: datetime = field(default_factory=datetime.now)

    # Tracking
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def record_attempt(self, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "difficulty": self.difficulty,
            "category": self.category,
            "expected_tools": self.expected_tools,
            "test_cases": self.test_cases,
            "hints": self.hints,
            "timeout_seconds": self.timeout_seconds,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.success_rate,
        }


@dataclass
class CurriculumStage:
    """A stage in the curriculum progression."""
    stage_id: int
    name: str
    difficulty_range: tuple  # (min, max)
    required_success_rate: float = 0.8
    tasks: List[CurriculumTask] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if stage requirements are met."""
        if not self.tasks:
            return False
        total_successes = sum(t.successes for t in self.tasks)
        total_attempts = sum(t.attempts for t in self.tasks)
        if total_attempts == 0:
            return False
        return (total_successes / total_attempts) >= self.required_success_rate


class CurriculumAgent(BaseAgent):
    """
    Curriculum Agent (Teacher) for Agent 0 - The School.

    Responsibilities:
    - Generate progressively harder tasks
    - Adapt curriculum based on student performance
    - Provide hints when student struggles
    - Track overall curriculum progress

    Per SYSTEM_PROMPT requirements:
    - CurriculumAgent and ExecutorAgent must be distinct
    - Part of symbiotic loop with verified success
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        task_generator: Optional[Callable[[float, str], CurriculumTask]] = None,
    ):
        super().__init__(name="CurriculumAgent", config=config)
        self.config = config or EvolutionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Curriculum state
        self.stages: List[CurriculumStage] = []
        self.current_stage_idx: int = 0
        self.task_history: List[CurriculumTask] = []

        # Custom task generator (or use default)
        self._task_generator = task_generator or self._default_task_generator

        # Performance tracking
        self.student_performance: Dict[str, float] = {}
        self._difficulty_adjustment = 0.0

        # Initialize stages
        self._initialize_stages()

    def _initialize_stages(self) -> None:
        """Initialize curriculum stages."""
        stage_configs = [
            (0, "Foundation", (0.1, 0.3), 0.9),
            (1, "Basic Operations", (0.2, 0.5), 0.85),
            (2, "Intermediate", (0.4, 0.7), 0.8),
            (3, "Advanced", (0.6, 0.85), 0.75),
            (4, "Expert", (0.8, 1.0), 0.7),
        ]

        for stage_id, name, diff_range, req_rate in stage_configs[:self.config.curriculum_stages]:
            self.stages.append(CurriculumStage(
                stage_id=stage_id,
                name=name,
                difficulty_range=diff_range,
                required_success_rate=req_rate,
            ))

    @property
    def current_stage(self) -> Optional[CurriculumStage]:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    def _default_task_generator(self, difficulty: float, category: str) -> CurriculumTask:
        """
        Default task generator. Override with LLM-based generation in production.
        """
        task_templates = {
            "terminal": [
                ("List files in current directory", ["terminal_exec"], "ls"),
                ("Show current working directory", ["terminal_exec"], "pwd"),
                ("Create a new directory", ["terminal_exec"], "mkdir"),
                ("Read file contents", ["file_read"], "cat"),
                ("Search for pattern in files", ["terminal_exec"], "grep"),
            ],
            "filesystem": [
                ("Read a configuration file", ["file_read"], None),
                ("Write data to a file", ["file_write"], None),
                ("List directory contents", ["file_list"], None),
                ("Check if file exists", ["file_exists"], None),
            ],
            "reasoning": [
                ("Analyze error message and suggest fix", ["terminal_exec"], None),
                ("Plan multi-step task execution", [], None),
                ("Debug failing test case", ["terminal_exec", "file_read"], None),
            ],
        }

        templates = task_templates.get(category, task_templates["terminal"])
        template = random.choice(templates)

        task_id = f"{category}_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"

        return CurriculumTask(
            task_id=task_id,
            description=template[0],
            difficulty=difficulty,
            category=category,
            expected_tools=template[1],
            timeout_seconds=int(60 + (difficulty * 120)),
        )

    async def generate_task(self, category: Optional[str] = None) -> CurriculumTask:
        """
        Generate a new task appropriate for current curriculum stage.

        Args:
            category: Optional task category (terminal, filesystem, reasoning)

        Returns:
            A new CurriculumTask
        """
        stage = self.current_stage
        if stage is None:
            # Default to medium difficulty if no stage
            difficulty = 0.5
        else:
            # Sample difficulty within stage range, adjusted by performance
            min_diff, max_diff = stage.difficulty_range
            base_difficulty = random.uniform(min_diff, max_diff)
            difficulty = max(0.1, min(1.0, base_difficulty + self._difficulty_adjustment))

        # Default category selection
        if category is None:
            categories = ["terminal", "filesystem", "reasoning"]
            weights = [0.5, 0.3, 0.2]  # Terminal tasks are most common initially
            category = random.choices(categories, weights=weights)[0]

        task = self._task_generator(difficulty, category)

        # Add to current stage
        if stage:
            stage.tasks.append(task)

        self.task_history.append(task)
        self.logger.info(f"Generated task {task.task_id} (difficulty: {difficulty:.2f})")

        return task

    async def process_result(self, task: CurriculumTask, reward: RewardSignal) -> Dict[str, Any]:
        """
        Process the result of a task attempt.

        Args:
            task: The completed task
            reward: Reward signal from task execution

        Returns:
            Feedback dictionary with next steps
        """
        success = reward.value >= 0.5 if not reward.binary else reward.value == 1.0
        task.record_attempt(success)

        # Update performance tracking
        self.student_performance[task.category] = (
            self.student_performance.get(task.category, 0.5) * 0.7 +
            reward.value * 0.3
        )

        # Adjust difficulty based on performance
        if success:
            self._difficulty_adjustment = min(0.2, self._difficulty_adjustment + 0.02)
        else:
            self._difficulty_adjustment = max(-0.2, self._difficulty_adjustment - 0.05)

        # Check for stage progression
        stage_advanced = False
        if self.current_stage and self.current_stage.is_complete():
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                stage_advanced = True
                self.logger.info(f"Advanced to stage {self.current_stage_idx}: {self.current_stage.name}")

        feedback = {
            "task_id": task.task_id,
            "success": success,
            "reward": reward.value,
            "task_success_rate": task.success_rate,
            "stage_advanced": stage_advanced,
            "current_stage": self.current_stage.name if self.current_stage else "Complete",
            "difficulty_adjustment": self._difficulty_adjustment,
        }

        # Generate hints if struggling
        if not success and task.attempts >= 2:
            feedback["hints"] = self._generate_hints(task)

        return feedback

    def _generate_hints(self, task: CurriculumTask) -> List[str]:
        """Generate hints for a struggling student."""
        hints = []

        if task.expected_tools:
            hints.append(f"Consider using: {', '.join(task.expected_tools)}")

        if task.difficulty > 0.5:
            hints.append("Break the problem into smaller steps")

        hints.append("Review the error messages carefully")

        return hints

    async def get_curriculum_status(self) -> Dict[str, Any]:
        """Get overall curriculum status."""
        total_tasks = len(self.task_history)
        total_successes = sum(1 for t in self.task_history if t.success_rate >= 0.5)

        return {
            "current_stage": self.current_stage_idx,
            "stage_name": self.current_stage.name if self.current_stage else "Complete",
            "total_stages": len(self.stages),
            "total_tasks_generated": total_tasks,
            "overall_success_rate": total_successes / total_tasks if total_tasks > 0 else 0,
            "category_performance": self.student_performance,
            "difficulty_adjustment": self._difficulty_adjustment,
            "stages": [
                {
                    "id": s.stage_id,
                    "name": s.name,
                    "complete": s.is_complete(),
                    "tasks": len(s.tasks),
                }
                for s in self.stages
            ],
        }

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message (interface compliance)."""
        # Parse message for task requests
        content = message.content.lower()

        if "generate" in content or "new task" in content:
            task = await self.generate_task()
            return AgentMessage.assistant(
                f"Generated task: {task.description}\n"
                f"Difficulty: {task.difficulty:.2f}\n"
                f"Category: {task.category}"
            )

        return AgentMessage.assistant("Ready to generate curriculum tasks.")

    async def run(self) -> None:
        """Main agent loop (interface compliance)."""
        self._is_running = True
        self.logger.info("CurriculumAgent started")

        while self._is_running:
            await asyncio.sleep(1)
