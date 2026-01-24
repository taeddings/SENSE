"""
SENSE-v2 Curriculum Agent (Teacher)
Generates progressive curriculum for the ExecutorAgent to learn from.
Part of Agent 0 - The School co-evolutionary system.

Enhanced with Agent0 research patterns:
- Structured question format with validation
- Difficulty calibration based on executor success rate
- Domain-aware task generation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
import logging
import asyncio
import random
import re
from datetime import datetime

from sense_v2.core.base import BaseAgent, BaseTool, AgentState
from sense_v2.core.schemas import AgentMessage, MessageRole, RewardSignal
from sense_v2.core.config import EvolutionConfig


# =============================================================================
# Structured Output Validation (from Agent0 research)
# =============================================================================

def extract_boxed_content(text: str) -> List[str]:
    """
    Extract content from \\boxed{...} patterns.

    Handles nested braces properly.

    Args:
        text: Input text containing boxed content

    Returns:
        List of extracted boxed contents
    """
    results = []
    prefix = r'\boxed{'
    plen = len(prefix)
    i = 0

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
        i = j

    return results


def validate_structured_output(output: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that output follows structured format.

    Checks for:
    - <question>...</question> tags
    - \\boxed{...} for answers
    - Optional <think>...</think> blocks

    Args:
        output: The output string to validate

    Returns:
        Tuple of (is_valid, extracted_data)
    """
    extracted = {
        "questions": [],
        "answers": [],
        "thinking": [],
        "has_question_tags": False,
        "has_boxed_answer": False,
        "has_thinking": False,
    }

    # Extract question tags
    questions = re.findall(r"<question>(.*?)</question>", output, re.DOTALL)
    extracted["questions"] = [q.strip() for q in questions]
    extracted["has_question_tags"] = len(questions) > 0

    # Extract boxed answers
    answers = extract_boxed_content(output)
    extracted["answers"] = answers
    extracted["has_boxed_answer"] = len(answers) > 0

    # Extract thinking blocks
    thinking = re.findall(r"<think>(.*?)</think>", output, re.DOTALL)
    extracted["thinking"] = [t.strip() for t in thinking]
    extracted["has_thinking"] = len(thinking) > 0

    # Valid if has at least question and answer
    is_valid = extracted["has_question_tags"] and extracted["has_boxed_answer"]

    return is_valid, extracted


def format_task_as_structured(
    question: str,
    answer: Optional[str] = None,
    include_thinking_prompt: bool = True
) -> str:
    """
    Format a task in structured format.

    Args:
        question: The question/task description
        answer: Optional expected answer
        include_thinking_prompt: Whether to prompt for thinking

    Returns:
        Formatted task string
    """
    parts = []

    if include_thinking_prompt:
        parts.append("Think step-by-step about this problem before providing your answer.")
        parts.append("Show your reasoning in <think>...</think> blocks.")
        parts.append("")

    parts.append(f"<question>\n{question}\n</question>")

    if answer:
        parts.append("")
        parts.append(f"Expected answer format: \\boxed{{{answer}}}")

    return "\n".join(parts)


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

    # Structured format support (from Agent0)
    expected_answer: Optional[str] = None
    structured_format: bool = True
    domain: str = "general"

    # Tracking
    attempts: int = 0
    successes: int = 0
    format_successes: int = 0  # Track proper format usage

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def format_compliance_rate(self) -> float:
        """Rate at which outputs follow the expected structured format."""
        if self.attempts == 0:
            return 0.0
        return self.format_successes / self.attempts

    def record_attempt(self, success: bool, format_valid: bool = True) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        if format_valid:
            self.format_successes += 1

    def get_structured_description(self) -> str:
        """Get description formatted for structured output."""
        if self.structured_format:
            return format_task_as_structured(
                question=self.description,
                answer=self.expected_answer,
                include_thinking_prompt=True,
            )
        return self.description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "difficulty": self.difficulty,
            "category": self.category,
            "domain": self.domain,
            "expected_tools": self.expected_tools,
            "test_cases": self.test_cases,
            "hints": self.hints,
            "timeout_seconds": self.timeout_seconds,
            "expected_answer": self.expected_answer,
            "structured_format": self.structured_format,
            "attempts": self.attempts,
            "successes": self.successes,
            "format_successes": self.format_successes,
            "success_rate": self.success_rate,
            "format_compliance_rate": self.format_compliance_rate,
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

    Enhanced with Agent0 research:
    - Structured question format with validation
    - Difficulty calibration based on executor success rate
    - Domain-aware task generation
    """

    # Supported domains for task generation
    DOMAINS = [
        "terminal",      # Shell commands, CLI operations
        "filesystem",    # File operations, path handling
        "reasoning",     # Logic, planning, analysis
        "coding",        # Code writing, debugging
        "math",          # Calculations, algorithms
        "data",          # Data processing, transformation
    ]

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        task_generator: Optional[Callable[[float, str], CurriculumTask]] = None,
        target_success_rate: float = 0.7,
        difficulty_learning_rate: float = 0.1,
        enable_structured_format: bool = True,
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

        # Difficulty calibration (from Agent0 research)
        self.target_success_rate = target_success_rate
        self.difficulty_learning_rate = difficulty_learning_rate
        self._domain_difficulties: Dict[str, float] = {
            domain: 0.5 for domain in self.DOMAINS
        }
        self._domain_success_history: Dict[str, List[bool]] = {
            domain: [] for domain in self.DOMAINS
        }

        # Structured format settings
        self.enable_structured_format = enable_structured_format
        self._format_compliance_rate = 0.5

        # Workplace feedback tracking
        self._tool_success_rates: Dict[str, float] = {}
        self._error_patterns: Dict[str, int] = {}
        self._successful_tool_sequences: List[List[str]] = []

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
        Default task generator with domain-aware and structured format support.
        Override with LLM-based generation in production.
        """
        # Enhanced task templates with expected answers and domains
        task_templates = {
            "terminal": [
                ("List files in current directory", ["terminal_exec"], "ls output", "terminal"),
                ("Show current working directory", ["terminal_exec"], "/path/to/dir", "terminal"),
                ("Create a new directory named 'test'", ["terminal_exec"], "directory created", "terminal"),
                ("Count lines in a file", ["terminal_exec"], "number", "terminal"),
                ("Search for pattern 'error' in log files", ["terminal_exec"], "matching lines", "terminal"),
            ],
            "filesystem": [
                ("Read the contents of config.json", ["file_read"], "json content", "filesystem"),
                ("Write 'hello world' to output.txt", ["file_write"], "success", "filesystem"),
                ("List all Python files in src/", ["file_list"], "file list", "filesystem"),
                ("Check if requirements.txt exists", ["file_exists"], "true/false", "filesystem"),
            ],
            "reasoning": [
                ("Analyze this error and suggest a fix: 'ModuleNotFoundError: No module named X'",
                 ["terminal_exec"], "install the module", "reasoning"),
                ("Plan steps to deploy a web application", [], "deployment steps", "reasoning"),
                ("Debug why this test is failing: assertion error on line 42",
                 ["terminal_exec", "file_read"], "fix description", "reasoning"),
            ],
            "coding": [
                ("Write a function to check if a number is prime", ["file_write"], "function code", "coding"),
                ("Fix the syntax error in this Python code", ["file_read", "file_write"], "corrected code", "coding"),
                ("Implement a binary search algorithm", ["file_write"], "implementation", "coding"),
            ],
            "math": [
                ("Calculate the factorial of 5", [], "120", "math"),
                ("Find the GCD of 48 and 18", [], "6", "math"),
                ("Solve: 2x + 5 = 15", [], "x = 5", "math"),
            ],
            "data": [
                ("Parse this JSON and extract the 'name' field", ["file_read"], "extracted name", "data"),
                ("Convert CSV data to JSON format", ["file_read", "file_write"], "json output", "data"),
                ("Filter list to keep only even numbers", [], "filtered list", "data"),
            ],
        }

        templates = task_templates.get(category, task_templates["terminal"])

        # Select template based on difficulty (harder tasks later in list)
        difficulty_index = min(int(difficulty * len(templates)), len(templates) - 1)
        # Add some randomness
        index_range = max(1, len(templates) // 3)
        min_idx = max(0, difficulty_index - index_range)
        max_idx = min(len(templates) - 1, difficulty_index + index_range)
        template = random.choice(templates[min_idx:max_idx + 1])

        task_id = f"{category}_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        domain = template[3] if len(template) > 3 else category

        return CurriculumTask(
            task_id=task_id,
            description=template[0],
            difficulty=difficulty,
            category=category,
            domain=domain,
            expected_tools=template[1],
            expected_answer=template[2] if len(template) > 2 else None,
            structured_format=self.enable_structured_format,
            timeout_seconds=int(60 + (difficulty * 120)),
        )

    def _calibrate_difficulty(self, domain: str) -> float:
        """
        Calibrate difficulty based on executor success rate for a domain.

        Uses a simple proportional controller to adjust difficulty:
        - If success rate is above target, increase difficulty
        - If success rate is below target, decrease difficulty

        Args:
            domain: The task domain

        Returns:
            Calibrated difficulty value
        """
        history = self._domain_success_history.get(domain, [])

        if len(history) < 3:
            # Not enough data, use current difficulty
            return self._domain_difficulties.get(domain, 0.5)

        # Calculate recent success rate (last 10 attempts)
        recent = history[-10:]
        success_rate = sum(recent) / len(recent)

        # Proportional adjustment
        error = success_rate - self.target_success_rate
        adjustment = error * self.difficulty_learning_rate

        # Update domain difficulty
        current = self._domain_difficulties.get(domain, 0.5)
        new_difficulty = max(0.1, min(1.0, current + adjustment))
        self._domain_difficulties[domain] = new_difficulty

        self.logger.debug(
            f"Calibrated {domain} difficulty: {current:.2f} -> {new_difficulty:.2f} "
            f"(success_rate={success_rate:.2f}, target={self.target_success_rate:.2f})"
        )

        return new_difficulty

    def get_calibrated_difficulty(self, category: str) -> float:
        """
        Get calibrated difficulty for a category/domain.

        Args:
            category: Task category

        Returns:
            Difficulty value calibrated by executor performance
        """
        # Map category to domain
        domain = category if category in self.DOMAINS else "reasoning"
        return self._calibrate_difficulty(domain)

    async def generate_task(self, category: Optional[str] = None) -> CurriculumTask:
        """
        Generate a new task appropriate for current curriculum stage.

        Uses calibrated difficulty based on executor success rate per domain.

        Args:
            category: Optional task category (terminal, filesystem, reasoning, coding, math, data)

        Returns:
            A new CurriculumTask
        """
        stage = self.current_stage

        # Default category selection with expanded options
        if category is None:
            categories = ["terminal", "filesystem", "reasoning", "coding", "math", "data"]
            # Weights favor terminal initially, shift based on stage
            base_weights = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]
            if stage and stage.stage_id >= 2:
                # Later stages: more coding/reasoning
                base_weights = [0.20, 0.15, 0.20, 0.25, 0.10, 0.10]
            category = random.choices(categories, weights=base_weights)[0]

        # Get calibrated difficulty for the domain
        calibrated_diff = self.get_calibrated_difficulty(category)

        if stage is None:
            difficulty = calibrated_diff
        else:
            # Blend stage range with calibrated difficulty
            min_diff, max_diff = stage.difficulty_range
            stage_difficulty = random.uniform(min_diff, max_diff)
            # Weighted blend: 60% calibrated, 40% stage-based
            difficulty = 0.6 * calibrated_diff + 0.4 * stage_difficulty
            difficulty = max(0.1, min(1.0, difficulty + self._difficulty_adjustment))

        task = self._task_generator(difficulty, category)

        # Add to current stage
        if stage:
            stage.tasks.append(task)

        self.task_history.append(task)
        self.logger.info(
            f"Generated task {task.task_id} (difficulty: {difficulty:.2f}, "
            f"domain: {task.domain}, calibrated: {calibrated_diff:.2f})"
        )

        return task

    async def process_result(
        self,
        task: CurriculumTask,
        reward: RewardSignal,
        output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process the result of a task attempt.

        Tracks domain success history for difficulty calibration and
        validates structured output format compliance.

        Args:
            task: The completed task
            reward: Reward signal from task execution
            output: Optional executor output for format validation

        Returns:
            Feedback dictionary with next steps
        """
        success = reward.value >= 0.5 if not reward.binary else reward.value == 1.0

        # Validate structured format if output provided
        format_valid = True
        format_data = {}
        if output and self.enable_structured_format:
            format_valid, format_data = validate_structured_output(output)
            # Update format compliance tracking
            self._format_compliance_rate = (
                self._format_compliance_rate * 0.9 +
                (1.0 if format_valid else 0.0) * 0.1
            )

        # Analyze workplace feedback: tool usage and errors
        workplace_feedback = self._analyze_workplace_feedback(output or "", success)

        # Record attempt with format tracking
        task.record_attempt(success, format_valid)

        # Update domain success history for calibration
        domain = task.domain if task.domain in self.DOMAINS else task.category
        if domain in self._domain_success_history:
            self._domain_success_history[domain].append(success)
            # Keep history bounded
            if len(self._domain_success_history[domain]) > 100:
                self._domain_success_history[domain] = self._domain_success_history[domain][-100:]

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
            "domain": domain,
            "calibrated_difficulty": self._domain_difficulties.get(domain, 0.5),
            "format_valid": format_valid,
            "format_compliance_rate": self._format_compliance_rate,
        }

        # Add extracted format data if available
        if format_data:
            feedback["format_data"] = {
                "has_question_tags": format_data.get("has_question_tags", False),
                "has_boxed_answer": format_data.get("has_boxed_answer", False),
                "has_thinking": format_data.get("has_thinking", False),
            }

        # Generate hints if struggling
        if not success and task.attempts >= 2:
            feedback["hints"] = self._generate_hints(task)

        # Incorporate workplace feedback into feedback
        feedback.update(workplace_feedback)

        return feedback

    def _analyze_workplace_feedback(self, output: str, success: bool) -> Dict[str, Any]:
        """
        Analyze output for workplace feedback: tool usage, errors, and patterns.

        Args:
            output: Executor output string
            success: Whether the task was successful

        Returns:
            Dictionary with workplace feedback metrics
        """
        feedback = {
            "tools_used": [],
            "errors_detected": [],
            "tool_sequence": [],
        }

        if not output:
            return feedback

        # Extract tool calls (simple pattern matching)
        tool_patterns = [
            r"tool_call\s*:\s*(\w+)",
            r"using\s+(\w+)\s+tool",
            r"execute\s+(\w+)",
            r"run\s+(\w+)",
            r"(\w+)\s*command",
        ]

        tools_used = set()
        for pattern in tool_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            tools_used.update(matches)

        feedback["tools_used"] = list(tools_used)
        feedback["tool_sequence"] = list(tools_used)  # Simple sequence

        # Detect common errors
        error_patterns = [
            r"error\s*:",
            r"failed",
            r"permission denied",
            r"command not found",
            r"no such file",
            r"syntax error",
        ]

        errors_found = []
        for pattern in error_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                errors_found.append(pattern)

        feedback["errors_detected"] = errors_found

        # Update tracking
        if success and tools_used:
            self._successful_tool_sequences.append(list(tools_used))
            # Keep last 50 sequences
            if len(self._successful_tool_sequences) > 50:
                self._successful_tool_sequences = self._successful_tool_sequences[-50:]

        # Update tool success rates
        for tool in tools_used:
            current_rate = self._tool_success_rates.get(tool, 0.5)
            self._tool_success_rates[tool] = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1

        # Update error patterns
        for error in errors_found:
            self._error_patterns[error] = self._error_patterns.get(error, 0) + 1

        # Add workplace insights to feedback
        feedback["tool_success_rates"] = self._tool_success_rates.copy()
        feedback["common_errors"] = sorted(self._error_patterns.items(), key=lambda x: x[1], reverse=True)[:3]

        return feedback

    def _generate_hints(self, task: CurriculumTask) -> List[str]:
        """Generate hints for a struggling student, incorporating workplace feedback."""
        hints = []

        if task.expected_tools:
            hints.append(f"Consider using: {', '.join(task.expected_tools)}")

        if task.difficulty > 0.5:
            hints.append("Break the problem into smaller steps")

        hints.append("Review the error messages carefully")

        # Workplace feedback hints
        if self._tool_success_rates:
            # Suggest highly successful tools
            best_tools = sorted(self._tool_success_rates.items(), key=lambda x: x[1], reverse=True)[:2]
            if best_tools:
                tool_names = [tool for tool, _ in best_tools if _ > 0.7]
                if tool_names:
                    hints.append(f"Try using these successful tools: {', '.join(tool_names)}")

        if self._error_patterns:
            # Common error hints
            common_errors = sorted(self._error_patterns.items(), key=lambda x: x[1], reverse=True)[:2]
            for error_pattern, count in common_errors:
                if "permission" in error_pattern.lower():
                    hints.append("Check file permissions and user privileges")
                elif "command not found" in error_pattern.lower():
                    hints.append("Ensure the command is installed and in PATH")
                elif "no such file" in error_pattern.lower():
                    hints.append("Verify file paths and directory existence")
                elif "syntax" in error_pattern.lower():
                    hints.append("Check command syntax and arguments")

        if self._successful_tool_sequences:
            # Suggest successful sequences
            recent_sequences = self._successful_tool_sequences[-3:]
            if recent_sequences:
                sequence_str = " -> ".join(recent_sequences[0])
                hints.append(f"Consider this successful tool sequence: {sequence_str}")

        return hints

    async def get_curriculum_status(self) -> Dict[str, Any]:
        """Get overall curriculum status including calibration data."""
        total_tasks = len(self.task_history)
        total_successes = sum(1 for t in self.task_history if t.success_rate >= 0.5)
        total_format_compliant = sum(1 for t in self.task_history if t.format_compliance_rate >= 0.5)

        # Calculate domain-specific statistics
        domain_stats = {}
        for domain in self.DOMAINS:
            history = self._domain_success_history.get(domain, [])
            if history:
                domain_stats[domain] = {
                    "attempts": len(history),
                    "successes": sum(history),
                    "success_rate": sum(history) / len(history),
                    "calibrated_difficulty": self._domain_difficulties.get(domain, 0.5),
                }
            else:
                domain_stats[domain] = {
                    "attempts": 0,
                    "successes": 0,
                    "success_rate": 0.0,
                    "calibrated_difficulty": self._domain_difficulties.get(domain, 0.5),
                }

        return {
            "current_stage": self.current_stage_idx,
            "stage_name": self.current_stage.name if self.current_stage else "Complete",
            "total_stages": len(self.stages),
            "total_tasks_generated": total_tasks,
            "overall_success_rate": total_successes / total_tasks if total_tasks > 0 else 0,
            "category_performance": self.student_performance,
            "difficulty_adjustment": self._difficulty_adjustment,
            # Enhanced calibration data
            "target_success_rate": self.target_success_rate,
            "domain_calibration": domain_stats,
            "format_compliance_rate": self._format_compliance_rate,
            "format_compliant_tasks": total_format_compliant,
            "structured_format_enabled": self.enable_structured_format,
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
