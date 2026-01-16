"""
SENSE-v2 Development Log
State tracking for evolutionary progress and system health.
Per SYSTEM_PROMPT: Maintains dev_log.json for tracking progress.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging
import threading


@dataclass
class EvolutionState:
    """State of the evolutionary system."""
    generation: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    population_size: int = 0
    curriculum_stage: int = 0
    total_tasks_completed: int = 0
    success_rate: float = 0.0


@dataclass
class AgentState:
    """State of a single agent."""
    name: str
    status: str = "idle"
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_fitness: float = 0.0
    last_activity: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """Overall system state."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "2.0.0"
    status: str = "running"
    uptime_seconds: int = 0
    evolution: EvolutionState = field(default_factory=EvolutionState)
    agents: Dict[str, AgentState] = field(default_factory=dict)
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class DevLog:
    """
    Development log for SENSE-v2.

    Persists system state to dev_log.json for:
    - Tracking evolutionary progress
    - Monitoring system health
    - Debugging and analysis

    Per SYSTEM_PROMPT requirements.
    """

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path or "dev_log.json")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Current state
        self._state = SystemState()
        self._start_time = datetime.now()

        # Thread safety
        self._lock = threading.Lock()

        # Load existing state if available
        self._load_state()

    def _load_state(self) -> None:
        """Load existing state from disk."""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    data = json.load(f)

                # Restore evolution state
                if 'evolution' in data:
                    self._state.evolution = EvolutionState(**data['evolution'])

                # Restore agents
                if 'agents' in data:
                    for name, agent_data in data['agents'].items():
                        self._state.agents[name] = AgentState(**agent_data)

                # Restore metrics
                if 'metrics' in data:
                    self._state.metrics = data['metrics']

                self.logger.info(f"Loaded state from {self.log_path}")

            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")

    def save(self) -> bool:
        """Save current state to disk."""
        with self._lock:
            try:
                # Update timestamp and uptime
                self._state.timestamp = datetime.now().isoformat()
                self._state.uptime_seconds = int(
                    (datetime.now() - self._start_time).total_seconds()
                )

                # Convert to dict
                data = self._state_to_dict()

                # Ensure directory exists
                self.log_path.parent.mkdir(parents=True, exist_ok=True)

                # Write atomically
                temp_path = self.log_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

                temp_path.replace(self.log_path)

                return True

            except Exception as e:
                self.logger.error(f"Failed to save state: {e}")
                return False

    def _state_to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "timestamp": self._state.timestamp,
            "version": self._state.version,
            "status": self._state.status,
            "uptime_seconds": self._state.uptime_seconds,
            "evolution": asdict(self._state.evolution),
            "agents": {
                name: asdict(agent)
                for name, agent in self._state.agents.items()
            },
            "memory_usage": self._state.memory_usage,
            "errors": self._state.errors[-100:],  # Keep last 100 errors
            "metrics": self._state.metrics,
        }

    def update_evolution(
        self,
        generation: Optional[int] = None,
        best_fitness: Optional[float] = None,
        average_fitness: Optional[float] = None,
        curriculum_stage: Optional[int] = None,
        **kwargs
    ) -> None:
        """Update evolutionary state."""
        with self._lock:
            if generation is not None:
                self._state.evolution.generation = generation
            if best_fitness is not None:
                self._state.evolution.best_fitness = best_fitness
            if average_fitness is not None:
                self._state.evolution.average_fitness = average_fitness
            if curriculum_stage is not None:
                self._state.evolution.curriculum_stage = curriculum_stage

            for key, value in kwargs.items():
                if hasattr(self._state.evolution, key):
                    setattr(self._state.evolution, key, value)

    def update_agent(
        self,
        name: str,
        status: Optional[str] = None,
        tasks_completed: Optional[int] = None,
        tasks_failed: Optional[int] = None,
        current_fitness: Optional[float] = None,
        **kwargs
    ) -> None:
        """Update agent state."""
        with self._lock:
            if name not in self._state.agents:
                self._state.agents[name] = AgentState(name=name)

            agent = self._state.agents[name]
            agent.last_activity = datetime.now().isoformat()

            if status is not None:
                agent.status = status
            if tasks_completed is not None:
                agent.tasks_completed = tasks_completed
            if tasks_failed is not None:
                agent.tasks_failed = tasks_failed
            if current_fitness is not None:
                agent.current_fitness = current_fitness

            agent.metadata.update(kwargs)

    def update_memory(self, memory_stats: Dict[str, Any]) -> None:
        """Update memory usage stats."""
        with self._lock:
            self._state.memory_usage = memory_stats

    def record_error(self, error: str, context: Optional[Dict] = None) -> None:
        """Record an error."""
        with self._lock:
            self._state.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": error,
                "context": context or {},
            })

    def update_metric(self, name: str, value: Any) -> None:
        """Update a custom metric."""
        with self._lock:
            self._state.metrics[name] = {
                "value": value,
                "updated_at": datetime.now().isoformat(),
            }

    def set_status(self, status: str) -> None:
        """Set system status."""
        with self._lock:
            self._state.status = status

    def get_state(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        with self._lock:
            return self._state_to_dict()

    def get_summary(self) -> Dict[str, Any]:
        """Get a brief summary of current state."""
        with self._lock:
            return {
                "status": self._state.status,
                "uptime_seconds": int(
                    (datetime.now() - self._start_time).total_seconds()
                ),
                "generation": self._state.evolution.generation,
                "best_fitness": self._state.evolution.best_fitness,
                "agents_active": sum(
                    1 for a in self._state.agents.values()
                    if a.status == "active"
                ),
                "recent_errors": len(self._state.errors[-10:]),
            }


class StateLogger:
    """
    Convenience class for logging state changes.
    Wraps DevLog with automatic saving.
    """

    def __init__(self, dev_log: DevLog, auto_save_interval: int = 60):
        self.dev_log = dev_log
        self.auto_save_interval = auto_save_interval
        self._last_save = datetime.now()
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_evolution_step(
        self,
        generation: int,
        best_fitness: float,
        average_fitness: float,
        **kwargs
    ) -> None:
        """Log an evolution step."""
        self.dev_log.update_evolution(
            generation=generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            **kwargs
        )
        self._maybe_save()

    def log_task_completion(
        self,
        agent_name: str,
        success: bool,
        fitness: Optional[float] = None,
    ) -> None:
        """Log task completion for an agent."""
        # Get current counts
        state = self.dev_log.get_state()
        agent_state = state.get('agents', {}).get(agent_name, {})

        completed = agent_state.get('tasks_completed', 0)
        failed = agent_state.get('tasks_failed', 0)

        if success:
            completed += 1
        else:
            failed += 1

        self.dev_log.update_agent(
            name=agent_name,
            status="active",
            tasks_completed=completed,
            tasks_failed=failed,
            current_fitness=fitness,
        )
        self._maybe_save()

    def log_error(self, error: str, agent_name: Optional[str] = None) -> None:
        """Log an error."""
        self.dev_log.record_error(
            error=error,
            context={"agent": agent_name} if agent_name else None,
        )
        self._maybe_save()

    def _maybe_save(self) -> None:
        """Save if auto-save interval has passed."""
        now = datetime.now()
        if (now - self._last_save).total_seconds() >= self.auto_save_interval:
            self.dev_log.save()
            self._last_save = now
