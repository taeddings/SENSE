"""
SENSE-v2 GRPO Trainer
Step-wise Group Relative Policy Optimization for Agent 0 training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import asyncio
import random
import numpy as np
from datetime import datetime

from sense_v2.core.config import EvolutionConfig
from sense_v2.core.schemas import RewardSignal
from sense_v2.agents.agent_0.curriculum import CurriculumAgent, CurriculumTask
from sense_v2.agents.agent_0.executor import ExecutorAgent, ExecutionTrace


@dataclass
class GRPOSample:
    """A single sample in a GRPO training group."""
    executor_id: int
    task: CurriculumTask
    trace: ExecutionTrace
    reward: float
    advantage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executor_id": self.executor_id,
            "task_id": self.task.task_id,
            "reward": self.reward,
            "advantage": self.advantage,
            "success": self.trace.success,
        }


@dataclass
class GRPOGroup:
    """A group of samples for GRPO comparison."""
    group_id: str
    samples: List[GRPOSample] = field(default_factory=list)
    mean_reward: float = 0.0
    std_reward: float = 0.0

    def compute_advantages(self) -> None:
        """Compute relative advantages within the group."""
        if not self.samples:
            return

        rewards = [s.reward for s in self.samples]
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards) + 1e-8  # Avoid division by zero

        for sample in self.samples:
            sample.advantage = (sample.reward - self.mean_reward) / self.std_reward


@dataclass
class TrainingMetrics:
    """Metrics from a training iteration."""
    iteration: int
    groups_processed: int
    mean_reward: float
    mean_advantage: float
    best_executor_id: int
    worst_executor_id: int
    kl_divergence: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "groups_processed": self.groups_processed,
            "mean_reward": self.mean_reward,
            "mean_advantage": self.mean_advantage,
            "best_executor_id": self.best_executor_id,
            "worst_executor_id": self.worst_executor_id,
            "kl_divergence": self.kl_divergence,
            "timestamp": self.timestamp.isoformat(),
        }


class GRPOTrainer:
    """
    Step-wise Group Relative Policy Optimization Trainer.

    Implements the training loop for Agent 0 where:
    - CurriculumAgent generates tasks
    - Multiple ExecutorAgents attempt tasks
    - Rewards are compared within groups
    - Executors are updated based on relative performance

    Per SYSTEM_PROMPT requirements:
    - Step-wise GRPO for co-evolutionary learning
    - Tool-verified success determines rewards
    """

    def __init__(
        self,
        curriculum_agent: CurriculumAgent,
        config: Optional[EvolutionConfig] = None,
    ):
        self.curriculum = curriculum_agent
        self.config = config or EvolutionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Population of executors
        self.executors: List[ExecutorAgent] = []
        self._initialize_population()

        # Training state
        self.current_iteration = 0
        self.training_history: List[TrainingMetrics] = []
        self.best_fitness_ever = 0.0

        # GRPO parameters
        self.temperature = self.config.grpo_temperature
        self.kl_coeff = self.config.grpo_kl_coeff

    def _initialize_population(self) -> None:
        """Initialize population of executor agents."""
        for i in range(self.config.population_size):
            executor = ExecutorAgent(config=self.config)
            executor.state.generation = 0
            self.executors.append(executor)

        self.logger.info(f"Initialized population of {len(self.executors)} executors")

    async def train_step(self) -> TrainingMetrics:
        """
        Execute one training step:
        1. Generate task from curriculum
        2. Run all executors on the task
        3. Compute group advantages
        4. Update executor weights
        """
        self.current_iteration += 1

        # Generate groups of tasks
        groups: List[GRPOGroup] = []

        for g in range(self.config.grpo_group_size):
            task = await self.curriculum.generate_task()
            group = GRPOGroup(group_id=f"iter{self.current_iteration}_group{g}")

            # Run each executor on the task
            for idx, executor in enumerate(self.executors):
                reward, trace = await executor.execute_task(task)

                sample = GRPOSample(
                    executor_id=idx,
                    task=task,
                    trace=trace,
                    reward=reward.value,
                )
                group.samples.append(sample)

            # Compute advantages within group
            group.compute_advantages()
            groups.append(group)

            # Update curriculum based on aggregate results
            avg_reward = RewardSignal(
                value=group.mean_reward,
                binary=False,
                source="grpo_group",
            )
            await self.curriculum.process_result(task, avg_reward)

        # Aggregate metrics
        all_rewards = [s.reward for g in groups for s in g.samples]
        all_advantages = [s.advantage for g in groups for s in g.samples]

        executor_rewards = {i: [] for i in range(len(self.executors))}
        for group in groups:
            for sample in group.samples:
                executor_rewards[sample.executor_id].append(sample.reward)

        avg_executor_rewards = {
            i: np.mean(rewards) for i, rewards in executor_rewards.items()
        }

        best_id = max(avg_executor_rewards, key=avg_executor_rewards.get)
        worst_id = min(avg_executor_rewards, key=avg_executor_rewards.get)

        # Selection and evolution
        await self._evolve_population(groups)

        # Compute KL divergence estimate
        kl_div = self._estimate_kl_divergence(groups)

        metrics = TrainingMetrics(
            iteration=self.current_iteration,
            groups_processed=len(groups),
            mean_reward=np.mean(all_rewards),
            mean_advantage=np.mean(all_advantages),
            best_executor_id=best_id,
            worst_executor_id=worst_id,
            kl_divergence=kl_div,
        )

        self.training_history.append(metrics)

        # Track best fitness
        if metrics.mean_reward > self.best_fitness_ever:
            self.best_fitness_ever = metrics.mean_reward

        self.logger.info(
            f"Iteration {self.current_iteration}: "
            f"mean_reward={metrics.mean_reward:.4f}, "
            f"best_executor={best_id}"
        )

        return metrics

    async def _evolve_population(self, groups: List[GRPOGroup]) -> None:
        """
        Evolve the population based on GRPO advantages.
        Uses selection, crossover, and mutation.
        """
        # Compute fitness for each executor
        executor_fitness = {i: 0.0 for i in range(len(self.executors))}
        executor_counts = {i: 0 for i in range(len(self.executors))}

        for group in groups:
            for sample in group.samples:
                executor_fitness[sample.executor_id] += sample.advantage
                executor_counts[sample.executor_id] += 1

        # Normalize by count
        for i in executor_fitness:
            if executor_counts[i] > 0:
                executor_fitness[i] /= executor_counts[i]

        # Selection: keep top-k
        sorted_ids = sorted(executor_fitness.keys(),
                          key=lambda x: executor_fitness[x],
                          reverse=True)

        elite_ids = sorted_ids[:self.config.selection_top_k]

        # Update generations for elites
        for idx in elite_ids:
            self.executors[idx].state.generation += 1

        # Replace worst performers with variations of best
        worst_ids = sorted_ids[-(len(sorted_ids) - self.config.selection_top_k):]

        for worst_idx in worst_ids:
            # Select parent from elites
            parent_idx = random.choice(elite_ids)
            parent = self.executors[parent_idx]

            # Create child with mutations
            child = self.executors[worst_idx]
            child.state.generation = parent.state.generation

            # Crossover strategy weights
            if random.random() < self.config.crossover_rate:
                for key in child.state.strategy_weights:
                    if key in parent.state.strategy_weights:
                        child.state.strategy_weights[key] = (
                            parent.state.strategy_weights[key] * 0.7 +
                            child.state.strategy_weights[key] * 0.3
                        )

            # Mutation
            if random.random() < self.config.mutation_rate:
                key = random.choice(list(child.state.strategy_weights.keys()))
                child.state.strategy_weights[key] += random.gauss(0, 0.1)

                # Normalize
                total = sum(child.state.strategy_weights.values())
                child.state.strategy_weights = {
                    k: max(0.01, v / total)
                    for k, v in child.state.strategy_weights.items()
                }

    def _estimate_kl_divergence(self, groups: List[GRPOGroup]) -> float:
        """Estimate KL divergence for monitoring training stability."""
        if not groups or not groups[0].samples:
            return 0.0

        # Simple estimate based on advantage variance
        all_advantages = [s.advantage for g in groups for s in g.samples]
        return float(np.var(all_advantages))

    async def train(self, num_iterations: int) -> List[TrainingMetrics]:
        """
        Run multiple training iterations.

        Args:
            num_iterations: Number of training steps

        Returns:
            List of training metrics
        """
        self.logger.info(f"Starting training for {num_iterations} iterations")

        metrics = []
        for i in range(num_iterations):
            step_metrics = await self.train_step()
            metrics.append(step_metrics)

            # Check for early stopping
            if self.current_iteration >= self.config.max_curriculum_steps:
                self.logger.info("Reached max curriculum steps, stopping")
                break

        return metrics

    def get_best_executor(self) -> ExecutorAgent:
        """Get the best performing executor."""
        return max(self.executors, key=lambda e: e.state.current_fitness)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history:
            return {"status": "not_started"}

        recent = self.training_history[-10:]

        return {
            "total_iterations": self.current_iteration,
            "best_fitness_ever": self.best_fitness_ever,
            "recent_mean_reward": np.mean([m.mean_reward for m in recent]),
            "recent_mean_advantage": np.mean([m.mean_advantage for m in recent]),
            "population_size": len(self.executors),
            "curriculum_status": asyncio.get_event_loop().run_until_complete(
                self.curriculum.get_curriculum_status()
            ) if asyncio.get_event_loop().is_running() else {},
            "executor_stats": [e.get_execution_stats() for e in self.executors[:3]],
        }
