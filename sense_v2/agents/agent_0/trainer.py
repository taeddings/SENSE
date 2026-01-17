"""
SENSE-v2 GRPO Trainer
Step-wise Group Relative Policy Optimization for Agent 0 training.

Enhanced with multi-faceted reward system from Agent0 research:
- Format reward (structured output validation)
- Tool usage reward (encourage appropriate tool calls)
- Diversity penalty (BLEU-based clustering to prevent repetitive solutions)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import asyncio
import random
import re
import numpy as np
from datetime import datetime

from sense_v2.core.config import EvolutionConfig, MemoryAwareConfig
from sense_v2.core.schemas import RewardSignal
from sense_v2.agents.agent_0.curriculum import CurriculumAgent, CurriculumTask
from sense_v2.agents.agent_0.executor import ExecutorAgent, ExecutionTrace
from sense_v2.utils.dev_log import DevLog, StateLogger

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional imports for enhanced rewards
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Enhanced Reward Functions (from Agent0 research)
# =============================================================================

def format_reward(output: str, expected_format: Optional[str] = None) -> float:
    """
    Validate structured output format.

    Checks for proper structure like:
    - <think>...</think> blocks for reasoning
    - \\boxed{...} for answers
    - Tool call formatting

    Args:
        output: The model's output string
        expected_format: Optional regex pattern for expected format

    Returns:
        1.0 if format is valid, 0.0 otherwise
    """
    if not output:
        return 0.0

    # Default format: check for thinking and answer structure
    if expected_format:
        pattern = re.compile(expected_format, re.DOTALL)
        return 1.0 if re.fullmatch(pattern, output) else 0.0

    # Check for common structured patterns
    has_reasoning = bool(re.search(r"<think>.*</think>", output, re.DOTALL))
    has_answer = bool(re.search(r"\\boxed\{.*\}", output, re.DOTALL))
    has_tool_call = bool(re.search(r"```\w+.*```", output, re.DOTALL))

    # Score based on structure presence
    if has_reasoning and has_answer:
        return 1.0
    elif has_reasoning or has_answer or has_tool_call:
        return 0.5
    return 0.0


def tool_usage_reward(output: str, weight: float = 0.05, cap: int = 4) -> float:
    """
    Reward appropriate tool usage.

    Encourages agents to use tools when appropriate, but caps the reward
    to prevent excessive tool calls.

    Args:
        output: The model's output string
        weight: Reward per tool call
        cap: Maximum number of tool calls to reward

    Returns:
        Reward value (capped at weight * cap)
    """
    if not output:
        return 0.0

    # Count tool call patterns
    tool_patterns = [
        r"```output",           # Code execution output blocks
        r"tool_call\s*:",       # Explicit tool calls
        r"<tool>.*?</tool>",    # XML-style tool calls
        r"\[TOOL:.*?\]",        # Bracket-style tool calls
    ]

    total_calls = 0
    for pattern in tool_patterns:
        total_calls += len(re.findall(pattern, output, re.IGNORECASE))

    capped_calls = min(total_calls, cap)
    return capped_calls * weight


def _bleu_distance_matrix(sentences: List[str]) -> np.ndarray:
    """
    Compute BLEU-based distance matrix for clustering.

    Args:
        sentences: List of sentences to compare

    Returns:
        Distance matrix (1 - BLEU score)
    """
    if not BLEU_AVAILABLE:
        # Fallback: simple character-level similarity
        n = len(sentences)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    dist[i, j] = 0.0
                else:
                    # Jaccard similarity on character trigrams
                    s1_trigrams = set(sentences[i][k:k+3] for k in range(len(sentences[i])-2))
                    s2_trigrams = set(sentences[j][k:k+3] for k in range(len(sentences[j])-2))
                    if s1_trigrams or s2_trigrams:
                        jaccard = len(s1_trigrams & s2_trigrams) / len(s1_trigrams | s2_trigrams)
                        dist[i, j] = dist[j, i] = 1 - jaccard
                    else:
                        dist[i, j] = dist[j, i] = 1.0
        return dist

    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1

    for i in range(n):
        for j in range(i, n):
            if i == j:
                dist[i, j] = 0.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                if hyp and ref[0]:
                    score = sentence_bleu(ref, hyp, smoothing_function=smoother)
                else:
                    score = 0.0
                dist[i, j] = dist[j, i] = 1 - score
    return dist


def diversity_penalty(
    outputs: List[str],
    distance_threshold: float = 0.5,
    linkage: str = "average"
) -> List[float]:
    """
    Calculate diversity penalty using BLEU-based clustering.

    Penalizes outputs that are too similar to others in the batch,
    encouraging diverse solutions.

    Args:
        outputs: List of output strings to compare
        distance_threshold: Clustering distance threshold
        linkage: Clustering linkage method

    Returns:
        List of penalty values (higher = more penalty for repetitive solutions)
    """
    if not outputs:
        return []

    if len(outputs) == 1:
        return [0.0]

    # Compute distance matrix
    dist_mat = _bleu_distance_matrix(outputs)

    if not SKLEARN_AVAILABLE:
        # Fallback: simple average distance penalty
        penalties = []
        for i in range(len(outputs)):
            avg_dist = np.mean([dist_mat[i, j] for j in range(len(outputs)) if i != j])
            # Lower distance = higher penalty (more similar to others)
            penalties.append(1.0 - avg_dist)
        return penalties

    # Cluster using agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)

    # Calculate cluster proportions as penalty
    from collections import Counter
    total = len(outputs)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    # Penalty is the proportion of cluster (larger cluster = higher penalty)
    penalties = [cluster_ratio[lab] for lab in labels]
    return penalties


# =============================================================================
# Memory-Aware Fitness Functions
# =============================================================================

def get_memory_usage() -> dict:
    """
    Get current system memory usage.

    Returns:
        Dictionary with memory statistics
    """
    if not PSUTIL_AVAILABLE:
        return {
            "ram_percent": 0.0,
            "ram_available_mb": 0,
            "ram_used_mb": 0,
            "psutil_available": False,
        }

    mem = psutil.virtual_memory()
    return {
        "ram_percent": mem.percent / 100.0,  # Normalize to 0-1
        "ram_available_mb": mem.available // (1024 * 1024),
        "ram_used_mb": mem.used // (1024 * 1024),
        "ram_total_mb": mem.total // (1024 * 1024),
        "psutil_available": True,
    }


def compute_memory_aware_fitness(
    accuracy: float,
    memory_mb: float,
    drift_resistance: float = 1.0,
    accuracy_weight: float = 0.6,
    memory_weight: float = 0.2,
    drift_weight: float = 0.2,
    memory_baseline_mb: float = 1000.0,
) -> float:
    """
    Multi-objective fitness function with memory penalty.

    Balances accuracy, memory efficiency, and drift resistance for
    evolutionary selection of memory-efficient agents.

    Args:
        accuracy: Task completion accuracy (0.0 to 1.0)
        memory_mb: Memory used during execution in MB
        drift_resistance: Resistance to distribution drift (0.0 to 1.0)
        accuracy_weight: Weight for accuracy component
        memory_weight: Weight for memory efficiency component
        drift_weight: Weight for drift resistance component
        memory_baseline_mb: Baseline memory for normalization

    Returns:
        Combined fitness score
    """
    # Normalize memory penalty: higher memory = lower score
    # Using inverse with baseline to keep values reasonable
    memory_efficiency = memory_baseline_mb / max(memory_mb, 1.0)
    memory_efficiency = min(memory_efficiency, 1.0)  # Cap at 1.0

    # Combine components
    fitness = (
        accuracy * accuracy_weight +
        memory_efficiency * memory_weight +
        drift_resistance * drift_weight
    )

    return fitness


@dataclass
class MemoryAwareRewardComponents:
    """Extended reward components including memory metrics."""
    base_reward: float
    format_reward: float
    tool_reward: float
    diversity_penalty: float
    memory_penalty: float
    drift_resistance: float
    total: float

    def to_dict(self) -> dict:
        return {
            "base_reward": self.base_reward,
            "format_reward": self.format_reward,
            "tool_reward": self.tool_reward,
            "diversity_penalty": self.diversity_penalty,
            "memory_penalty": self.memory_penalty,
            "drift_resistance": self.drift_resistance,
            "total": self.total,
        }


@dataclass
class RewardComponents:
    """Breakdown of reward components for analysis."""
    base_reward: float
    format_reward: float
    tool_reward: float
    diversity_penalty: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "base_reward": self.base_reward,
            "format_reward": self.format_reward,
            "tool_reward": self.tool_reward,
            "diversity_penalty": self.diversity_penalty,
            "total": self.total,
        }


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

    Enhanced with Agent0 reward components:
    - Format reward (structured output validation)
    - Tool usage reward (encourages appropriate tool calls)
    - Diversity penalty (BLEU-based clustering)
    """

    def __init__(
        self,
        curriculum_agent: CurriculumAgent,
        config: Optional[EvolutionConfig] = None,
        format_reward_weight: float = 0.1,
        tool_reward_weight: float = 0.05,
        tool_reward_cap: int = 4,
        diversity_penalty_weight: float = 0.1,
        enable_enhanced_rewards: bool = True,
        dev_log: Optional[DevLog] = None,
    ):
        self.curriculum = curriculum_agent
        self.config = config or EvolutionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize DevLog and StateLogger
        self.dev_log = dev_log or DevLog()
        self.state_logger = StateLogger(self.dev_log)

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

        # Enhanced reward parameters (from Agent0)
        self.format_reward_weight = format_reward_weight
        self.tool_reward_weight = tool_reward_weight
        self.tool_reward_cap = tool_reward_cap
        self.diversity_penalty_weight = diversity_penalty_weight
        self.enable_enhanced_rewards = enable_enhanced_rewards

        # Reward component tracking
        self.reward_component_history: List[Dict[str, Any]] = []

    def _initialize_population(self) -> None:
        """Initialize population of executor agents."""
        for i in range(self.config.population_size):
            executor = ExecutorAgent(config=self.config)
            executor.state.generation = 0
            self.executors.append(executor)

        self.logger.info(f"Initialized population of {len(self.executors)} executors")

    def compute_enhanced_reward(
        self,
        base_reward: float,
        output: str,
        all_outputs: Optional[List[str]] = None,
        output_index: int = 0,
    ) -> RewardComponents:
        """
        Compute enhanced reward with multiple components.

        Args:
            base_reward: The base task completion reward
            output: The executor's output string
            all_outputs: All outputs in the batch (for diversity penalty)
            output_index: Index of this output in the batch

        Returns:
            RewardComponents with breakdown of all components
        """
        if not self.enable_enhanced_rewards:
            return RewardComponents(
                base_reward=base_reward,
                format_reward=0.0,
                tool_reward=0.0,
                diversity_penalty=0.0,
                total=base_reward,
            )

        # Compute format reward
        fmt_reward = format_reward(output) * self.format_reward_weight

        # Compute tool usage reward
        tool_rew = tool_usage_reward(
            output,
            weight=self.tool_reward_weight,
            cap=self.tool_reward_cap
        )

        # Compute diversity penalty
        div_penalty = 0.0
        if all_outputs and len(all_outputs) > 1:
            penalties = diversity_penalty(all_outputs)
            if output_index < len(penalties):
                div_penalty = penalties[output_index] * self.diversity_penalty_weight

        # Compute total reward
        total = base_reward + fmt_reward + tool_rew - div_penalty

        return RewardComponents(
            base_reward=base_reward,
            format_reward=fmt_reward,
            tool_reward=tool_rew,
            diversity_penalty=div_penalty,
            total=total,
        )

    async def train_step(self) -> TrainingMetrics:
        """
        Execute one training step:
        1. Generate task from curriculum
        2. Run all executors on the task
        3. Compute enhanced rewards with format, tool, and diversity components
        4. Compute group advantages
        5. Update executor weights
        """
        try:
            self.current_iteration += 1

            # Generate groups of tasks
            groups: List[GRPOGroup] = []
            iteration_reward_components: List[Dict[str, Any]] = []

            # Log current generation and curriculum stage
            curriculum_status = await self.curriculum.get_curriculum_status()
            self.state_logger.log_evolution_step(
                generation=self.current_iteration,
                best_fitness=self.best_fitness_ever,
                average_fitness=0.0, # Will be updated later
                curriculum_stage=curriculum_status.get("current_stage", 0),
                total_tasks_completed=len(self.curriculum.task_history),
                success_rate=curriculum_status.get("overall_success_rate", 0.0),
            )

            for g in range(self.config.grpo_group_size):
                task = await self.curriculum.generate_task()
                group = GRPOGroup(group_id=f"iter{self.current_iteration}_group{g}")

                # First pass: collect all outputs for diversity calculation
                executor_results: List[Tuple[int, RewardSignal, ExecutionTrace]] = []
                all_outputs: List[str] = []

                for idx, executor in enumerate(self.executors):
                    base_reward, trace = await executor.execute_task(task)
                    executor_results.append((idx, base_reward, trace))
                    # Collect output for diversity penalty calculation
                    output_str = trace.output if hasattr(trace, 'output') else str(trace)
                    all_outputs.append(output_str)

                # Second pass: compute enhanced rewards with diversity
                for result_idx, (executor_idx, base_reward, trace) in enumerate(executor_results):
                    output_str = all_outputs[result_idx]

                    # Compute enhanced reward with all components
                    reward_components = self.compute_enhanced_reward(
                        base_reward=base_reward.value,
                        output=output_str,
                        all_outputs=all_outputs,
                        output_index=result_idx,
                    )

                    # Track reward components for analysis
                    iteration_reward_components.append({
                        "iteration": self.current_iteration,
                        "group": g,
                        "executor_id": executor_idx,
                        "components": reward_components.to_dict(),
                    })

                    sample = GRPOSample(
                        executor_id=executor_idx,
                        task=task,
                        trace=trace,
                        reward=reward_components.total,  # Use enhanced total reward
                    )
                    group.samples.append(sample)

                    # Log task completion for each executor
                    self.state_logger.log_task_completion(
                        agent_name=f"executor_{executor_idx}",
                        success=trace.success,
                        fitness=reward_components.total,
                    )

                # Compute advantages within group
                group.compute_advantages()
                groups.append(group)

                # Update curriculum based on aggregate results
                avg_reward = RewardSignal(
                    value=group.mean_reward,
                    binary=False,
                    source="grpo_group",
                )
                await self.curriculum.process_result(task, avg_reward, output=all_outputs[0]) # Assuming first output is representative

            # Store reward component history for analysis
            self.reward_component_history.extend(iteration_reward_components)

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

            # Update overall average fitness in DevLog
            self.state_logger.log_evolution_step(
                generation=self.current_iteration,
                average_fitness=metrics.mean_reward,
                best_fitness=self.best_fitness_ever,
            )

            self.dev_log.save() # Save state after each training step

            return metrics
        except Exception as e:
            self.logger.error(f"Error during training step: {e}", exc_info=True)
            self.state_logger.log_error(
                error=f"Training step failed: {e}",
                context={"iteration": self.current_iteration}
            )
            raise

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

        self.dev_log.save() # Save state after training completes

        return metrics

    def get_best_executor(self) -> ExecutorAgent:
        """Get the best performing executor."""
        return max(self.executors, key=lambda e: e.state.current_fitness)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history:
            return {"status": "not_started"}

        recent = self.training_history[-10:]

        summary = {
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

        # Add enhanced reward statistics
        if self.enable_enhanced_rewards:
            summary["reward_components"] = self.get_reward_component_stats()

        return summary

    def get_reward_component_stats(self) -> Dict[str, Any]:
        """
        Get statistics on reward components.

        Returns breakdown of how different reward components have contributed
        to total rewards across training.
        """
        if not self.reward_component_history:
            return {"status": "no_data"}

        # Extract component values
        base_rewards = []
        format_rewards = []
        tool_rewards = []
        diversity_penalties = []
        totals = []

        for entry in self.reward_component_history:
            components = entry.get("components", {})
            base_rewards.append(components.get("base_reward", 0))
            format_rewards.append(components.get("format_reward", 0))
            tool_rewards.append(components.get("tool_reward", 0))
            diversity_penalties.append(components.get("diversity_penalty", 0))
            totals.append(components.get("total", 0))

        return {
            "sample_count": len(self.reward_component_history),
            "base_reward": {
                "mean": float(np.mean(base_rewards)),
                "std": float(np.std(base_rewards)),
                "min": float(np.min(base_rewards)),
                "max": float(np.max(base_rewards)),
            },
            "format_reward": {
                "mean": float(np.mean(format_rewards)),
                "std": float(np.std(format_rewards)),
                "contribution_pct": float(np.sum(format_rewards) / max(np.sum(totals), 1e-8) * 100),
            },
            "tool_reward": {
                "mean": float(np.mean(tool_rewards)),
                "std": float(np.std(tool_rewards)),
                "contribution_pct": float(np.sum(tool_rewards) / max(np.sum(totals), 1e-8) * 100),
            },
            "diversity_penalty": {
                "mean": float(np.mean(diversity_penalties)),
                "std": float(np.std(diversity_penalties)),
                "penalty_pct": float(np.sum(diversity_penalties) / max(np.sum(base_rewards), 1e-8) * 100),
            },
            "total_reward": {
                "mean": float(np.mean(totals)),
                "std": float(np.std(totals)),
            },
            "enabled_features": {
                "bleu_available": BLEU_AVAILABLE,
                "sklearn_available": SKLEARN_AVAILABLE,
            },
        }
