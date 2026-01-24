"""
SENSE-v2 PopulationManager
Manages population of ReasoningGenomes with evolutionary selection.

Part of Sprint 1: The Core (base implementation)
Part of Sprint 3: The Loop (DEAP integration)

Features:
- Population initialization and management
- Selection, crossover, and mutation operations
- LTM checkpointing for genome persistence
- Retention policy for elite preservation
- Drift-adaptive mutation rates
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import random
import asyncio
# from sense_v2.grounding import GroundingSystem  # Legacy, comment out for Phase 2
from datetime import datetime
import json
import numpy as np

from sense.core.evolution.genome import ReasoningGenome, Genome, create_random_genome
from sense_v2.core.config import EvolutionConfig

# Optional DEAP import (for Sprint 3.1)
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation_id: int
    population_size: int
    best_fitness: float
    average_fitness: float
    fitness_std: float
    worst_fitness: float
    elite_count: int
    mutation_rate: float
    drift_metric: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "population_size": self.population_size,
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "fitness_std": self.fitness_std,
            "worst_fitness": self.worst_fitness,
            "elite_count": self.elite_count,
            "mutation_rate": self.mutation_rate,
            "drift_metric": self.drift_metric,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RetentionPolicy:
    """
    Policy for genome retention in LTM.

    Determines which genomes are kept and for how long.
    """
    elite_percentile: float = 0.1  # Top 10% kept indefinitely
    prune_after_generations: int = 5  # Prune bottom 50% after 5 gens
    prune_percentile: float = 0.5  # Bottom 50% to prune
    min_population: int = 4  # Never go below this

    def should_retain(
        self,
        genome: ReasoningGenome,
        population_fitness: List[float],
    ) -> bool:
        """
        Determine if a genome should be retained.

        Args:
            genome: The genome to evaluate
            population_fitness: All fitness values in population

        Returns:
            True if genome should be retained
        """
        if not population_fitness:
            return True

        # Always retain top performers
        elite_threshold = np.percentile(population_fitness, (1 - self.elite_percentile) * 100)
        if genome.current_fitness >= elite_threshold:
            return True

        # Check age and fitness for pruning
        if genome.generation_id >= self.prune_after_generations:
            prune_threshold = np.percentile(population_fitness, self.prune_percentile * 100)
            if genome.current_fitness < prune_threshold:
                return False

        return True


class PopulationManager:
    """
    Manages a population of ReasoningGenomes.

    Responsibilities:
    - Initialize and maintain population
    - Perform selection, crossover, and mutation
    - Track generation statistics
    - Checkpoint genomes to LTM for persistence
    - Apply retention policy

    The manager supports both basic evolutionary operations and
    integration with DEAP for more advanced algorithms.

    Example:
        config = EvolutionConfig(population_size=16)
        manager = PopulationManager(config=config)
        await manager.initialize_population()

        for generation in range(100):
            fitness_scores = evaluate_population(manager.population)
            await manager.evolve(fitness_scores)
            await manager.checkpoint_generation()
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        agemem: Optional[Any] = None,  # AgeMem instance
        retention_policy: Optional[RetentionPolicy] = None,
        base_model_id: str = "default",
    ):
        """
        Initialize the PopulationManager.

        Args:
            config: Evolution configuration
            agemem: AgeMem instance for LTM checkpointing
            retention_policy: Policy for genome retention
            base_model_id: Identifier for the base model
        """
        self.config = config or EvolutionConfig()
        self.agemem = agemem
        self.retention_policy = retention_policy or RetentionPolicy()
        self.base_model_id = base_model_id
        self.logger = logging.getLogger(self.__class__.__name__)

        # Population state
        self.population: List[ReasoningGenome] = []
        self.current_generation: int = 0
        self.generation_history: List[GenerationStats] = []

        # Drift tracking
        self._drift_metric: float = 0.0
        self._previous_fitness_distribution: Optional[List[float]] = None

        # DEAP toolbox (initialized if DEAP available)
        self._toolbox: Optional[Any] = None
        if DEAP_AVAILABLE:
            self._setup_deap()

    def _setup_deap(self) -> None:
        """Set up DEAP toolbox for evolutionary operations."""
        if not DEAP_AVAILABLE:
            return

        # Create fitness and individual types if not already created
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        if not hasattr(creator, "Individual"):
            creator.create("Individual", ReasoningGenome, fitness=creator.FitnessMax)

        self._toolbox = base.Toolbox()

        # Register population and individual creation
        self._toolbox.register("individual", creator.Individual)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

        # Register selection operator (tournament selection)
        self._toolbox.register(
            "select",
            tools.selTournament,
            tournsize=3,
            fit_attr="current_fitness",
        )

        # Register crossover operator using genome crossover
        self._toolbox.register(
            "mate",
            self._deap_crossover,
        )

        # Register mutation operator using genome mutation
        self._toolbox.register(
            "mutate",
            self._deap_mutate,
        )

        # Register evaluation placeholder (will be overridden)
        self._toolbox.register(
            "evaluate",
            lambda ind: (0.0,),  # Placeholder
        )

        self.logger.info("DEAP toolbox initialized with enhanced operators")

    async def initialize_population(
        self,
        initial_genomes: Optional[List[ReasoningGenome]] = None,
    ) -> None:
        """
        Initialize the population.

        Args:
            initial_genomes: Optional pre-existing genomes to include
        """
        self.population = []

        # Add any provided genomes
        if initial_genomes:
            self.population.extend(initial_genomes)

        # Fill remaining slots with random genomes
        while len(self.population) < self.config.population_size:
            genome = create_random_genome(
                base_model_id=self.base_model_id,
                generation_id=0,
            )
            self.population.append(genome)

        self.logger.info(f"Initialized population with {len(self.population)} genomes")

    async def evolve(
        self,
        fitness_scores: List[float],
        drift_metric: Optional[float] = None,
    ) -> GenerationStats:
        """
        Evolve the population for one generation.

        Args:
            fitness_scores: Fitness values for each genome (same order as population)
            drift_metric: Optional drift metric for adaptive mutation

        Returns:
            Statistics for this generation
        """
        if len(fitness_scores) != len(self.population):
            raise ValueError(
                f"Fitness scores length ({len(fitness_scores)}) doesn't match "
                f"population size ({len(self.population)})"
            )

        # Record fitness for each genome
        for genome, fitness in zip(self.population, fitness_scores):
            genome.record_fitness(fitness)

        # Calculate drift if not provided
        if drift_metric is None:
            drift_metric = self._calculate_drift(fitness_scores)
        self._drift_metric = drift_metric

        # Adaptive mutation rate based on drift
        effective_mutation_rate = self._get_adaptive_mutation_rate(drift_metric)

        # Selection
        elite_count = self.config.selection_top_k
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending
        elite_indices = sorted_indices[:elite_count]
        elites = [self.population[i] for i in elite_indices]

        # Create new population
        new_population: List[ReasoningGenome] = []

        # Keep elites
        for genome in elites:
            new_population.append(genome)

        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Select parents
            if DEAP_AVAILABLE and self._toolbox:
                # Use tournament selection via DEAP
                parents = self._toolbox.select(
                    self.population,
                    k=2,
                )
                parent_a = parents[0]
                parent_b = parents[1]

                # Use DEAP variation operators
                offspring = [parent_a, parent_b]
                offspring = self._deap_var_and(
                    offspring,
                    crossover_prob=self.config.crossover_rate,
                    mutation_prob=effective_mutation_rate,
                    drift_metric=drift_metric,
                )
                child = offspring[0]  # Take first offspring
            else:
                # Fallback: fitness-proportionate selection
                parent_a, parent_b = self._select_parents(fitness_scores)

                # Crossover
                if random.random() < self.config.crossover_rate:
                    child = parent_a.crossover(parent_b)
                else:
                    child = parent_a.mutate(
                        mutation_rate=effective_mutation_rate,
                        drift_metric=drift_metric,
                    )

                # Mutation
                if random.random() < effective_mutation_rate:
                    child = child.mutate(
                        mutation_rate=effective_mutation_rate,
                        drift_metric=drift_metric,
                    )

            new_population.append(child)

        self.population = new_population
        self.current_generation += 1

        # Calculate generation statistics
        stats = GenerationStats(
            generation_id=self.current_generation,
            population_size=len(self.population),
            best_fitness=max(fitness_scores),
            average_fitness=np.mean(fitness_scores),
            fitness_std=np.std(fitness_scores),
            worst_fitness=min(fitness_scores),
            elite_count=elite_count,
            mutation_rate=effective_mutation_rate,
            drift_metric=drift_metric,
        )
        self.generation_history.append(stats)

        # Update drift tracking
        self._previous_fitness_distribution = fitness_scores.copy()

        self.logger.info(
            f"Generation {self.current_generation}: "
            f"best={stats.best_fitness:.4f}, avg={stats.average_fitness:.4f}, "
            f"drift={drift_metric:.3f}"
        )

        return stats

    def _select_parents(
        self,
        fitness_scores: List[float],
    ) -> Tuple[ReasoningGenome, ReasoningGenome]:
        """
        Select two parents using fitness-proportionate selection.

        Args:
            fitness_scores: Fitness values for population

        Returns:
            Two parent genomes
        """
        # Normalize fitness to probabilities
        min_fitness = min(fitness_scores)
        shifted = [f - min_fitness + 0.01 for f in fitness_scores]  # Shift to positive
        total = sum(shifted)
        probabilities = [f / total for f in shifted]

        # Select two parents
        indices = np.random.choice(
            len(self.population),
            size=2,
            replace=False,
            p=probabilities,
        )

        return self.population[indices[0]], self.population[indices[1]]

    def _deap_var_and(
        self,
        population: List[ReasoningGenome],
        crossover_prob: float,
        mutation_prob: float,
        drift_metric: float,
    ) -> List[ReasoningGenome]:
        """
        Apply DEAP-style variation (crossover and mutation) to population.

        Args:
            population: List of genomes to vary
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            drift_metric: Current drift metric for adaptive mutation

        Returns:
            Varied population
        """
        if not DEAP_AVAILABLE or not self._toolbox:
            return population

        offspring = list(population)

        # Apply crossover
        for i in range(1, len(offspring), 2):
            if random.random() < crossover_prob:
                offspring[i-1], offspring[i] = self._toolbox.mate(
                    offspring[i-1], offspring[i]
                )

        # Apply mutation
        for i in range(len(offspring)):
            if random.random() < mutation_prob:
                offspring[i], = self._toolbox.mutate(
                    offspring[i], drift_metric=drift_metric, mutation_rate=mutation_prob
                )

        return offspring

    def _deap_crossover(
        self,
        parent_a: ReasoningGenome,
        parent_b: ReasoningGenome,
    ) -> Tuple[ReasoningGenome, ReasoningGenome]:
        """
        DEAP-compatible crossover operator.

        Args:
            parent_a: First parent genome
            parent_b: Second parent genome

        Returns:
            Tuple of offspring genomes
        """
        offspring_a = parent_a.crossover(parent_b)
        offspring_b = parent_b.crossover(parent_a)
        return offspring_a, offspring_b

    def _deap_mutate(
        self,
        genome: ReasoningGenome,
        drift_metric: float = 0.0,
        mutation_rate: float = 0.1,
    ) -> Tuple[ReasoningGenome]:
        """
        DEAP-compatible mutation operator.

        Args:
            genome: Genome to mutate
            drift_metric: Drift metric for adaptive mutation
            mutation_rate: Base mutation rate

        Returns:
            Tuple containing mutated genome
        """
        mutated = genome.mutate(mutation_rate=mutation_rate, drift_metric=drift_metric)
        return (mutated,)

    def _calculate_drift(self, current_fitness: List[float]) -> float:
        """
        Calculate concept drift metric.

        Compares current fitness distribution to previous generation.

        Args:
            current_fitness: Current fitness values

        Returns:
            Drift metric (0.0 = stable, 1.0 = high drift)
        """
        if self._previous_fitness_distribution is None:
            return 0.0

        # Use KL divergence approximation
        prev = np.array(self._previous_fitness_distribution)
        curr = np.array(current_fitness)

        # Normalize to probability distributions
        prev_norm = (prev - prev.min() + 0.01) / (prev.max() - prev.min() + 0.02)
        curr_norm = (curr - curr.min() + 0.01) / (curr.max() - curr.min() + 0.02)

        prev_norm = prev_norm / prev_norm.sum()
        curr_norm = curr_norm / curr_norm.sum()

        # KL divergence (capped)
        kl = np.sum(curr_norm * np.log(curr_norm / (prev_norm + 1e-10) + 1e-10))
        drift = max(0.0, min(1.0, kl / 2.0))  # Normalize to 0-1 and clamp

        return float(drift)

    def _get_adaptive_mutation_rate(self, drift_metric: float) -> float:
        """
        Get mutation rate adapted to current drift level.

        HIGH drift -> higher mutation for exploration
        LOW drift -> lower mutation for exploitation

        Args:
            drift_metric: Current drift level (0-1)

        Returns:
            Adapted mutation rate
        """
        base_rate = self.config.mutation_rate

        if drift_metric > 0.7:
            # High drift: increase mutation significantly
            return min(0.3, base_rate * 2.0)
        elif drift_metric > 0.3:
            # Medium drift: moderate increase
            return min(0.2, base_rate * 1.5)
        else:
            # Low drift: standard rate
            return base_rate

    def mutate_genome(
        self,
        genome: ReasoningGenome,
        drift_metric: Optional[float] = None,
    ) -> ReasoningGenome:
        """
        Apply adaptive mutation to a genome.

        HIGH drift (>0.5) -> +10-20% reasoning_budget
        LOW drift (<0.2) -> Decrease reasoning_budget for efficiency

        Args:
            genome: Genome to mutate
            drift_metric: Current drift level (uses stored if not provided)

        Returns:
            New mutated genome
        """
        drift = drift_metric if drift_metric is not None else self._drift_metric
        mutation_rate = self._get_adaptive_mutation_rate(drift)

        return genome.mutate(mutation_rate=mutation_rate, drift_metric=drift)

    def crossover_prompts(
        self,
        parent_a: ReasoningGenome,
        parent_b: ReasoningGenome,
    ) -> List[str]:
        """
        Merge thinking patterns from two parents weighted by success.

        Args:
            parent_a: First parent genome
            parent_b: Second parent genome

        Returns:
            Merged list of thinking patterns
        """
        # Weight by fitness
        fitness_a = parent_a.current_fitness + 0.01
        fitness_b = parent_b.current_fitness + 0.01
        total = fitness_a + fitness_b

        weight_a = fitness_a / total
        weight_b = fitness_b / total

        # Combine patterns
        all_patterns = []

        # Add patterns from parent_a with probability weight_a
        for pattern in parent_a.thinking_patterns:
            if random.random() < weight_a:
                all_patterns.append(pattern)

        # Add patterns from parent_b with probability weight_b
        for pattern in parent_b.thinking_patterns:
            if pattern not in all_patterns and random.random() < weight_b:
                all_patterns.append(pattern)

        # Ensure at least one pattern
        if not all_patterns:
            all_patterns = parent_a.thinking_patterns[:1] or parent_b.thinking_patterns[:1]

        # Limit to reasonable size
        return all_patterns[:10]

    async def checkpoint_generation(
        self,
        generation_id: Optional[int] = None,
    ) -> int:
        """
        Store genomes to AgeMem LTM with FAISS embedding.

        Args:
            generation_id: Generation to checkpoint (defaults to current)

        Returns:
            Number of genomes checkpointed
        """
        if self.agemem is None:
            self.logger.warning("No AgeMem instance - skipping checkpoint")
            return 0

        gen_id = generation_id if generation_id is not None else self.current_generation
        checkpointed = 0

        # Get current fitness values for retention policy
        fitness_values = [g.current_fitness for g in self.population]

        for genome in self.population:
            # Check retention policy
            if not self.retention_policy.should_retain(genome, fitness_values):
                continue

            # Create embedding vector for FAISS
            embedding = genome.get_embedding_vector()

            # Store genome data
            key = f"genome_{genome.genome_id}_gen{gen_id}"
            metadata = {
                "genome_id": genome.genome_id,
                "generation": gen_id,
                "fitness": genome.current_fitness,
                "best_fitness": genome.best_fitness,
                "embedding": embedding,
                "is_elite": genome.current_fitness >= np.percentile(
                    fitness_values,
                    (1 - self.retention_policy.elite_percentile) * 100
                ),
                "checkpointed_at": datetime.now().isoformat(),
            }

            try:
                await self.agemem.store(
                    key=key,
                    value=json.dumps(genome.to_dict()),
                    metadata=metadata,
                    persist=True,
                    priority=genome.current_fitness,
                )
                checkpointed += 1
            except Exception as e:
                self.logger.error(f"Failed to checkpoint genome {genome.genome_id}: {e}")

        self.logger.info(f"Checkpointed {checkpointed} genomes for generation {gen_id}")
        return checkpointed

    async def load_from_checkpoint(
        self,
        generation_id: Optional[int] = None,
    ) -> int:
        """
        Load genomes from LTM checkpoint.

        Args:
            generation_id: Specific generation to load (latest if None)

        Returns:
            Number of genomes loaded
        """
        if self.agemem is None:
            self.logger.warning("No AgeMem instance - cannot load checkpoint")
            return 0

        # Search for checkpointed genomes
        query = f"genome generation {generation_id}" if generation_id else "genome"
        results = await self.agemem.search(
            query=query,
            top_k=self.config.population_size * 2,  # Get extras for filtering
        )

        loaded = 0
        loaded_genomes: List[ReasoningGenome] = []

        for result in results:
            try:
                content = result.get("content", "")
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if "genome_type" in data and data["genome_type"] == "ReasoningGenome":
                    genome = ReasoningGenome.from_dict(data)
                    loaded_genomes.append(genome)
                    loaded += 1
            except Exception as e:
                self.logger.warning(f"Failed to load genome: {e}")
                continue

        if loaded_genomes:
            self.population = loaded_genomes[:self.config.population_size]
            # Update generation counter
            max_gen = max(g.generation_id for g in self.population)
            self.current_generation = max_gen

        self.logger.info(f"Loaded {loaded} genomes from checkpoint")
        return loaded

    async def prune_old_checkpoints(self, keep_generations: int = 10) -> int:
        """
        Remove old checkpoints from LTM based on retention policy.

        Args:
            keep_generations: Number of recent generations to keep

        Returns:
            Number of checkpoints removed
        """
        if self.agemem is None:
            return 0

        removed = 0
        cutoff_generation = self.current_generation - keep_generations

        # This would need iteration over LTM entries
        # Implementation depends on AgeMem's iteration capabilities
        self.logger.info(f"Pruning checkpoints older than generation {cutoff_generation}")

        return removed

    def get_best_genome(self) -> Optional[ReasoningGenome]:
        """Get the genome with highest current fitness."""
        if not self.population:
            return None
        return max(self.population, key=lambda g: g.current_fitness)

    def get_elite_genomes(self, count: Optional[int] = None) -> List[ReasoningGenome]:
        """
        Get the top-performing genomes.

        Args:
            count: Number of elites to return (defaults to selection_top_k)

        Returns:
            List of elite genomes sorted by fitness (descending)
        """
        n = count if count is not None else self.config.selection_top_k
        sorted_pop = sorted(
            self.population,
            key=lambda g: g.current_fitness,
            reverse=True,
        )
        return sorted_pop[:n]

    def get_diversity_metric(self) -> float:
        """
        Calculate population diversity.

        Uses hyperparameter variance as a proxy for diversity.

        Returns:
            Diversity metric (higher = more diverse)
        """
        if len(self.population) < 2:
            return 0.0

        # Collect hyperparameter vectors
        vectors = [g.get_embedding_vector() for g in self.population]
        vectors_array = np.array(vectors)

        # Calculate variance across each dimension
        variances = np.var(vectors_array, axis=0)

        # Return mean variance as diversity metric
        return float(np.mean(variances))

    def get_population_stats(self) -> Dict[str, Any]:
        """Get current population statistics."""
        if not self.population:
            return {"status": "empty"}

        fitness_values = [g.current_fitness for g in self.population]

        return {
            "generation": self.current_generation,
            "population_size": len(self.population),
            "best_fitness": max(fitness_values),
            "average_fitness": np.mean(fitness_values),
            "fitness_std": np.std(fitness_values),
            "worst_fitness": min(fitness_values),
            "diversity": self.get_diversity_metric(),
            "drift_metric": self._drift_metric,
            "deap_available": DEAP_AVAILABLE,
            "best_genome_id": self.get_best_genome().genome_id if self.get_best_genome() else None,
        }
