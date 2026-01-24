"""GRPO Trainer for SENSE v3.0

Group Relative Policy Optimization for evolving prompt fragments and genomes.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from deap import base, creator, tools
import logging
import multiprocessing as mp
import random
from .genome import Genome
from .population import PopulationManager

logger = logging.getLogger("GRPOTrainer")

class GRPOTrainer:
    """
    GRPO: Group Relative Policy Optimization.
    Evolves prompt fragments/genomes by comparing groups on fitness.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.group_size = config.get('grpo_group_size', 8)
        self.temperature = config.get('grpo_temperature', 1.0)
        self.kl_coeff = config.get('grpo_kl_coeff', 0.1)
        self.population_manager = PopulationManager(config)
        self.setup_deap()
        self.logger = logger

    def setup_deap(self):
        creator.create("GRPOFitnessMax", base.Fitness, weights=(1.0,))
        creator.create("GRPOIndividual", Genome, fitness=creator.GRPOFitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initRepeat, creator.GRPOIndividual, lambda: Genome.random(), n=10)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_fitness)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.select_group_relative)

    def evaluate_fitness(self, individual: Genome) -> Tuple[float]:
        """Evaluate genome on curriculum tasks."""
        score = individual.test_on_curriculum()  # Stub: integrate with CurriculumAgent
        return score,

    def crossover(self, ind1: Genome, ind2: Genome) -> Tuple[Genome, Genome]:
        """DEAP crossover with KL divergence penalty."""
        offspring1, offspring2 = tools.cxTwoPoint(ind1, ind2)
        offspring1.fitness.values = (0,)
        offspring2.fitness.values = (0,)
        return offspring1, offspring2

    def mutate(self, individual: Genome, indpb: float) -> Genome:
        """Mutation with backward transfer."""
        individual.mutate(indpb)
        individual.fitness.values = (0,)
        return individual,

    def select_group_relative(self, individuals: List[Genome], k: int) -> List[Genome]:
        """GRPO selection: Relative policy within groups with KL penalty."""
        groups = [individuals[i:i+self.group_size] for i in range(0, len(individuals), self.group_size)]
        selected = []
        for group in groups:
            if len(group) < 2:
                selected.extend(group)
                continue
            # Compute relative fitness with KL regularization
            fitnesses = [ind.fitness.values[0] for ind in group]
            mean_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses) + 1e-6  # Avoid div by zero
            # Normalize and add KL coeff
            normalized = (np.array(fitnesses) - mean_fitness) / std_fitness
            relative = np.exp(normalized / self.temperature)
            # KL penalty: penalize deviations from uniform
            uniform_prob = 1.0 / len(group)
            kl_penalty = self.kl_coeff * np.sum(relative * np.log(relative / uniform_prob + 1e-8))
            relative = relative * np.exp(-kl_penalty / len(group))
            relative /= np.sum(relative)
            # Sample
            selected.extend(np.random.choice(group, min(k, len(group)), p=relative, replace=False))
        return selected[:k]

    def train(self, generations: int) -> List[Genome]:
        """Run GRPO training loop with parallel evaluation."""
        pop = self.toolbox.population(n=self.config.get('population_size', 16))
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for gen in range(generations):
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.config.get('crossover_rate', 0.7):
                        self.toolbox.mate(child1, child2)
                    if random.random() < self.config.get('mutation_rate', 0.05):
                        self.toolbox.mutate(child1, 0.05)
                        self.toolbox.mutate(child2, 0.05)
                # Parallel fitness evaluation
                fitnesses = pool.map(self.evaluate_fitness, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
                pop[:] = offspring
                mean_fit = np.mean([ind.fitness.values[0] for ind in pop])
                self.logger.info(f"Gen {gen}: Mean fitness {mean_fit:.3f}")
                # Early stopping if fitness plateau
                if gen > 5 and mean_fit < 0.1:
                    self.logger.warning("Early stopping due to low fitness")
                    break
        return tools.selBest(pop, k=10)