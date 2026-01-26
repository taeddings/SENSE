"""SENSE v3.0 Main: Self-Evolution Loop"""

import os
import sys
import asyncio
import time
import logging

# Handle both script and module execution
if __name__ == "__main__" or not __package__:
    # Running as script - add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sense.core.reasoning_orchestrator import ReasoningOrchestrator
    from sense.core.evolution.curriculum import CurriculumAgent
    from sense.core.evolution.grpo import GRPOTrainer
    from sense.core.evolution.population import PopulationManager
    from sense.core.memory.ltm import AgeMem
else:
    # Running as module
    from .core.reasoning_orchestrator import ReasoningOrchestrator
    from .core.evolution.curriculum import CurriculumAgent
    from .core.evolution.grpo import GRPOTrainer
    from .core.evolution.population import PopulationManager
    from .core.memory.ltm import AgeMem

logging.basicConfig(level=logging.INFO)

async def self_evolution_loop():
    """
    Full self-evolution: Curriculum → GRPO → Orchestrator → Memory.
    """
    config = {
        'population_size': 4,
        'grpo_group_size': 2,
        'stm_max_entries': 50
    }
    curriculum = CurriculumAgent(config)
    grpo = GRPOTrainer(config)  # Now enabled!
    population = grpo.population_manager
    memory = AgeMem(config)
    orch = ReasoningOrchestrator()

    logger = logging.getLogger("MainLoop")
    loop_count = 0
    training_interval = 5  # Train GRPO every N iterations

    while True:
        loop_count += 1
        logger.info(f"Starting loop {loop_count}")

        # Generate task
        task = await curriculum.get_next_task()
        logger.info(f"Task: {task}")

        # Solve with memory
        result = await orch.solve_task(task)

        # Store in memory
        memory.add_memory(task, result.plan, result.execution_result, result.success)

        # Update genome fitness based on task success
        if population.population:
            # Use the best genome's fitness as proxy
            best_genome = population.get_best_genome()
            if best_genome:
                # Update fitness: success = +1, failure = -0.5
                fitness_delta = 1.0 if result.success else -0.5
                best_genome.fitness = best_genome.fitness + fitness_delta
                logger.info(f"Updated genome fitness: {best_genome.fitness}")

        # Evolve population periodically
        if loop_count % training_interval == 0:
            logger.info("Running evolution step...")
            try:
                # Evolve population
                population.evolve(generations=1)
                logger.info("Evolution step complete")

                # Train GRPO
                # grpo.train(generations=1)  # Optional: full GRPO training
            except Exception as e:
                logger.warning(f"Evolution step failed: {e}")

        logger.info(f"Loop {loop_count} complete. Success: {result.success}")
        time.sleep(10)  # Slow loop for demo

def main():
    """Entry point for console script."""
    asyncio.run(self_evolution_loop())

if __name__ == "__main__":
    main()
