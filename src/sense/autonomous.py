"""
SENSE v3.0 Autonomous Runner - Unified Entry Point

Provides three operational modes:
1. continuous: Full self-evolution loop (Curriculum → GRPO → Orchestrator → Memory)
2. single: Solve a single task and exit
3. evolve: Run GRPO evolution for specified generations

Usage:
    sense --mode continuous
    sense --mode single --task "Calculate 15 * 23"
    sense --mode evolve --generations 10
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import time

# Handle both script and module execution
if __name__ == "__main__" or not __package__:
    # Running as script - add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sense.core.reasoning_orchestrator import ReasoningOrchestrator
    from sense.core.evolution.curriculum import CurriculumAgent
    from sense.core.evolution.grpo import GRPOTrainer
    from sense.core.evolution.population import PopulationManager
    from sense.core.memory.ltm import AgeMem
    from sense.core.plugins.forge import ToolForge
    from sense.bridge import Bridge, EmergencyStop
    try:
        from sense_v2.core.config import Config
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False
else:
    # Running as module
    from .core.reasoning_orchestrator import ReasoningOrchestrator
    from .core.evolution.curriculum import CurriculumAgent
    from .core.evolution.grpo import GRPOTrainer
    from .core.evolution.population import PopulationManager
    from .core.memory.ltm import AgeMem
    from .core.plugins.forge import ToolForge
    from .bridge import Bridge, EmergencyStop
    try:
        from sense_v2.core.config import Config
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False


class AutonomousRunner:
    """
    Unified runner that wires all SENSE v3.0 components together.

    Manages the full autonomous system with proper config propagation
    and intuitive operational modes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the autonomous runner with all components.

        Args:
            config: Optional configuration dict. If None, loads from file or uses defaults.
        """
        self.logger = logging.getLogger("AutonomousRunner")

        # Load configuration
        if config is None:
            if CONFIG_AVAILABLE:
                try:
                    config_obj = Config.from_file()
                    config = config_obj.to_dict()
                    self.logger.info("Loaded config from file")
                except Exception as e:
                    self.logger.warning(f"Config file load failed: {e}, using defaults")
                    config = self._default_config()
            else:
                config = self._default_config()

        self.config = config
        self.logger.info(f"Config: {config}")

        # Initialize core components with config
        self.memory = AgeMem(config)
        self.bridge = Bridge()
        self.tool_forge = ToolForge(config=config)
        self.orchestrator = ReasoningOrchestrator(
            tool_forge=self.tool_forge,
            config=config
        )

        # Initialize evolution components
        self.curriculum = CurriculumAgent(config)
        self.grpo = GRPOTrainer(config)
        self.population = self.grpo.population_manager

        # State tracking
        self.loop_count = 0
        self.total_successes = 0
        self.total_failures = 0

        self.logger.info("AutonomousRunner initialized successfully")

    def _default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            'population_size': 4,
            'grpo_group_size': 2,
            'grpo_temperature': 1.0,
            'grpo_kl_coeff': 0.1,
            'stm_max_entries': 50,
            'model_name': 'gpt2',
            'evolution_interval': 5,  # Evolve every N iterations
            'loop_sleep': 10,  # Seconds between loops in continuous mode
        }

    async def run(self, mode: str = "continuous", **kwargs) -> Any:
        """
        Run the autonomous system in specified mode.

        Args:
            mode: One of "continuous", "single", or "evolve"
            **kwargs: Mode-specific parameters
                - task (str): Task for single mode
                - generations (int): Generations for evolve mode

        Returns:
            Result depends on mode (TaskResult, evolution stats, etc.)
        """
        self.logger.info(f"Starting in {mode} mode")

        if mode == "continuous":
            return await self._continuous_evolution()
        elif mode == "single":
            task = kwargs.get("task", "Calculate 2 + 2")
            return await self._solve_single_task(task)
        elif mode == "evolve":
            generations = kwargs.get("generations", 10)
            return await self._run_evolution(generations)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'continuous', 'single', or 'evolve'")

    async def _continuous_evolution(self) -> None:
        """
        Run continuous self-evolution loop.

        This is the full autonomous mode: Curriculum → Task → Orchestrator → Memory → Evolution
        Runs indefinitely until interrupted or EmergencyStop is triggered.
        """
        self.logger.info("Starting continuous self-evolution loop...")
        training_interval = self.config.get('evolution_interval', 5)
        loop_sleep = self.config.get('loop_sleep', 10)

        try:
            while True:
                # Check emergency stop
                if EmergencyStop.check():
                    self.logger.warning("Emergency stop activated, halting loop")
                    break

                self.loop_count += 1
                self.logger.info(f"=== Loop {self.loop_count} ===")

                # Phase 1: Generate task from curriculum
                task = await self.curriculum.get_next_task()
                self.logger.info(f"Task: {task}")

                # Phase 2: Solve task with orchestrator
                result = await self.orchestrator.solve_task(task)

                # Update stats
                if result.success:
                    self.total_successes += 1
                else:
                    self.total_failures += 1

                # Phase 3: Store in memory
                self.memory.add_memory(
                    task,
                    result.plan,
                    result.execution_result,
                    result.success
                )

                # Phase 4: Update genome fitness
                if self.population.population:
                    best_genome = self.population.get_best_genome()
                    if best_genome:
                        # Fitness delta: +1 for success, -0.5 for failure
                        fitness_delta = 1.0 if result.success else -0.5
                        best_genome.fitness = best_genome.fitness + fitness_delta
                        self.logger.info(f"Genome fitness: {best_genome.fitness:.2f}")

                # Phase 5: Evolve population periodically
                if self.loop_count % training_interval == 0:
                    self.logger.info("Running evolution step...")
                    try:
                        self.population.evolve(generations=1)
                        self.logger.info("Evolution complete")
                    except Exception as e:
                        self.logger.warning(f"Evolution failed: {e}")

                # Log summary
                success_rate = self.total_successes / (self.total_successes + self.total_failures)
                self.logger.info(
                    f"Loop {self.loop_count} complete. "
                    f"Success: {result.success} | "
                    f"Overall: {self.total_successes}/{self.loop_count} ({success_rate:.1%})"
                )

                # Sleep before next iteration
                await asyncio.sleep(loop_sleep)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Loop error: {e}", exc_info=True)
        finally:
            self.logger.info(f"Shutting down after {self.loop_count} loops")
            self._print_summary()

    async def _solve_single_task(self, task: str) -> Any:
        """
        Solve a single task and return the result.

        Args:
            task: Task description to solve

        Returns:
            TaskResult from the orchestrator
        """
        self.logger.info(f"Solving task: {task}")

        # Retrieve memory context
        result = await self.orchestrator.solve_task(task)

        # Store in memory
        self.memory.add_memory(
            task,
            result.plan,
            result.execution_result,
            result.success
        )

        # Print result
        print("\n" + "="*60)
        print(f"Task: {task}")
        print(f"Success: {result.success}")
        print(f"Execution time: {result.execution_time:.2f}s")
        print(f"Retries: {result.retry_count}")
        print(f"\nPlan:\n{result.plan}")
        print(f"\nResult:\n{result.execution_result}")
        print(f"\nVerification:\n{result.verification.feedback}")
        print("="*60 + "\n")

        return result

    async def _run_evolution(self, generations: int) -> Dict[str, Any]:
        """
        Run GRPO evolution for specified generations.

        Args:
            generations: Number of generations to evolve

        Returns:
            Evolution statistics
        """
        self.logger.info(f"Running evolution for {generations} generations...")

        try:
            # Run evolution
            for gen in range(generations):
                self.logger.info(f"Generation {gen + 1}/{generations}")
                self.population.evolve(generations=1)

                # Get stats
                if self.population.population:
                    best = self.population.get_best_genome()
                    avg_fitness = sum(g.fitness for g in self.population.population) / len(self.population.population)
                    self.logger.info(
                        f"Best fitness: {best.fitness:.2f} | "
                        f"Avg fitness: {avg_fitness:.2f}"
                    )

            # Final stats
            stats = {
                "generations": generations,
                "population_size": len(self.population.population),
                "best_fitness": self.population.get_best_genome().fitness if self.population.population else 0,
            }

            print("\n" + "="*60)
            print("Evolution Complete")
            print(f"Generations: {generations}")
            print(f"Population size: {stats['population_size']}")
            print(f"Best fitness: {stats['best_fitness']:.2f}")
            print("="*60 + "\n")

            return stats

        except Exception as e:
            self.logger.error(f"Evolution error: {e}", exc_info=True)
            return {"error": str(e)}

    def _print_summary(self) -> None:
        """Print execution summary."""
        print("\n" + "="*60)
        print("SENSE v3.0 Autonomous Runner - Summary")
        print("="*60)
        print(f"Total loops: {self.loop_count}")
        print(f"Successes: {self.total_successes}")
        print(f"Failures: {self.total_failures}")
        if self.loop_count > 0:
            success_rate = self.total_successes / self.loop_count
            print(f"Success rate: {success_rate:.1%}")
        print(f"Memory entries: {len(self.memory.stm)}")
        if self.population.population:
            print(f"Population size: {len(self.population.population)}")
            print(f"Best fitness: {self.population.get_best_genome().fitness:.2f}")
        print("="*60 + "\n")


def main():
    """
    Main entry point for CLI.

    Parses arguments and starts the autonomous runner.
    """
    parser = argparse.ArgumentParser(
        description="SENSE v3.0 Autonomous Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start continuous self-evolution
  sense --mode continuous

  # Solve single task
  sense --mode single --task "Calculate 15 * 23"

  # Run GRPO evolution only
  sense --mode evolve --generations 10

  # Use custom config
  sense --mode continuous --config /path/to/config.json
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["continuous", "single", "evolve"],
        help="Operational mode (default: single)"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="Calculate 2 + 2",
        help="Task to solve (for single mode)"
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations (for evolve mode)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (JSON)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config if provided
    config = None
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)

    # Create runner
    runner = AutonomousRunner(config=config)

    # Run in selected mode
    try:
        if args.mode == "continuous":
            asyncio.run(runner.run("continuous"))
        elif args.mode == "single":
            asyncio.run(runner.run("single", task=args.task))
        elif args.mode == "evolve":
            asyncio.run(runner.run("evolve", generations=args.generations))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
