"""SENSE v3.0 Main: Self-Evolution Loop"""

import os
import sys
import asyncio
import time
import logging
from sense.core.reasoning_orchestrator import ReasoningOrchestrator
from sense.core.evolution.curriculum import CurriculumAgent
from sense.core.evolution.grpo import GRPOTrainer
from sense.core.memory.ltm import AgeMem

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
    # grpo = GRPOTrainer(config)  # Stub for now
    memory = AgeMem(config)
    orch = ReasoningOrchestrator()

    logger = logging.getLogger("MainLoop")
    loop_count = 0

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

        # Evolve if successful (stub for GRPO)
        # if result.success:
        #     grpo.train(generations=1)

        logger.info(f"Loop {loop_count} complete. Success: {result.success}")
        time.sleep(10)  # Slow loop for demo

def main():
    """Entry point for console script."""
    asyncio.run(self_evolution_loop())

if __name__ == "__main__":
    main()