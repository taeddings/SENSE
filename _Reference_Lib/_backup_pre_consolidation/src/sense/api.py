"""FastAPI Server for SENSE v3.0"""

import os
import sys

# Handle both script and module execution
if __name__ == "__main__" or not __package__:
    # Running as script - add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sense.core.reasoning_orchestrator import ReasoningOrchestrator
    from sense.core.evolution.grpo import GRPOTrainer
else:
    # Running as module
    from .core.reasoning_orchestrator import ReasoningOrchestrator
    from .core.evolution.grpo import GRPOTrainer

from fastapi import FastAPI
import asyncio

app = FastAPI(title="SENSE API", version="3.0")

# Global instances
orch = ReasoningOrchestrator()

@app.post("/solve")
async def solve_task(task: str):
    """
    Solve a task using the full Reflexion loop.
    """
    result = await orch.solve_task(task)
    return result.to_dict()

@app.post("/evolve")
def evolve_population(generations: int = 1):
    """
    Run GRPO evolution for specified generations.
    """
    grpo = GRPOTrainer({'population_size': 8})
    elites = grpo.train(generations)
    return {"elites": [str(e) for e in elites], "generations": generations}

@app.get("/stats")
def get_stats():
    """
    Get execution stats.
    """
    return orch.get_execution_stats()

# Run with: uvicorn sense.api:app --host 0.0.0.0 --port 8000


def main():
    """Main entry point for console script."""
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="SENSE v3.0 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    uvicorn.run(
        "sense.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
