"""FastAPI Server for SENSE v3.0"""

from fastapi import FastAPI
import asyncio
from sense.core.reasoning_orchestrator import ReasoningOrchestrator
from sense.core.evolution.grpo import GRPOTrainer

app = FastAPI(title="SENSE API", version="3.0")

# Global instances
orch = ReasoningOrchestrator()

@app.post("/solve")
async def solve_task(task: str):
    \"\"\"
    Solve a task using the full Reflexion loop.
    \"\"\"
    result = await orch.solve_task(task)
    return result.to_dict()

@app.post("/evolve")
def evolve_population(generations: int = 1):
    \"\"\"
    Run GRPO evolution for specified generations.
    \"\"\"
    grpo = GRPOTrainer({'population_size': 8})
    elites = grpo.train(generations)
    return {"elites": [str(e) for e in elites], "generations": generations}

@app.get("/stats")
def get_stats():
    \"\"\"
    Get execution stats.
    \"\"\"
    return orch.get_execution_stats()

# Run with: uvicorn sense.api:app --host 0.0.0.0 --port 8000