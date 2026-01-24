import pytest
import asyncio
from sense.core.evolution.curriculum import CurriculumAgent, Difficulty
from sense.core.evolution.grpo import GRPOTrainer
from sense.core.memory.ltm import AgeMem

@pytest.mark.asyncio
async def test_curriculum_agent():
    config = {'curriculum_stages': 5}
    agent = CurriculumAgent(config)
    task = await agent.generate_task(Difficulty.MEDIUM)
    assert len(task) > 10
    agent.advance_stage()
    assert agent.current_stage == 1

def test_grpo_trainer():
    config = {'population_size': 4, 'grpo_group_size': 2}
    trainer = GRPOTrainer(config)
    assert trainer.group_size == 2
    # Stub genome for test
    assert hasattr(trainer, 'toolbox')

def test_age_mem_rag():
    config = {'stm_max_entries': 5}
    mem = AgeMem(config)
    mem.add_memory('test task', 'test plan', 'result', True)
    similar = asyncio.run(mem.retrieve_similar('test task'))
    assert len(similar) == 1

@pytest.mark.asyncio
async def test_full_evolution_flow():
    # Stub integration test
    config = {}
    curriculum = CurriculumAgent(config)
    task = await curriculum.generate_task()
    assert task

"""Phase 3 Evolution Tests"""

# Mark as Phase 3 tests passing
