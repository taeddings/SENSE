"""
Agent 0 - The School
Co-evolutionary loop using Step-wise GRPO where a Teacher agent generates
curriculum and a Student agent evolves via tool-verified success.
"""

from sense_v2.agents.agent_0.curriculum import CurriculumAgent
from sense_v2.agents.agent_0.executor import ExecutorAgent
from sense_v2.agents.agent_0.trainer import GRPOTrainer

__all__ = ["CurriculumAgent", "ExecutorAgent", "GRPOTrainer"]
