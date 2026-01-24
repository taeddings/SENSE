"""
SENSE-v2 Agents Module
Contains Agent 0 (The School) and Agent Zero (The Workplace) implementations.
"""

from sense_v2.agents.agent_0.curriculum import CurriculumAgent
from sense_v2.agents.agent_0.executor import ExecutorAgent
from sense_v2.agents.agent_0.trainer import GRPOTrainer
from sense_v2.agents.agent_zero.master import MasterAgent
from sense_v2.agents.agent_zero.sub_agents import (
    TerminalAgent,
    FileSystemAgent,
    BrowserAgent,
)

__all__ = [
    "CurriculumAgent",
    "ExecutorAgent",
    "GRPOTrainer",
    "MasterAgent",
    "TerminalAgent",
    "FileSystemAgent",
    "BrowserAgent",
]
