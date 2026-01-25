"""
Agent Zero - The Workplace
Hierarchical orchestration layer for OS-level task execution.
MasterAgent delegates to specialized sub-agents (Terminal, Files, Browser).
"""

from sense.agents.agent_zero.master import MasterAgent
from sense.agents.agent_zero.sub_agents import (
    TerminalAgent,
    FileSystemAgent,
    BrowserAgent,
)

__all__ = ["MasterAgent", "TerminalAgent", "FileSystemAgent", "BrowserAgent"]
