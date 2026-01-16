"""
SENSE-v2: Systematic Enhancement for Neural Selection and Evolution
A self-evolving, agent-driven framework merging neural evolution with autonomous operational control.

Core Components:
- Agent 0 (The School): Co-evolutionary loop using Step-wise GRPO
- Agent Zero (The Workplace): Hierarchical orchestration for OS-level tasks
- AgeMem (The Filing Cabinet): Structured agentic memory system (LTM/STM)

Architecture:
- Optimized for 128GB Unified Memory Architecture (UMA) with 256-bit bus
- vLLM with ROCm (AMD RDNA 3.5) support
- Tool-centric logic with schema-based Python tools
- Self-correction loops via stderr parsing
"""

__version__ = "2.0.0"
__author__ = "Todd Eddings"

# Core
from sense_v2.core.config import Config, HardwareConfig, EvolutionConfig, OrchestrationConfig, MemoryConfig
from sense_v2.core.base import BaseAgent, BaseTool, BaseMemory, ToolRegistry, AgentState
from sense_v2.core.schemas import ToolSchema, ToolResult, AgentMessage, RewardSignal

# Memory System (AgeMem)
from sense_v2.memory.agemem import AgeMem, create_agemem

# Agents - Agent 0 (The School)
from sense_v2.agents.agent_0.curriculum import CurriculumAgent, CurriculumTask
from sense_v2.agents.agent_0.executor import ExecutorAgent
from sense_v2.agents.agent_0.trainer import GRPOTrainer

# Agents - Agent Zero (The Workplace)
from sense_v2.agents.agent_zero.master import MasterAgent
from sense_v2.agents.agent_zero.sub_agents import TerminalAgent, FileSystemAgent, BrowserAgent

# Tools
from sense_v2.tools.terminal import TerminalTool
from sense_v2.tools.filesystem import FileReadTool, FileWriteTool, FileListTool
from sense_v2.tools.memory_tools import MemoryStoreTool, MemorySearchTool
from sense_v2.tools.anomaly import AnomalyDetectionTool

# Utilities
from sense_v2.utils.dev_log import DevLog, StateLogger
from sense_v2.utils.health import HealthMonitor, SystemHealth, get_health_monitor

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core
    "Config",
    "HardwareConfig",
    "EvolutionConfig",
    "OrchestrationConfig",
    "MemoryConfig",
    "BaseAgent",
    "BaseTool",
    "BaseMemory",
    "ToolRegistry",
    "AgentState",
    "ToolSchema",
    "ToolResult",
    "AgentMessage",
    "RewardSignal",
    # Memory
    "AgeMem",
    "create_agemem",
    # Agent 0
    "CurriculumAgent",
    "CurriculumTask",
    "ExecutorAgent",
    "GRPOTrainer",
    # Agent Zero
    "MasterAgent",
    "TerminalAgent",
    "FileSystemAgent",
    "BrowserAgent",
    # Tools
    "TerminalTool",
    "FileReadTool",
    "FileWriteTool",
    "FileListTool",
    "MemoryStoreTool",
    "MemorySearchTool",
    "AnomalyDetectionTool",
    # Utilities
    "DevLog",
    "StateLogger",
    "HealthMonitor",
    "SystemHealth",
    "get_health_monitor",
]
