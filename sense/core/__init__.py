"""
SENSE v2.3 Core Module

Contains the ReasoningOrchestrator and supporting components for the Reflexion loop.
"""

from .reasoning_orchestrator import (
    ReasoningOrchestrator,
    UnifiedGrounding,
    ToolForgeStub,
    TaskResult,
    VerificationResult,
    Phase,
    create_orchestrator,
)

from .plugins import (
    ToolForge,
    CodePattern,
    CandidateSkill,
    ProposedPlugin,
    ForgeStatus,
    create_tool_forge,
)

__all__ = [
    # Orchestrator
    "ReasoningOrchestrator",
    "UnifiedGrounding",
    "ToolForgeStub",
    "TaskResult",
    "VerificationResult",
    "Phase",
    "create_orchestrator",
    # ToolForge
    "ToolForge",
    "CodePattern",
    "CandidateSkill",
    "ProposedPlugin",
    "ForgeStatus",
    "create_tool_forge",
]
