"""
SENSE v2.3 - Self-Evolving Neural System Engine

A model-agnostic intelligence amplification wrapper that transforms any LLM/SLM
into a self-evolving, grounded, agentic system with persistent memory.

Core Philosophy: Intelligence through architecture, not scale.
"""

from .core import (
    # Orchestrator
    ReasoningOrchestrator,
    UnifiedGrounding,
    TaskResult,
    VerificationResult,
    Phase,
    create_orchestrator,
    # ToolForge
    ToolForge,
    CodePattern,
    CandidateSkill,
    ProposedPlugin,
    ForgeStatus,
    create_tool_forge,
)

__version__ = "2.3.0"

__all__ = [
    # Orchestrator
    "ReasoningOrchestrator",
    "UnifiedGrounding",
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
    # Version
    "__version__",
]
