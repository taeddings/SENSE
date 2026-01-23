"""
SENSE v2.3 Plugins Module

Contains the ToolForge and plugin management for dynamic tool creation.
"""

from .forge import (
    ToolForge,
    CodePattern,
    CandidateSkill,
    ProposedPlugin,
    ForgeStatus,
    PatternMatcher,
    CodeAbstractor,
    SyntheticVerifier,
    PluginGenerator,
    create_tool_forge,
)

__all__ = [
    "ToolForge",
    "CodePattern",
    "CandidateSkill",
    "ProposedPlugin",
    "ForgeStatus",
    "PatternMatcher",
    "CodeAbstractor",
    "SyntheticVerifier",
    "PluginGenerator",
    "create_tool_forge",
]
