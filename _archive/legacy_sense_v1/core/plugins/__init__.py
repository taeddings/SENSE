"""
SENSE Plugins Module
Contains ToolForge, plugin interface, and ingestion pipeline.
"""

# ToolForge Components
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

# Base Interface & Sensors
from .interface import PluginABC, PluginCapability, PluginManifest
from .mock_sensor import MockSensorPlugin

__all__ = [
    # Forge
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
    # Interface
    "PluginABC",
    "PluginCapability",
    "PluginManifest",
    "MockSensorPlugin",
]