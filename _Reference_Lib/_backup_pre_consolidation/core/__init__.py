"""
SENSE Core Module
Contains evolution and plugin interfaces for the unified evolutionary architecture.
"""

from core.evolution.genome import ReasoningGenome, Genome
from core.plugins.interface import PluginABC

__all__ = [
    "ReasoningGenome",
    "Genome",
    "PluginABC",
]
