"""
SENSE Evolution Module
Contains genome definitions and population management for evolutionary optimization.
"""

from .genome import ReasoningGenome, Genome
from .population import PopulationManager

__all__ = [
    "ReasoningGenome",
    "Genome",
    "PopulationManager",
]
