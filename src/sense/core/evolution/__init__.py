"""
SENSE Evolution Module
Contains genome definitions and population management for evolutionary optimization.
"""

from sense.core.evolution.genome import ReasoningGenome, Genome
from sense.core.evolution.population import PopulationManager

__all__ = [
    "ReasoningGenome",
    "Genome",
    "PopulationManager",
]
