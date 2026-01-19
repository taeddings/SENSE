"""
SENSE Evolution Module
Contains genome definitions and population management for evolutionary optimization.
"""

from core.evolution.genome import ReasoningGenome, Genome
from core.evolution.population import PopulationManager

__all__ = [
    "ReasoningGenome",
    "Genome",
    "PopulationManager",
]
