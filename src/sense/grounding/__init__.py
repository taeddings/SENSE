"""
SENSE-v2 Unified Grounding System
Combines three tiers of grounding for fitness calculation in evolution.

Part of Phase 1: Three-Tier Grounding System
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from sense.core.schemas import ToolResult  # Assuming existing schema for results

@dataclass
class GroundingSource(Enum):
    """Enum for grounding sources."""
    SYNTHETIC = "synthetic"
    REALWORLD = "realworld"
    EXPERIENTIAL = "experiential"

@dataclass
class GroundingResult:
    """Result from a single grounding verification."""
    confidence: float  # 0.0 to 1.0
    source: GroundingSource
    evidence: str  # Brief explanation or data
    timestamp: Optional[float] = None  # Unix timestamp
    verified: bool = field(default=False)  # Binary success

class UnifiedGrounding:
    """
    Combines outputs from all three grounding tiers.
    
    Weights: synthetic=0.4, realworld=0.3, experiential=0.3 (configurable).
    """
    
    def __init__(self, weights: Optional[Dict[GroundingSource, float]] = None):
        if weights is None:
            self.weights = {
                GroundingSource.SYNTHETIC: 0.4,
                GroundingSource.REALWORLD: 0.3,
                GroundingSource.EXPERIENTIAL: 0.3,
            }
        else:
            self.weights = weights
        
        # Import tiers lazily to avoid circular imports
        from .synthetic import SyntheticGrounding
        from .realworld import RealWorldGrounding
        from .experiential import ExperientialGrounding
        
        self.synthetic = SyntheticGrounding()
        self.realworld = RealWorldGrounding()
        self.experiential = ExperientialGrounding()
    
    def verify(self, claim: str, context: Dict[str, Any]) -> List[GroundingResult]:
        """
        Route claim to appropriate tiers and combine results.
        
        Args:
            claim: The statement or action outcome to verify.
            context: Additional data (e.g., code for synthetic, tool results for experiential).
            
        Returns:
            List of GroundingResult from each tier.
        """
        results = []
        
        # Tier 1: Synthetic (always attempt)
        try:
            syn_result = self.synthetic.verify(claim, context)
            results.append(syn_result)
        except Exception as e:
            # FLAG: Potential improvement – Log synthetic failure and fallback
            syn_result = GroundingResult(
                confidence=0.0, source=GroundingSource.SYNTHETIC,
                evidence=f"Synthetic verification failed: {str(e)}"
            )
            results.append(syn_result)
        
        # Tier 2: Realworld (if claim is factual)
        if self._is_factual(claim):
            try:
                rw_result = self.realworld.verify(claim, context)
                results.append(rw_result)
            except Exception as e:
                rw_result = GroundingResult(
                    confidence=0.0, source=GroundingSource.REALWORLD,
                    evidence=f"Realworld verification failed: {str(e)}"
                )
                results.append(rw_result)
        
        # Tier 3: Experiential (if action-based)
        if "tool_result" in context:
            try:
                exp_result = self.experiential.verify(claim, context)
                results.append(exp_result)
            except Exception as e:
                exp_result = GroundingResult(
                    confidence=0.0, source=GroundingSource.EXPERIENTIAL,
                    evidence=f"Experiential verification failed: {str(e)}"
                )
                results.append(exp_result)
        
        return results
    
    def calculate_fitness_component(self, results: List[GroundingResult]) -> float:
        """
        Compute weighted fitness from grounding results.
        
        Args:
            results: List of GroundingResult.
            
        Returns:
            Aggregated fitness score (0.0 to 1.0).
        """
        total_fitness = 0.0
        for result in results:
            weight = self.weights.get(result.source, 0.0)
            total_fitness += result.confidence * weight
        
        # Normalize if needed (sum weights should be 1.0)
        return total_fitness
    
    def _is_factual(self, claim: str) -> bool:
        """Simple heuristic to detect factual claims (e.g., contains '?', numbers, proper nouns)."""
        # FLAG: Potential improvement – Use more advanced NLP for claim classification
        factual_indicators = ["what", "who", "when", "where", "why", "how", "is", "are"]
        return any(indicator in claim.lower() for indicator in factual_indicators) or any(char.isdigit() for char in claim)

# Export for convenience
__all__ = ["GroundingSource", "GroundingResult", "UnifiedGrounding"]
