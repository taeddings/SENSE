"""
Uncertainty Detection Module.

Detects ambiguous or low-confidence situations that require human intervention.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import asyncio

@dataclass
class UncertaintySignal:
    """Detected uncertainty in reasoning."""
    location: str  # "planning", "execution", "verification"
    confidence: float  # 0.0 - 1.0
    ambiguity_type: str  # "interpretation", "method", "parameter"
    question: str  # Generated clarification question
    options: List[Dict[str, Any]]  # Possible interpretations

class UncertaintyDetector:
    """
    Detects when SENSE should ask for human guidance.
    
    Triggers:
    - Low verification confidence (< 0.6)
    - Multiple equally-valid interpretations
    - Conflicting grounding signals
    - Novel task category
    """

    CONFIDENCE_THRESHOLD = 0.6
    AMBIGUITY_THRESHOLD = 0.8  # Similarity between interpretations

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("UncertaintyDetector")
        self.history = []  # Track past uncertainties

    def detect_uncertainty(
        self, 
        task: str,
        verification: Any = None,  # VerificationResult
        interpretations: Optional[List[str]] = None
    ) -> Optional[UncertaintySignal]:
        """
        Analyze task and results for uncertainty signals.
        """
        # Signal 1: Low confidence (if verification provided)
        if verification and verification.confidence < self.CONFIDENCE_THRESHOLD:
            return UncertaintySignal(
                location="verification",
                confidence=verification.confidence,
                ambiguity_type="method",
                question=f"The approach for '{task}' has low confidence ({verification.confidence:.2f}). Which method should I use?",
                options=self._generate_method_options(task)
            )

        # Signal 2: Multiple interpretations (Ambiguity)
        if interpretations and len(interpretations) > 1:
            # Simplified similarity check stub
            similarity = 0.5 
            if similarity < self.AMBIGUITY_THRESHOLD: # If distinct enough
                return UncertaintySignal(
                    location="planning",
                    confidence=0.5,
                    ambiguity_type="interpretation",
                    question=f"I found {len(interpretations)} possible meanings for '{task}'. Which did you intend?",
                    options=[{"label": interp, "description": "Interpretation"} for interp in interpretations]
                )

        # Signal 3: Novel task category
        if self._is_novel_task(task):
            return UncertaintySignal(
                location="planning",
                confidence=0.5,
                ambiguity_type="interpretation",
                question=f"I haven't seen a task like '{task}' before. What's the goal?",
                options=self._generate_goal_options(task)
            )

        return None

    def _is_novel_task(self, task: str) -> bool:
        """Check if task category is new (stub)."""
        # In real impl, query AgeMem LTM
        return False

    def _generate_method_options(self, task: str) -> List[Dict[str, Any]]:
        """Generate alternative method options for ambiguous tasks."""
        return [
            {"label": "Standard Approach", "description": "Standard execution"},
            {"label": "Robust Approach", "description": "Slower but safer"},
        ]

    def _generate_goal_options(self, task: str) -> List[Dict[str, Any]]:
        return [
            {"label": "Execute Code", "description": "Run Python code"},
            {"label": "Analyze Only", "description": "Provide analysis"},
        ]
