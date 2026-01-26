"""
SENSE v4.0: Uncertainty Detection Module

Detects when SENSE is uncertain and should seek clarification.
Uses multi-signal analysis: linguistic markers, logprobs, response patterns.
"""

import logging
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    HIGH = "high"          # > 0.8
    MEDIUM = "medium"      # 0.5 - 0.8
    LOW = "low"            # 0.3 - 0.5
    VERY_LOW = "very_low"  # < 0.3


@dataclass
class UncertaintyScore:
    """
    Comprehensive uncertainty scoring.

    Attributes:
        confidence: Overall confidence (0.0 - 1.0)
        level: Categorical confidence level
        hedging_count: Number of hedging phrases detected
        contradiction_detected: Whether contradictions found
        repetition_score: Degree of circular reasoning (0.0 - 1.0)
        response_length_factor: Length anomaly factor (0.0 - 1.0)
        signals: List of specific uncertainty signals detected
    """
    confidence: float
    level: ConfidenceLevel
    hedging_count: int = 0
    contradiction_detected: bool = False
    repetition_score: float = 0.0
    response_length_factor: float = 0.0
    signals: List[str] = None

    def __post_init__(self):
        if self.signals is None:
            self.signals = []

    def should_clarify(self, threshold: float = 0.6) -> bool:
        """Check if uncertainty warrants clarification."""
        return self.confidence < threshold


@dataclass
class AmbiguityScore:
    """
    Task ambiguity analysis.

    Attributes:
        score: Ambiguity score (0.0 = clear, 1.0 = very ambiguous)
        reasons: List of ambiguity reasons
        missing_context: Whether task lacks context
        multiple_interpretations: Whether task has multiple meanings
        underspecified: Whether requirements are unclear
    """
    score: float
    reasons: List[str]
    missing_context: bool = False
    multiple_interpretations: bool = False
    underspecified: bool = False

    def is_ambiguous(self, threshold: float = 0.5) -> bool:
        """Check if task is ambiguous."""
        return self.score > threshold


class UncertaintyDetector:
    """
    Multi-signal uncertainty detection system.

    Analyzes responses and tasks to detect when SENSE is uncertain
    or when user input is ambiguous.
    """

    # Linguistic hedging patterns (ordered by strength)
    HEDGING_PATTERNS = [
        # Strong hedging
        (r'\b(I don\'t know|I\'m not sure|I cannot say|unclear|uncertain)\b', 0.9),
        (r'\b(perhaps|maybe|possibly|conceivably|presumably)\b', 0.8),
        (r'\b(might|could|may|would)\b', 0.6),
        (r'\b(I think|I believe|I feel|seems|appears)\b', 0.5),
        (r'\b(likely|probably|generally|typically|usually)\b', 0.4),
        # Confidence hedging (false positives to avoid)
        (r'\b(I can definitely|I\'m confident|certainly|absolutely)\b', -0.3),
    ]

    # Contradiction markers
    CONTRADICTION_PATTERNS = [
        r'\b(however|but|although|on the other hand|conversely)\b.*\b(initially|earlier|previously)\b',
        r'\b(yes).*\b(no)\b.*\b(yes)\b',
        r'\b(correct).*\b(incorrect)\b',
    ]

    # Ambiguity indicators in tasks
    AMBIGUITY_PATTERNS = {
        'vague_pronouns': r'\b(it|this|that|these|those)\b(?! (is|are|was|were))',
        'underspecified': r'\b(something|somehow|somewhere|someone)\b',
        'multiple_questions': r'\?.*\?',
        'open_ended': r'\b(what about|how about|thoughts on)\b',
    }

    def __init__(self, threshold: float = 0.6, max_clarification_attempts: int = 2):
        """
        Initialize uncertainty detector.

        Args:
            threshold: Confidence threshold below which to seek clarification (0.0 - 1.0)
            max_clarification_attempts: Maximum times to ask for clarification
        """
        self.threshold = threshold
        self.max_clarification_attempts = max_clarification_attempts
        self.logger = logging.getLogger("Intelligence.Uncertainty")

    def analyze_response(
        self,
        response: str,
        logprobs: Optional[List[float]] = None,
        attempt: int = 0
    ) -> UncertaintyScore:
        """
        Analyze response for uncertainty signals.

        Args:
            response: The response text to analyze
            logprobs: Optional log probabilities from LLM
            attempt: Current clarification attempt number

        Returns:
            UncertaintyScore with detailed confidence metrics
        """
        if not response:
            return UncertaintyScore(
                confidence=0.0,
                level=ConfidenceLevel.VERY_LOW,
                signals=["empty_response"]
            )

        signals = []

        # 1. Linguistic hedging analysis
        hedging_score, hedging_count, hedging_signals = self._analyze_hedging(response)
        signals.extend(hedging_signals)

        # 2. Logprob analysis (if available)
        logprob_score = self._analyze_logprobs(logprobs) if logprobs else 1.0
        if logprobs and logprob_score < 0.7:
            signals.append(f"low_logprob_confidence: {logprob_score:.2f}")

        # 3. Response length analysis
        length_factor = self._analyze_response_length(response)
        if length_factor < 0.8:
            signals.append(f"unusual_length: {length_factor:.2f}")

        # 4. Repetition/circular reasoning
        repetition_score = self._detect_repetition(response)
        if repetition_score > 0.3:
            signals.append(f"repetitive_reasoning: {repetition_score:.2f}")

        # 5. Contradiction detection
        contradiction = self._detect_contradictions(response)
        if contradiction:
            signals.append("internal_contradiction")

        # Combine scores (weighted average)
        confidence = (
            hedging_score * 0.4 +
            logprob_score * 0.3 +
            length_factor * 0.1 +
            (1.0 - repetition_score) * 0.1 +
            (0.0 if contradiction else 1.0) * 0.1
        )

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # Determine level
        if confidence > 0.8:
            level = ConfidenceLevel.HIGH
        elif confidence > 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence > 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        return UncertaintyScore(
            confidence=confidence,
            level=level,
            hedging_count=hedging_count,
            contradiction_detected=contradiction,
            repetition_score=repetition_score,
            response_length_factor=length_factor,
            signals=signals
        )

    def _analyze_hedging(self, text: str) -> tuple:
        """
        Analyze text for hedging language.

        Returns:
            (hedging_score, count, signals) tuple
        """
        text_lower = text.lower()
        hedging_count = 0
        hedging_score = 1.0  # Start confident
        signals = []

        for pattern, weight in self.HEDGING_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                hedging_count += len(matches)
                # Accumulate hedging (negative weights increase confidence)
                if weight > 0:
                    hedging_score -= weight * len(matches) * 0.1
                    signals.append(f"hedging: {matches[0]}")
                else:
                    hedging_score += abs(weight) * len(matches) * 0.1

        # Clamp score
        hedging_score = max(0.0, min(1.0, hedging_score))

        return hedging_score, hedging_count, signals

    def _analyze_logprobs(self, logprobs: List[float]) -> float:
        """
        Analyze log probabilities for confidence.

        Low logprobs indicate model uncertainty.
        """
        if not logprobs:
            return 1.0

        # Average logprob (higher is more confident)
        # Logprobs are typically negative, closer to 0 = more confident
        avg_logprob = sum(logprobs) / len(logprobs)

        # Convert to confidence score (0-1)
        # Assume logprobs in range [-10, 0]
        confidence = max(0.0, min(1.0, (avg_logprob + 10) / 10))

        return confidence

    def _analyze_response_length(self, text: str) -> float:
        """
        Analyze response length for anomalies.

        Very short or very long responses may indicate uncertainty.
        """
        word_count = len(text.split())

        # Expected range: 20-200 words
        if 20 <= word_count <= 200:
            return 1.0
        elif word_count < 10:
            return 0.5  # Too short
        elif word_count > 500:
            return 0.7  # Too verbose (may be overcompensating)
        else:
            # Gradual degradation
            if word_count < 20:
                return 0.5 + (word_count / 20) * 0.5
            else:
                return 1.0 - min(0.3, (word_count - 200) / 1000)

    def _detect_repetition(self, text: str) -> float:
        """
        Detect circular reasoning or excessive repetition.

        Returns repetition score (0.0 = no repetition, 1.0 = highly repetitive)
        """
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) < 2:
            return 0.0

        # Check for repeated sentences
        unique_sentences = set(sentences)
        repetition_ratio = 1.0 - (len(unique_sentences) / len(sentences))

        # Check for repeated phrases (3+ words)
        words = text.lower().split()
        phrases = []
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrases.append(phrase)

        if phrases:
            unique_phrases = set(phrases)
            phrase_repetition = 1.0 - (len(unique_phrases) / len(phrases))
        else:
            phrase_repetition = 0.0

        # Combine metrics
        return (repetition_ratio * 0.6 + phrase_repetition * 0.4)

    def _detect_contradictions(self, text: str) -> bool:
        """
        Detect internal contradictions in response.
        """
        text_lower = text.lower()

        for pattern in self.CONTRADICTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        return False

    def analyze_task_ambiguity(self, task: str) -> AmbiguityScore:
        """
        Analyze task for ambiguity.

        Detects:
        - Missing context
        - Multiple interpretations
        - Underspecified requirements
        """
        if not task:
            return AmbiguityScore(
                score=1.0,
                reasons=["empty_task"],
                missing_context=True
            )

        reasons = []
        ambiguity_signals = {}

        task_lower = task.lower()

        # Check each ambiguity pattern
        for signal_type, pattern in self.AMBIGUITY_PATTERNS.items():
            if re.search(pattern, task_lower):
                reasons.append(signal_type)
                ambiguity_signals[signal_type] = True

        # Check for very short tasks (likely underspecified)
        word_count = len(task.split())
        if word_count < 3:
            reasons.append("too_short")
            ambiguity_signals['underspecified'] = True

        # Check for missing subject
        has_subject = any(word in task_lower for word in ['i', 'you', 'we', 'they', 'the', 'a', 'an'])
        if not has_subject and word_count > 2:
            reasons.append("missing_subject")
            ambiguity_signals['missing_context'] = True

        # Calculate ambiguity score
        score = len(reasons) * 0.2
        score = min(1.0, score)

        return AmbiguityScore(
            score=score,
            reasons=reasons,
            missing_context=ambiguity_signals.get('missing_context', False),
            multiple_interpretations=ambiguity_signals.get('multiple_questions', False),
            underspecified=ambiguity_signals.get('underspecified', False)
        )

    def should_seek_clarification(
        self,
        score: UncertaintyScore,
        attempt: int = 0
    ) -> bool:
        """
        Determine if clarification should be sought.

        Edge case handling: Prevents infinite clarification loops.
        """
        # Hard limit on clarification attempts
        if attempt >= self.max_clarification_attempts:
            self.logger.info(
                f"Max clarification attempts ({self.max_clarification_attempts}) reached, proceeding"
            )
            return False

        return score.should_clarify(self.threshold)
