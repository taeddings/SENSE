"""
SENSE v4.0 Intelligence Layer

Provides robust intelligence capabilities:
- Uncertainty detection and confidence scoring
- Knowledge RAG with vector retrieval
- Preference learning from user feedback
- Metacognitive reasoning quality awareness
"""

from sense.intelligence.uncertainty import (
    UncertaintyDetector,
    UncertaintyScore,
    AmbiguityScore,
    ConfidenceLevel
)

from sense.intelligence.metacognition import (
    MetacognitiveEngine,
    ReasoningTrace,
    ReasoningStep,
    QualityScore,
    StepType
)

from sense.intelligence.knowledge import (
    KnowledgeRAG,
    VectorStore,
    Document,
    SearchResult,
    FactCheckResult
)

from sense.intelligence.preferences import (
    PreferenceLearner,
    FeedbackType,
    PreferenceHint
)

from sense.intelligence.integration import (
    IntelligenceLayer,
    IntelligenceContext,
    IntelligenceResult
)

__all__ = [
    # Uncertainty
    "UncertaintyDetector",
    "UncertaintyScore",
    "AmbiguityScore",
    "ConfidenceLevel",
    # Metacognition
    "MetacognitiveEngine",
    "ReasoningTrace",
    "ReasoningStep",
    "QualityScore",
    "StepType",
    # Knowledge
    "KnowledgeRAG",
    "VectorStore",
    "Document",
    "SearchResult",
    "FactCheckResult",
    # Preferences
    "PreferenceLearner",
    "FeedbackType",
    "PreferenceHint",
    # Integration
    "IntelligenceLayer",
    "IntelligenceContext",
    "IntelligenceResult",
]

__version__ = "4.0.0"
