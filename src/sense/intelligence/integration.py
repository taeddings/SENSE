"""
SENSE v4.0: Intelligence Integration Layer

Coordinates uncertainty detection, knowledge RAG, preferences, and metacognition.
Provides clean integration with ReasoningOrchestrator.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from sense.intelligence.uncertainty import UncertaintyDetector, UncertaintyScore, AmbiguityScore
from sense.intelligence.knowledge import KnowledgeRAG, VectorStore
from sense.intelligence.preferences import PreferenceLearner, FeedbackType
from sense.intelligence.metacognition import MetacognitiveEngine, ReasoningTrace, QualityScore
from sense.memory.bridge import UniversalMemory


@dataclass
class IntelligenceContext:
    """
    Context provided by intelligence layer before task execution.

    Attributes:
        original_task: The original user task
        knowledge_context: Retrieved knowledge context
        preference_hints: Learned user preferences
        ambiguity: Task ambiguity analysis
        trace: Reasoning trace object
        enhanced_prompt: Prompt enhanced with context and preferences
    """
    original_task: str
    knowledge_context: str = ""
    preference_hints: list = field(default_factory=list)
    ambiguity: Optional[AmbiguityScore] = None
    trace: Optional[ReasoningTrace] = None
    enhanced_prompt: Optional[str] = None


@dataclass
class IntelligenceResult:
    """
    Result from intelligence layer after task execution.

    Attributes:
        response: The generated response
        uncertainty: Uncertainty analysis of response
        quality: Reasoning quality score
        needs_clarification: Whether clarification is needed
        suggestions: Optional suggestions for improvement
    """
    response: str
    uncertainty: Optional[UncertaintyScore] = None
    quality: Optional[QualityScore] = None
    needs_clarification: bool = False
    suggestions: list = field(default_factory=list)


class IntelligenceLayer:
    """
    Unified intelligence layer that wraps orchestrator calls.

    Coordinates all intelligence components:
    - Uncertainty detection
    - Knowledge RAG
    - Preference learning
    - Metacognition
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize intelligence layer.

        Args:
            config: Optional configuration dictionary
        """
        if config is None:
            config = {}

        self.logger = logging.getLogger("Intelligence.Layer")
        self.enabled = config.get('enabled', True)

        # Initialize components
        uncertainty_config = config.get('uncertainty', {})
        self.uncertainty = UncertaintyDetector(
            threshold=uncertainty_config.get('threshold', 0.6),
            max_clarification_attempts=uncertainty_config.get('max_clarification_attempts', 2)
        )

        knowledge_config = config.get('knowledge', {})
        self.vector_store = VectorStore(
            dimension=knowledge_config.get('vector_dimension', 384),
            use_faiss=knowledge_config.get('use_faiss', True)
        )

        memory = UniversalMemory()
        self.knowledge = KnowledgeRAG(
            vector_store=self.vector_store,
            memory=memory,
            max_context_tokens=knowledge_config.get('max_context_tokens', 500)
        )

        preferences_config = config.get('preferences', {})
        self.preferences_enabled = preferences_config.get('enabled', True)
        if self.preferences_enabled:
            self.preferences = PreferenceLearner(
                decay_days=preferences_config.get('decay_days', 30)
            )
        else:
            self.preferences = None

        metacog_config = config.get('metacognition', {})
        self.metacog = MetacognitiveEngine(
            trace_enabled=metacog_config.get('trace_enabled', True),
            log_level=metacog_config.get('log_level', 'info')
        )

        self.logger.info("Intelligence Layer initialized")

    async def preprocess(
        self,
        task: str,
        system_prompt: Optional[str] = None
    ) -> IntelligenceContext:
        """
        Pre-processing before task execution.

        Steps:
        1. Check task ambiguity
        2. Retrieve relevant knowledge
        3. Get preference hints
        4. Start metacognitive trace
        5. Enhance system prompt

        Args:
            task: The task to process
            system_prompt: Optional system prompt to enhance

        Returns:
            IntelligenceContext with enriched information
        """
        if not self.enabled:
            return IntelligenceContext(
                original_task=task,
                enhanced_prompt=system_prompt
            )

        # 1. Analyze task ambiguity
        ambiguity = self.uncertainty.analyze_task_ambiguity(task)

        if ambiguity.is_ambiguous(threshold=0.7):
            self.logger.warning(
                f"Task is ambiguous (score={ambiguity.score:.2f}): {', '.join(ambiguity.reasons)}"
            )

        # 2. Retrieve knowledge context
        knowledge_context = self.knowledge.retrieve_context(task)

        # 3. Get preference hints
        preference_hints = []
        if self.preferences_enabled and self.preferences:
            preference_hints = self.preferences.get_preference_hints(task)

        # 4. Start reasoning trace
        trace = self.metacog.start_trace(task, metadata={
            'ambiguity_score': ambiguity.score,
            'knowledge_retrieved': len(knowledge_context) > 0
        })

        # 5. Enhance prompt
        enhanced_prompt = system_prompt
        if system_prompt:
            # Add knowledge context
            if knowledge_context:
                enhanced_prompt += f"\n\n{knowledge_context}"

            # Add preferences
            if self.preferences_enabled and self.preferences:
                enhanced_prompt = self.preferences.apply_preferences(
                    enhanced_prompt,
                    task
                )

        context = IntelligenceContext(
            original_task=task,
            knowledge_context=knowledge_context,
            preference_hints=preference_hints,
            ambiguity=ambiguity,
            trace=trace,
            enhanced_prompt=enhanced_prompt
        )

        return context

    async def postprocess(
        self,
        context: IntelligenceContext,
        response: str,
        attempt: int = 0
    ) -> IntelligenceResult:
        """
        Post-processing after task execution.

        Steps:
        1. Analyze response uncertainty
        2. Complete reasoning trace
        3. Evaluate reasoning quality
        4. Determine if clarification needed

        Args:
            context: The intelligence context from preprocessing
            response: The generated response
            attempt: Current clarification attempt number

        Returns:
            IntelligenceResult with analysis
        """
        if not self.enabled:
            return IntelligenceResult(response=response)

        # 1. Analyze uncertainty
        uncertainty = self.uncertainty.analyze_response(response, attempt=attempt)

        self.logger.info(
            f"Response uncertainty: confidence={uncertainty.confidence:.2f}, "
            f"level={uncertainty.level.value}"
        )

        # 2. Complete reasoning trace
        completed_trace = self.metacog.complete_trace()

        # 3. Evaluate quality
        quality = None
        if completed_trace:
            quality = completed_trace.quality_score

        # 4. Check if clarification needed
        needs_clarification = self.uncertainty.should_seek_clarification(
            uncertainty,
            attempt=attempt
        )

        # 5. Generate suggestions
        suggestions = []
        if uncertainty.confidence < 0.5:
            suggestions.append("Consider gathering more information")
        if quality and quality.efficiency < 0.6:
            suggestions.append("Reasoning could be more efficient")
        if context.ambiguity and context.ambiguity.is_ambiguous():
            suggestions.append("Task specification could be clearer")

        result = IntelligenceResult(
            response=response,
            uncertainty=uncertainty,
            quality=quality,
            needs_clarification=needs_clarification,
            suggestions=suggestions
        )

        return result

    def log_metacognitive_step(
        self,
        step_type: str,
        content: str,
        confidence: float = 1.0
    ):
        """
        Log a metacognitive reasoning step.

        Args:
            step_type: Type of step (from StepType enum)
            content: Description of the step
            confidence: Confidence in this step
        """
        self.metacog.log_step(step_type, content, confidence)

    def record_feedback(
        self,
        task: str,
        response: str,
        feedback_type: FeedbackType,
        correction: Optional[str] = None
    ):
        """
        Record user feedback for preference learning.

        Args:
            task: The task that was performed
            response: The response that was generated
            feedback_type: Type of feedback
            correction: Optional corrected response
        """
        if self.preferences_enabled and self.preferences:
            self.preferences.record_feedback(
                task=task,
                response=response,
                feedback_type=feedback_type,
                correction=correction
            )

    def add_knowledge(
        self,
        content: str,
        source: str = "user",
        metadata: Optional[Dict] = None
    ):
        """
        Add knowledge to the RAG system.

        Args:
            content: Knowledge content
            source: Source of the knowledge
            metadata: Additional metadata
        """
        self.knowledge.add_knowledge(content, source, metadata)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the intelligence layer.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'enabled': self.enabled,
            'vector_store_documents': len(self.vector_store.documents),
            'metacognition': self.metacog.get_quality_stats()
        }

        if self.preferences_enabled and self.preferences:
            stats['preferences'] = self.preferences.get_stats()

        return stats

    def should_backtrack(self) -> bool:
        """
        Check if reasoning should backtrack.

        Returns:
            True if reasoning quality is poor and should restart
        """
        return self.metacog.should_backtrack()
