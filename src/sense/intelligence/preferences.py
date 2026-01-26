"""
SENSE v4.0: Preference Learning Module

Learns from user feedback to personalize responses.
Uses Bayesian preference model with decay.
"""

import logging
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class FeedbackType(Enum):
    """Types of user feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    NEUTRAL = "neutral"


@dataclass
class FeedbackEntry:
    """
    Single feedback entry.

    Attributes:
        task: The task that was performed
        response: The response that was generated
        feedback_type: Type of feedback given
        correction: Optional corrected response
        timestamp: When feedback was given
        metadata: Additional feedback metadata
    """
    task: str
    response: str
    feedback_type: str
    correction: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task': self.task,
            'response': self.response,
            'feedback_type': self.feedback_type,
            'correction': self.correction,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Deserialize from dictionary."""
        return cls(
            task=data['task'],
            response=data['response'],
            feedback_type=data['feedback_type'],
            correction=data.get('correction'),
            timestamp=data.get('timestamp', time.time()),
            metadata=data.get('metadata', {})
        )


@dataclass
class PreferenceHint:
    """
    Learned preference hint.

    Attributes:
        category: Category of preference (style, format, domain)
        hint: The preference hint text
        strength: Strength of this preference (0.0 - 1.0)
        examples: Example scenarios where this applies
    """
    category: str
    hint: str
    strength: float
    examples: List[str] = field(default_factory=list)


# Law 3: OS-Agnostic Workspace
def get_preferences_path():
    """Get OS-agnostic path for preferences."""
    if hasattr(sys, 'getandroidapilevel') or os.path.exists('/data/data/com.termux'):
        base = "/sdcard/Download/SENSE_Data"
    elif os.name == 'nt':
        base = os.path.join(os.path.expanduser("~"), "Documents", "SENSE_Data")
    else:
        base = os.path.join(os.path.expanduser("~"), "SENSE_Data")
    return os.path.join(base, "preferences.json")


class PreferenceLearner:
    """
    Bayesian preference model that learns from user corrections.

    Stores preferences in OS-agnostic location (Law 3).
    """

    def __init__(self, decay_days: int = 30):
        """
        Initialize preference learner.

        Args:
            decay_days: Number of days after which preferences decay
        """
        self.decay_days = decay_days
        self.logger = logging.getLogger("Intelligence.Preferences")

        # Load preferences
        self.pref_path = get_preferences_path()
        self.preferences = self._load_preferences()
        self.feedback_history = self.preferences.get('feedback_history', [])

        # Ensure directory exists
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure preferences directory exists."""
        directory = os.path.dirname(self.pref_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Could not create preferences directory: {e}")

    def _load_preferences(self) -> Dict[str, Any]:
        """
        Load preferences from disk.

        Edge case handling: Corrupted file returns fresh state.
        """
        try:
            if os.path.exists(self.pref_path):
                with open(self.pref_path, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "version": 1,
                    "feedback_history": [],
                    "learned_preferences": {},
                    "last_updated": time.time()
                }
        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            self.logger.warning(f"Preferences corrupted/missing ({e}), starting fresh")
            return {
                "version": 1,
                "feedback_history": [],
                "learned_preferences": {},
                "last_updated": time.time()
            }

    def _persist(self):
        """Persist preferences to disk."""
        try:
            self.preferences['last_updated'] = time.time()
            with open(self.pref_path, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to persist preferences: {e}")

    def record_feedback(
        self,
        task: str,
        response: str,
        feedback_type: FeedbackType,
        correction: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record user feedback.

        Args:
            task: The task that was performed
            response: The response that was generated
            feedback_type: Type of feedback
            correction: Optional corrected response
            metadata: Additional feedback metadata
        """
        if metadata is None:
            metadata = {}

        entry = FeedbackEntry(
            task=task,
            response=response,
            feedback_type=feedback_type.value,
            correction=correction,
            metadata=metadata
        )

        self.feedback_history.append(entry.to_dict())
        self.preferences['feedback_history'] = self.feedback_history

        # Learn from feedback
        self._update_preferences(entry)

        # Persist
        self._persist()

        self.logger.info(
            f"Recorded {feedback_type.value} feedback for task: {task[:50]}..."
        )

    def _update_preferences(self, entry: FeedbackEntry):
        """
        Update learned preferences based on feedback.

        Uses simple Bayesian updating.
        """
        learned = self.preferences.get('learned_preferences', {})

        # Extract preference signals
        if entry.feedback_type == FeedbackType.POSITIVE.value:
            # Reinforce response style
            self._reinforce_style(entry.response, learned, weight=0.1)

        elif entry.feedback_type == FeedbackType.NEGATIVE.value:
            # Penalize response style
            self._reinforce_style(entry.response, learned, weight=-0.1)

        elif entry.feedback_type == FeedbackType.CORRECTION.value and entry.correction:
            # Learn from correction
            self._learn_correction(entry.response, entry.correction, learned)

        self.preferences['learned_preferences'] = learned

    def _reinforce_style(self, response: str, learned: Dict, weight: float):
        """
        Reinforce (or penalize) response style characteristics.
        """
        # Analyze response characteristics
        word_count = len(response.split())
        has_code = '```' in response
        has_bullet_points = '\n-' in response or '\n*' in response

        # Update style preferences
        if 'response_length' not in learned:
            learned['response_length'] = {'preferred_range': [50, 200], 'strength': 0.5}

        # Adjust preferred length based on feedback
        if weight > 0:
            # Positive feedback: move preferred range toward this length
            current_range = learned['response_length']['preferred_range']
            new_range = [
                int(current_range[0] * 0.9 + word_count * 0.1),
                int(current_range[1] * 0.9 + word_count * 0.1)
            ]
            learned['response_length']['preferred_range'] = new_range
            learned['response_length']['strength'] = min(1.0, learned['response_length']['strength'] + abs(weight))

        # Code preference
        if 'include_code' not in learned:
            learned['include_code'] = {'preferred': False, 'strength': 0.5}

        if has_code:
            if weight > 0:
                learned['include_code']['preferred'] = True
                learned['include_code']['strength'] = min(1.0, learned['include_code']['strength'] + abs(weight))
            else:
                learned['include_code']['preferred'] = False
                learned['include_code']['strength'] = max(0.0, learned['include_code']['strength'] - abs(weight))

        # Bullet point preference
        if 'use_bullet_points' not in learned:
            learned['use_bullet_points'] = {'preferred': False, 'strength': 0.5}

        if has_bullet_points:
            if weight > 0:
                learned['use_bullet_points']['preferred'] = True
                learned['use_bullet_points']['strength'] = min(1.0, learned['use_bullet_points']['strength'] + abs(weight))

    def _learn_correction(self, original: str, correction: str, learned: Dict):
        """
        Learn from user corrections.

        Stores correction patterns.
        """
        if 'corrections' not in learned:
            learned['corrections'] = []

        learned['corrections'].append({
            'original': original[:100],
            'correction': correction[:100],
            'timestamp': time.time()
        })

        # Keep only recent corrections (last 10)
        learned['corrections'] = learned['corrections'][-10:]

    def get_preference_hints(self, task: str) -> List[str]:
        """
        Get learned preferences relevant to task.

        Returns:
            List of preference hint strings
        """
        learned = self.preferences.get('learned_preferences', {})
        hints = []

        # Apply decay to old preferences
        self._apply_decay(learned)

        # Response length preference
        if 'response_length' in learned:
            pref = learned['response_length']
            if pref['strength'] > 0.5:
                range_str = f"{pref['preferred_range'][0]}-{pref['preferred_range'][1]}"
                hints.append(f"User prefers responses around {range_str} words")

        # Code inclusion preference
        if 'include_code' in learned:
            pref = learned['include_code']
            if pref['strength'] > 0.6:
                if pref['preferred']:
                    hints.append("User appreciates code examples")
                else:
                    hints.append("User prefers explanations without excessive code")

        # Bullet points preference
        if 'use_bullet_points' in learned:
            pref = learned['use_bullet_points']
            if pref['strength'] > 0.6 and pref['preferred']:
                hints.append("User likes bullet-point formatting")

        return hints

    def _apply_decay(self, learned: Dict):
        """
        Apply time-based decay to preferences.

        Old preferences become weaker over time.
        """
        last_updated = self.preferences.get('last_updated', time.time())
        days_since_update = (time.time() - last_updated) / (24 * 3600)

        if days_since_update > self.decay_days:
            # Apply decay
            decay_factor = 0.9 ** (days_since_update / self.decay_days)

            for pref_key in learned:
                if isinstance(learned[pref_key], dict) and 'strength' in learned[pref_key]:
                    learned[pref_key]['strength'] *= decay_factor

    def apply_preferences(self, prompt: str, task: str) -> str:
        """
        Inject preference hints into system prompt.

        Args:
            prompt: Original system prompt
            task: The task being performed

        Returns:
            Enhanced prompt with preferences
        """
        hints = self.get_preference_hints(task)

        if not hints:
            return prompt

        # Inject hints at end of prompt
        hint_section = "\n\n### USER PREFERENCES (Learned):\n"
        for hint in hints:
            hint_section += f"- {hint}\n"

        return prompt + hint_section

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about learned preferences.

        Returns:
            Dictionary with stats
        """
        feedback_counts = {
            FeedbackType.POSITIVE.value: 0,
            FeedbackType.NEGATIVE.value: 0,
            FeedbackType.CORRECTION.value: 0,
        }

        for entry in self.feedback_history:
            ftype = entry['feedback_type']
            if ftype in feedback_counts:
                feedback_counts[ftype] += 1

        learned = self.preferences.get('learned_preferences', {})

        return {
            'total_feedback': len(self.feedback_history),
            'positive': feedback_counts[FeedbackType.POSITIVE.value],
            'negative': feedback_counts[FeedbackType.NEGATIVE.value],
            'corrections': feedback_counts[FeedbackType.CORRECTION.value],
            'learned_preferences_count': len(learned),
            'last_updated': datetime.fromtimestamp(
                self.preferences.get('last_updated', time.time())
            ).isoformat()
        }
