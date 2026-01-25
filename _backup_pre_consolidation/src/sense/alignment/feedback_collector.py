"""
Feedback Collection Module.

Interfaces for gathering human feedback via CLI, Streamlit, or API.
"""

from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime
import asyncio
import uuid
from .uncertainty_detector import UncertaintySignal

@dataclass
class HumanFeedback:
    """Human response to clarification request."""
    question: str
    selected_option: str
    custom_input: Optional[str]
    confidence: float  # User's confidence
    timestamp: datetime

class FeedbackCollector:
    """
    Collects human feedback through multiple interfaces.
    """

    def __init__(self, interface: str = "cli", config: Dict[str, Any] = None):
        self.interface = interface
        self.config = config or {}
        self.pending_requests = {} # id -> signal

    async def request_feedback(
        self,
        signal: UncertaintySignal,
        timeout: int = 300  # 5 minutes
    ) -> HumanFeedback:
        """
        Request human feedback for uncertainty.
        Blocks until feedback received or timeout.
        """
        if self.interface == "cli":
            return await self._cli_feedback(signal)
        # Add streamlit/api support here
        return await self._cli_feedback(signal)

    async def _cli_feedback(self, signal: UncertaintySignal) -> HumanFeedback:
        """CLI-based feedback collection."""
        print("\n" + "="*60)
        print("🤔 CLARIFICATION NEEDED")
        print("="*60)
        print(f"\n{signal.question}\n")

        for i, option in enumerate(signal.options, 1):
            print(f"{i}. {option['label']}")
            print(f"   {option['description']}\n")

        print(f"{len(signal.options) + 1}. Custom (type your own)")

        # Simple synchronous input for CLI (in async wrapper)
        # Note: In a real async loop this blocks the thread, typically
        # we'd use run_in_executor for input()
        loop = asyncio.get_event_loop()
        choice_idx = await loop.run_in_executor(None, self._get_cli_input, len(signal.options))
        
        if 0 <= choice_idx < len(signal.options):
            selected = signal.options[choice_idx]["label"]
            custom = None
        else:
            selected = "custom"
            print("Enter your response: ")
            custom = await loop.run_in_executor(None, input)

        return HumanFeedback(
            question=signal.question,
            selected_option=selected,
            custom_input=custom,
            confidence=1.0,
            timestamp=datetime.now()
        )

    def _get_cli_input(self, max_opt: int) -> int:
        while True:
            try:
                val = int(input("\nYour choice (number): ").strip()) - 1
                if 0 <= val <= max_opt:
                    return val
            except ValueError:
                pass
            print("Invalid choice.")
