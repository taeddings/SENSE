"""
Learning Trajectory Tracker.

Tracks performance over time to calculate the Learning Rate.
"""

from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrajectoryPoint:
    timestamp: datetime
    difficulty: float
    success: bool

class TrajectoryTracker:
    """
    Tracks learning progress and computes the 'meta-fitness'.
    """

    def __init__(self, config: Dict[str, Any]):
        self.history: List[TrajectoryPoint] = []

    def add_point(self, difficulty: float, success: bool):
        self.history.append(TrajectoryPoint(datetime.now(), difficulty, success))

    def compute_learning_rate(self, window: int = 20) -> float:
        """
        Compute the rate of improvement (Learning Rate).
        Positive = Agent is getting better.
        """
        if len(self.history) < 10:
            return 0.0

        points = self.history[-window:]
        half = len(points) // 2
        
        first_half = points[:half]
        second_half = points[half:]
        
        score1 = np.mean([p.difficulty * (1 if p.success else 0) for p in first_half])
        score2 = np.mean([p.difficulty * (1 if p.success else 0) for p in second_half])
        
        return score2 - score1
