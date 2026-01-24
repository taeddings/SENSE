"""
SENSE-v2 Anomaly Detection Tool
Schema-based tool for detecting anomalies in data/behavior.
"""

from typing import Any, Dict, List, Optional
import logging
import numpy as np
from datetime import datetime
from collections import deque

from sense_v2.core.base import BaseTool, ToolRegistry
from sense_v2.core.schemas import ToolSchema, ToolParameter, ToolResult


@ToolRegistry.register
class AnomalyDetectionTool(BaseTool):
    """
    Tool for detecting anomalies in numerical data.

    Uses multiple detection methods:
    - Z-score based detection
    - IQR (Interquartile Range) method
    - Moving average deviation

    Optimized for streaming data.
    """

    def __init__(self, config: Optional[Any] = None, window_size: int = 100):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Sliding window for streaming detection
        self.window_size = window_size
        self._data_buffer: deque = deque(maxlen=window_size)

        # Statistics cache
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._last_update: Optional[datetime] = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="anomaly_detect",
            description="Detect anomalies in numerical data",
            parameters=[
                ToolParameter(
                    name="data",
                    param_type="array",
                    description="Array of numerical values to analyze",
                    required=True,
                ),
                ToolParameter(
                    name="method",
                    param_type="string",
                    description="Detection method: zscore, iqr, or combined",
                    required=False,
                    default="combined",
                    enum=["zscore", "iqr", "combined"],
                ),
                ToolParameter(
                    name="threshold",
                    param_type="float",
                    description="Sensitivity threshold (lower = more sensitive)",
                    required=False,
                    default=3.0,
                    min_value=1.0,
                    max_value=10.0,
                ),
                ToolParameter(
                    name="return_scores",
                    param_type="boolean",
                    description="Return anomaly scores for each point",
                    required=False,
                    default=False,
                ),
            ],
            returns="object",
            returns_description="Anomaly detection results",
            category="analysis",
            max_retries=0,
        )

    async def execute(
        self,
        data: List[float],
        method: str = "combined",
        threshold: float = 3.0,
        return_scores: bool = False,
        **kwargs
    ) -> ToolResult:
        """Detect anomalies in data."""
        try:
            if not data:
                return ToolResult.error("Empty data array")

            # Convert to numpy array
            arr = np.array(data, dtype=np.float64)

            if len(arr) < 3:
                return ToolResult.error("Need at least 3 data points")

            # Detect anomalies based on method
            if method == "zscore":
                result = self._detect_zscore(arr, threshold)
            elif method == "iqr":
                result = self._detect_iqr(arr, threshold)
            else:  # combined
                result = self._detect_combined(arr, threshold)

            # Build response
            response = {
                "anomaly_count": int(np.sum(result["is_anomaly"])),
                "anomaly_indices": [int(i) for i in np.where(result["is_anomaly"])[0]],
                "anomaly_values": [float(arr[i]) for i in np.where(result["is_anomaly"])[0]],
                "data_points": len(arr),
                "method": method,
                "threshold": threshold,
                "statistics": {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                },
            }

            if return_scores:
                response["scores"] = [float(s) for s in result["scores"]]

            # Update buffer for streaming
            self._update_buffer(arr)

            return ToolResult.success(response)

        except ValueError as e:
            return ToolResult.error(f"Invalid data: {e}")
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return ToolResult.error(str(e))

    def _detect_zscore(self, data: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Z-score based anomaly detection."""
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return {
                "is_anomaly": np.zeros(len(data), dtype=bool),
                "scores": np.zeros(len(data)),
            }

        z_scores = np.abs((data - mean) / std)

        return {
            "is_anomaly": z_scores > threshold,
            "scores": z_scores,
        }

    def _detect_iqr(self, data: np.ndarray, threshold: float) -> Dict[str, Any]:
        """IQR-based anomaly detection."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        if iqr == 0:
            return {
                "is_anomaly": np.zeros(len(data), dtype=bool),
                "scores": np.zeros(len(data)),
            }

        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)

        is_anomaly = (data < lower_bound) | (data > upper_bound)

        # Calculate scores as distance from bounds
        scores = np.zeros(len(data))
        below = data < lower_bound
        above = data > upper_bound
        scores[below] = (lower_bound - data[below]) / iqr
        scores[above] = (data[above] - upper_bound) / iqr

        return {
            "is_anomaly": is_anomaly,
            "scores": scores,
        }

    def _detect_combined(self, data: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Combined detection using multiple methods."""
        zscore_result = self._detect_zscore(data, threshold)
        iqr_result = self._detect_iqr(data, threshold * 0.5)  # Lower threshold for IQR

        # Combine: anomaly if detected by either method
        is_anomaly = zscore_result["is_anomaly"] | iqr_result["is_anomaly"]

        # Average scores
        scores = (zscore_result["scores"] + iqr_result["scores"]) / 2

        return {
            "is_anomaly": is_anomaly,
            "scores": scores,
        }

    def _update_buffer(self, data: np.ndarray) -> None:
        """Update streaming buffer with new data."""
        for val in data[-self.window_size:]:
            self._data_buffer.append(val)

        # Update cached statistics
        if len(self._data_buffer) > 0:
            buffer_arr = np.array(self._data_buffer)
            self._mean = float(np.mean(buffer_arr))
            self._std = float(np.std(buffer_arr))
            self._last_update = datetime.now()

    def detect_streaming(self, value: float, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect if a single streaming value is anomalous.
        Uses cached statistics from buffer.
        """
        if self._mean is None or self._std is None:
            # Not enough data yet
            self._data_buffer.append(value)
            return {"is_anomaly": False, "score": 0.0, "reason": "warming_up"}

        if self._std == 0:
            self._data_buffer.append(value)
            return {"is_anomaly": False, "score": 0.0, "reason": "zero_variance"}

        z_score = abs((value - self._mean) / self._std)
        is_anomaly = z_score > threshold

        # Update buffer
        self._data_buffer.append(value)

        return {
            "is_anomaly": is_anomaly,
            "score": float(z_score),
            "value": value,
            "mean": self._mean,
            "std": self._std,
        }

    def reset(self) -> None:
        """Reset the detection buffer."""
        self._data_buffer.clear()
        self._mean = None
        self._std = None
        self._last_update = None
