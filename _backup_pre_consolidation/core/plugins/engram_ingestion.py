"""
SENSE-v2 Engram Ingestion Pipeline
Processes sensor data streams into compressed engram format for memory storage.

Part of Sprint 2: The Brain

Handles:
- Real-time sensor data ingestion from plugins
- Data compression and deduplication
- Engram formatting for AgeMem integration
- Quality validation and filtering
- Temporal aggregation for efficiency
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from datetime import datetime
import asyncio
import logging
import math

from .interface import (
    PluginABC,
    PluginCapability,
    PluginManifest,
    SensorReading,
    SafetyConstraint,
)
from sense_v2.memory.engram_schemas import DriftSnapshot


@dataclass
class EngramConfig:
    """
    Configuration for engram ingestion pipeline.

    Controls compression, filtering, and storage parameters.
    """
    # Compression settings
    compression_enabled: bool = True
    compression_threshold: float = 0.05  # Minimum change to store (5%)

    # Temporal aggregation
    aggregation_window_seconds: float = 1.0  # Aggregate readings over 1 second
    max_readings_per_window: int = 10

    # Quality filtering
    min_quality_score: float = 0.7  # Discard readings below this quality
    outlier_detection: bool = True
    outlier_threshold_sigma: float = 3.0

    # Storage settings
    max_engrams_per_second: int = 100  # Rate limiting
    engram_retention_days: int = 30

    # Drift sensitivity
    drift_adaptive_compression: bool = True


@dataclass
class EngramEntry:
    """
    Processed engram ready for memory storage.

    Represents compressed, validated sensor data with temporal context.
    """
    sensor_name: str
    timestamp: datetime
    value: float
    unit: str
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Aggregation data
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    reading_count: int = 1

    # Compression info
    compression_applied: bool = False
    previous_value: Optional[float] = None
    change_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize engram for storage."""
        return {
            "sensor_name": self.sensor_name,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "unit": self.unit,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "avg_value": self.avg_value,
            "reading_count": self.reading_count,
            "compression_applied": self.compression_applied,
            "previous_value": self.previous_value,
            "change_ratio": self.change_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngramEntry":
        """Deserialize engram from storage."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            sensor_name=data["sensor_name"],
            timestamp=timestamp,
            value=data["value"],
            unit=data["unit"],
            quality_score=data.get("quality_score", 1.0),
            metadata=data.get("metadata", {}),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            avg_value=data.get("avg_value"),
            reading_count=data.get("reading_count", 1),
            compression_applied=data.get("compression_applied", False),
            previous_value=data.get("previous_value"),
            change_ratio=data.get("change_ratio"),
        )


class EngramIngestionPipeline(PluginABC):
    """
    Engram Ingestion Pipeline - Processes sensor data into memory-ready engrams.

    This virtual plugin doesn't connect to physical hardware but processes
    data streams from other plugins into compressed, validated engrams
    suitable for long-term memory storage.

    Key Features:
    - Real-time data compression and deduplication
    - Quality validation and outlier detection
    - Temporal aggregation for efficiency
    - Drift-adaptive processing
    - Memory-optimized engram formatting

    Architecture:
    - Receives SensorReading objects from plugin streams
    - Applies compression algorithms to reduce storage overhead
    - Validates data quality and filters outliers
    - Aggregates readings over time windows
    - Formats as EngramEntry objects for AgeMem storage
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the engram ingestion pipeline.

        Args:
            config: Pipeline configuration
        """
        super().__init__(config)

        # Pipeline configuration
        self.pipeline_config = EngramConfig(
            compression_enabled=config.get("compression_enabled", True) if config else True,
            compression_threshold=config.get("compression_threshold", 0.05) if config else 0.05,
            aggregation_window_seconds=config.get("aggregation_window_seconds", 1.0) if config else 1.0,
            max_readings_per_window=config.get("max_readings_per_window", 10) if config else 10,
            min_quality_score=config.get("min_quality_score", 0.7) if config else 0.7,
            outlier_detection=config.get("outlier_detection", True) if config else True,
            outlier_threshold_sigma=config.get("outlier_threshold_sigma", 3.0) if config else 3.0,
            max_engrams_per_second=config.get("max_engrams_per_second", 100) if config else 100,
            engram_retention_days=config.get("engram_retention_days", 30) if config else 30,
            drift_adaptive_compression=config.get("drift_adaptive_compression", True) if config else True,
        )

        # Processing state
        self._last_values: Dict[str, float] = {}  # Last stored value per sensor
        self._window_buffers: Dict[str, List[SensorReading]] = {}  # Aggregation buffers
        self._window_start_times: Dict[str, datetime] = {}  # Window start times
        self._engram_queue: asyncio.Queue[EngramEntry] = asyncio.Queue()

        # Statistics
        self.engrams_processed = 0
        self.readings_compressed = 0
        self.outliers_filtered = 0
        self.quality_rejected = 0

        # Drift context
        self.current_drift_context: Optional[DriftSnapshot] = None

    def get_manifest(self) -> Dict[str, Any]:
        """
        Return plugin manifest for the ingestion pipeline.

        This is a virtual plugin that processes data internally.
        """
        return PluginManifest(
            name="EngramIngestionPipeline",
            version="2.0.0",
            capability=PluginCapability.VIRTUAL,
            sensors=[],  # Doesn't provide sensors, consumes them
            actuators=[],  # No actuators
            safety_critical=False,
            requires_calibration=False,
            update_rate_hz=0.0,  # Event-driven
            description="Processes sensor data into compressed engrams for memory storage",
        ).to_dict()

    async def stream_auxiliary_input(self) -> AsyncIterator[Union[SensorReading, float, int]]:
        """
        Yield processed engrams for consumption by memory systems.

        This stream provides the output of the ingestion pipeline -
        compressed, validated engrams ready for storage.

        Note: This plugin consumes sensor data rather than producing it.
        The engrams yielded here are the processed output.
        """
        while self._is_running:
            try:
                # Wait for engrams with timeout
                engram = await asyncio.wait_for(
                    self._engram_queue.get(),
                    timeout=1.0
                )

                # Yield the engram value (could be wrapped in SensorReading if needed)
                yield engram.value

            except asyncio.TimeoutError:
                # No engrams available, continue
                continue

    def emergency_stop(self) -> None:
        """Emergency stop - flush all buffers and stop processing."""
        self.logger.warning("EngramIngestionPipeline emergency stop triggered")
        self._is_running = False

        # Clear all buffers
        self._window_buffers.clear()
        self._window_start_times.clear()

        # Clear queue (best effort)
        while not self._engram_queue.empty():
            try:
                self._engram_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def safety_policy(self) -> Dict[str, Any]:
        """
        Safety policy for the ingestion pipeline.

        Focuses on data integrity and processing limits.
        """
        return {
            "max_queue_size": 1000,  # Prevent memory overflow
            "max_processing_rate": self.pipeline_config.max_engrams_per_second,
            "min_quality_threshold": self.pipeline_config.min_quality_score,
            "max_aggregation_window": 60.0,  # 1 minute max window
        }

    def get_grounding_truth(self, query: str) -> Optional[float]:
        """
        Return ground truth for engram-related queries.

        Not applicable for this virtual plugin.
        """
        return None

    # --------------------------------------------------------------------------
    # Pipeline-specific methods
    # --------------------------------------------------------------------------

    async def ingest_reading(self, reading: SensorReading) -> bool:
        """
        Ingest a sensor reading into the pipeline.

        Args:
            reading: SensorReading to process

        Returns:
            bool: True if reading was accepted for processing
        """
        try:
            # Quality validation
            if not self._validate_reading_quality(reading):
                self.quality_rejected += 1
                return False

            # Outlier detection
            if self.pipeline_config.outlier_detection and self._is_outlier(reading):
                self.outliers_filtered += 1
                return False

            # Add to aggregation window
            await self._add_to_aggregation_window(reading)

            return True

        except Exception as e:
            self.logger.error(f"Failed to ingest reading {reading}: {e}")
            return False

    async def process_pending_windows(self) -> int:
        """
        Process all pending aggregation windows.

        Returns:
            Number of engrams generated
        """
        engrams_created = 0

        for sensor_name in list(self._window_buffers.keys()):
            if self._should_process_window(sensor_name):
                engram = await self._process_aggregation_window(sensor_name)
                if engram:
                    await self._enqueue_engram(engram)
                    engrams_created += 1

        return engrams_created

    def update_drift_context(self, drift_snapshot: DriftSnapshot) -> None:
        """
        Update the pipeline with current drift context for adaptive processing.

        Args:
            drift_snapshot: Current drift metrics
        """
        self.current_drift_context = drift_snapshot

        # Adjust compression threshold based on drift
        if self.pipeline_config.drift_adaptive_compression:
            drift_level = drift_snapshot.drift_level
            # Increase compression during high drift (store more data)
            # Decrease compression during low drift (store less data)
            adjustment = 1.0 + (drift_level * 0.5)  # 0.5x to 1.5x
            self.pipeline_config.compression_threshold *= adjustment

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dict with pipeline statistics
        """
        return {
            "engrams_processed": self.engrams_processed,
            "readings_compressed": self.readings_compressed,
            "outliers_filtered": self.outliers_filtered,
            "quality_rejected": self.quality_rejected,
            "active_windows": len(self._window_buffers),
            "queue_size": self._engram_queue.qsize(),
            "compression_ratio": self._calculate_compression_ratio(),
        }

    # --------------------------------------------------------------------------
    # Internal processing methods
    # --------------------------------------------------------------------------

    def _validate_reading_quality(self, reading: SensorReading) -> bool:
        """Validate the quality of a sensor reading."""
        # Check if value is reasonable (not NaN, not infinite)
        if not isinstance(reading.value, (int, float)) or not math.isfinite(reading.value):
            return False

        # Check unit consistency (if we have previous readings)
        if reading.sensor_name in self._last_values:
            # For now, assume units are consistent
            pass

        # Quality score check (if available in metadata)
        quality = reading.metadata.get("quality_score", 1.0)
        return quality >= self.pipeline_config.min_quality_score

    def _is_outlier(self, reading: SensorReading) -> bool:
        """Detect if a reading is an outlier."""
        sensor_name = reading.sensor_name

        # Need at least some history for outlier detection
        if sensor_name not in self._last_values:
            return False

        # Simple statistical outlier detection
        # In a full implementation, this would use rolling statistics
        last_value = self._last_values[sensor_name]

        # Calculate z-score (simplified)
        if last_value == 0:
            return False

        change_ratio = abs(reading.value - last_value) / abs(last_value)
        threshold = self.pipeline_config.outlier_threshold_sigma * 0.1  # Simplified

        return change_ratio > threshold

    async def _add_to_aggregation_window(self, reading: SensorReading) -> None:
        """Add reading to appropriate aggregation window."""
        sensor_name = reading.sensor_name

        # Initialize window if needed
        if sensor_name not in self._window_buffers:
            self._window_buffers[sensor_name] = []
            self._window_start_times[sensor_name] = reading.timestamp

        buffer = self._window_buffers[sensor_name]

        # Add reading to buffer
        buffer.append(reading)

        # Check if window should be processed
        if self._should_process_window(sensor_name):
            engram = await self._process_aggregation_window(sensor_name)
            if engram:
                await self._enqueue_engram(engram)

    def _should_process_window(self, sensor_name: str) -> bool:
        """Check if aggregation window should be processed."""
        if sensor_name not in self._window_buffers:
            return False

        buffer = self._window_buffers[sensor_name]
        start_time = self._window_start_times[sensor_name]

        # Process if window is full or time limit reached
        time_elapsed = (datetime.now() - start_time).total_seconds()
        window_full = len(buffer) >= self.pipeline_config.max_readings_per_window
        time_up = time_elapsed >= self.pipeline_config.aggregation_window_seconds

        return window_full or time_up

    async def _process_aggregation_window(self, sensor_name: str) -> Optional[EngramEntry]:
        """Process an aggregation window into an engram."""
        buffer = self._window_buffers[sensor_name]
        if not buffer:
            return None

        # Calculate aggregated values
        values = [r.value for r in buffer]
        avg_value = sum(values) / len(values)
        min_value = min(values)
        max_value = max(values)

        # Use latest reading as representative
        latest_reading = buffer[-1]

        # Apply compression
        compression_applied = False
        change_ratio = None

        if self.pipeline_config.compression_enabled and sensor_name in self._last_values:
            last_stored = self._last_values[sensor_name]
            change_ratio = abs(avg_value - last_stored) / abs(last_stored) if last_stored != 0 else 0

            if change_ratio < self.pipeline_config.compression_threshold:
                # Change too small, don't create engram
                self.readings_compressed += len(buffer)
                self._clear_window(sensor_name)
                return None

            compression_applied = True

        # Create engram
        engram = EngramEntry(
            sensor_name=sensor_name,
            timestamp=latest_reading.timestamp,
            value=avg_value,
            unit=latest_reading.unit,
            quality_score=latest_reading.metadata.get("quality_score", 1.0),
            metadata={
                "source_plugin": latest_reading.metadata.get("source_plugin", "unknown"),
                "aggregation_method": "average",
                "original_readings": len(buffer),
            },
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            reading_count=len(buffer),
            compression_applied=compression_applied,
            previous_value=self._last_values.get(sensor_name),
            change_ratio=change_ratio,
        )

        # Update last values
        self._last_values[sensor_name] = avg_value

        # Clear window
        self._clear_window(sensor_name)

        self.engrams_processed += 1
        return engram

    def _clear_window(self, sensor_name: str) -> None:
        """Clear aggregation window for a sensor."""
        if sensor_name in self._window_buffers:
            del self._window_buffers[sensor_name]
        if sensor_name in self._window_start_times:
            del self._window_start_times[sensor_name]

    async def _enqueue_engram(self, engram: EngramEntry) -> None:
        """Add engram to processing queue."""
        try:
            await self._engram_queue.put(engram)
        except asyncio.QueueFull:
            self.logger.warning("Engram queue full, dropping engram")

    def _calculate_compression_ratio(self) -> float:
        """Calculate current compression ratio."""
        total_readings = (self.engrams_processed +
                         self.readings_compressed +
                         self.outliers_filtered +
                         self.quality_rejected)

        if total_readings == 0:
            return 1.0

        return self.engrams_processed / total_readings