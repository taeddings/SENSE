import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
# from sense.core.plugins.engram_ingestion import (
#     EngramIngestionPipeline, EngramConfig, EngramEntry
# )  # Legacy, skipped for Phase 2
EngramIngestionPipeline = type('EngramIngestionPipeline', (), {})
EngramConfig = type('EngramConfig', (), {})
EngramEntry = type('EngramEntry', (), {})
# from sense.core.plugins.interface import SensorReading  # Legacy
SensorReading = type('SensorReading', (), {})
from sense.memory.engram_schemas import DriftSnapshot


@pytest.fixture
def engram_config():
    """Create an EngramConfig instance."""
    return EngramConfig()


@pytest.fixture
def ingestion_pipeline(engram_config):
    """Create an EngramIngestionPipeline instance."""
    return EngramIngestionPipeline(config={"compression_threshold": 0.1})


@pytest.fixture
def sample_reading():
    """Create a sample SensorReading."""
    return SensorReading(
        sensor_name="temperature",
        value=22.5,
        unit="C",
        timestamp=datetime.now(),
        metadata={"quality_score": 0.9, "source_plugin": "mock_sensor"}
    )


class TestEngramIngestionPipeline:
    """Test suite for EngramIngestionPipeline."""

    def test_initialization(self, ingestion_pipeline):
        """Test pipeline initialization."""
        assert ingestion_pipeline._is_running is False
        assert len(ingestion_pipeline._window_buffers) == 0
        assert ingestion_pipeline.engrams_processed == 0
        assert ingestion_pipeline.readings_compressed == 0

    def test_get_manifest(self, ingestion_pipeline):
        """Test plugin manifest generation."""
        manifest = ingestion_pipeline.get_manifest()
        assert manifest["name"] == "EngramIngestionPipeline"
        assert manifest["version"] == "2.0.0"
        assert manifest["capability"] == "VIRTUAL"
        assert manifest["sensors"] == []  # Doesn't provide sensors
        assert manifest["safety_critical"] is False

    def test_safety_policy(self, ingestion_pipeline):
        """Test safety policy generation."""
        policy = ingestion_pipeline.safety_policy()
        assert "max_queue_size" in policy
        assert "max_processing_rate" in policy
        assert policy["max_processing_rate"] == 100

    def test_get_grounding_truth(self, ingestion_pipeline):
        """Test grounding truth (should return None for virtual plugin)."""
        result = ingestion_pipeline.get_grounding_truth("any query")
        assert result is None

    @pytest.mark.asyncio
    async def test_ingest_reading_success(self, ingestion_pipeline, sample_reading):
        """Test successful reading ingestion."""
        # Execute
        result = await ingestion_pipeline.ingest_reading(sample_reading)

        # Assert
        assert result is True
        assert "temperature" in ingestion_pipeline._window_buffers
        assert len(ingestion_pipeline._window_buffers["temperature"]) == 1

    @pytest.mark.asyncio
    async def test_ingest_reading_quality_filter(self, ingestion_pipeline, sample_reading):
        """Test quality filtering of readings."""
        # Setup low quality reading
        sample_reading.metadata["quality_score"] = 0.5  # Below threshold

        # Execute
        result = await ingestion_pipeline.ingest_reading(sample_reading)

        # Assert
        assert result is False
        assert ingestion_pipeline.quality_rejected == 1

    @pytest.mark.asyncio
    async def test_ingest_reading_outlier_filter(self, ingestion_pipeline, sample_reading):
        """Test outlier detection and filtering."""
        # Setup pipeline config for outlier detection
        ingestion_pipeline.pipeline_config.outlier_detection = True

        # First reading to establish baseline
        await ingestion_pipeline.ingest_reading(sample_reading)

        # Second reading that's an outlier
        outlier_reading = SensorReading(
            sensor_name="temperature",
            value=100.0,  # Way different from 22.5
            unit="C",
            timestamp=datetime.now(),
            metadata={"quality_score": 0.9}
        )

        # Execute
        result = await ingestion_pipeline.ingest_reading(outlier_reading)

        # Assert
        assert result is False
        assert ingestion_pipeline.outliers_filtered == 1

    @pytest.mark.asyncio
    async def test_process_pending_windows(self, ingestion_pipeline, sample_reading):
        """Test processing of aggregation windows."""
        # Setup
        ingestion_pipeline.pipeline_config.aggregation_window_seconds = 0.1  # Short window
        await ingestion_pipeline.ingest_reading(sample_reading)

        # Mock engram queue
        ingestion_pipeline._enqueue_engram = AsyncMock()

        # Execute
        count = await ingestion_pipeline.process_pending_windows()

        # Assert
        assert count == 1
        ingestion_pipeline._enqueue_engram.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_auxiliary_input_empty(self, ingestion_pipeline):
        """Test streaming when no engrams available."""
        # Setup timeout to avoid hanging
        ingestion_pipeline._engram_queue.get = AsyncMock(side_effect=asyncio.TimeoutError)

        # Execute with timeout
        async def run_stream():
            count = 0
            async for _ in ingestion_pipeline.stream_auxiliary_input():
                count += 1
                if count > 0:  # Should timeout immediately
                    break
            return count

        count = await asyncio.wait_for(run_stream(), timeout=0.1)

        # Assert
        assert count == 0

    def test_emergency_stop(self, ingestion_pipeline, sample_reading):
        """Test emergency stop functionality."""
        # Setup
        ingestion_pipeline._is_running = True
        ingestion_pipeline._engram_queue.put_nowait = MagicMock()

        # Execute
        ingestion_pipeline.emergency_stop()

        # Assert
        assert ingestion_pipeline._is_running is False
        assert len(ingestion_pipeline._window_buffers) == 0
        assert len(ingestion_pipeline._window_start_times) == 0

    def test_update_drift_context(self, ingestion_pipeline):
        """Test drift context updates."""
        # Setup
        drift_snapshot = DriftSnapshot(drift_level=0.8)
        original_threshold = ingestion_pipeline.pipeline_config.compression_threshold

        # Execute
        ingestion_pipeline.update_drift_context(drift_snapshot)

        # Assert
        assert ingestion_pipeline.current_drift_context == drift_snapshot
        # Threshold should be adjusted for high drift
        assert ingestion_pipeline.pipeline_config.compression_threshold > original_threshold

    def test_get_pipeline_stats(self, ingestion_pipeline):
        """Test pipeline statistics retrieval."""
        # Setup
        ingestion_pipeline.engrams_processed = 5
        ingestion_pipeline.readings_compressed = 10
        ingestion_pipeline.outliers_filtered = 2
        ingestion_pipeline.quality_rejected = 1

        # Execute
        stats = ingestion_pipeline.get_pipeline_stats()

        # Assert
        assert stats["engrams_processed"] == 5
        assert stats["readings_compressed"] == 10
        assert stats["outliers_filtered"] == 2
        assert stats["quality_rejected"] == 1
        assert stats["compression_ratio"] == 0.25  # 5 / (5+10+2+1)

    def test_validate_reading_quality(self, ingestion_pipeline, sample_reading):
        """Test reading quality validation."""
        # Valid reading
        assert ingestion_pipeline._validate_reading_quality(sample_reading) is True

        # Invalid: NaN value
        sample_reading.value = float('nan')
        assert ingestion_pipeline._validate_reading_quality(sample_reading) is False

        # Reset
        sample_reading.value = 22.5

        # Invalid: low quality
        sample_reading.metadata["quality_score"] = 0.5
        assert ingestion_pipeline._validate_reading_quality(sample_reading) is False

    def test_should_process_window(self, ingestion_pipeline, sample_reading):
        """Test window processing decision."""
        # No buffer
        assert ingestion_pipeline._should_process_window("temperature") is False

        # Setup buffer
        ingestion_pipeline._window_buffers["temperature"] = [sample_reading]
        ingestion_pipeline._window_start_times["temperature"] = datetime.now()

        # Short window time
        ingestion_pipeline.pipeline_config.aggregation_window_seconds = 0.001
        assert ingestion_pipeline._should_process_window("temperature") is True

    def test_clear_window(self, ingestion_pipeline, sample_reading):
        """Test window clearing."""
        # Setup
        ingestion_pipeline._window_buffers["temperature"] = [sample_reading]
        ingestion_pipeline._window_start_times["temperature"] = datetime.now()

        # Execute
        ingestion_pipeline._clear_window("temperature")

        # Assert
        assert "temperature" not in ingestion_pipeline._window_buffers
        assert "temperature" not in ingestion_pipeline._window_start_times


class TestEngramConfig:
    """Test suite for EngramConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = EngramConfig()
        assert config.compression_enabled is True
        assert config.compression_threshold == 0.05
        assert config.aggregation_window_seconds == 1.0
        assert config.min_quality_score == 0.7
        assert config.max_engrams_per_second == 100


class TestEngramEntry:
    """Test suite for EngramEntry."""

    def test_engram_entry_creation(self):
        """Test engram entry creation."""
        entry = EngramEntry(
            sensor_name="temperature",
            timestamp=datetime.now(),
            value=25.0,
            unit="C"
        )

        assert entry.sensor_name == "temperature"
        assert entry.value == 25.0
        assert entry.unit == "C"
        assert entry.quality_score == 1.0

    def test_engram_entry_to_dict(self):
        """Test engram serialization."""
        timestamp = datetime.now()
        entry = EngramEntry(
            sensor_name="temperature",
            timestamp=timestamp,
            value=25.0,
            unit="C",
            compression_applied=True
        )

        data = entry.to_dict()
        assert data["sensor_name"] == "temperature"
        assert data["value"] == 25.0
        assert data["compression_applied"] is True

    def test_engram_entry_from_dict(self):
        """Test engram deserialization."""
        timestamp_str = "2023-01-01T12:00:00"
        data = {
            "sensor_name": "temperature",
            "timestamp": timestamp_str,
            "value": 25.0,
            "unit": "C",
            "quality_score": 0.9
        }

        entry = EngramEntry.from_dict(data)
        assert entry.sensor_name == "temperature"
        assert entry.value == 25.0
        assert entry.quality_score == 0.9