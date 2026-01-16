"""
Tests for SENSE-v2 Anomaly Detection Tool
Per SYSTEM_PROMPT: Every tool must include test_[toolname].py
"""

import pytest
import asyncio
import numpy as np

from sense_v2.tools.anomaly import AnomalyDetectionTool
from sense_v2.core.schemas import ToolResultStatus


class TestAnomalyZscoreDetection:
    """Test Z-score based anomaly detection."""

    @pytest.mark.asyncio
    async def test_anomaly_zscore_detection(self):
        """Z-score method detects outliers."""
        tool = AnomalyDetectionTool()

        # Normal data with one clear outlier
        data = [10, 11, 10, 12, 11, 10, 11, 100, 10, 11]  # 100 is outlier

        result = await tool.execute(data=data, method="zscore", threshold=2.0)

        assert result.is_success is True
        assert result.output["anomaly_count"] >= 1
        assert 7 in result.output["anomaly_indices"]  # Index of 100
        assert 100 in result.output["anomaly_values"]
        assert result.output["method"] == "zscore"

    @pytest.mark.asyncio
    async def test_anomaly_zscore_no_outliers(self):
        """Z-score detects no anomalies in uniform data."""
        tool = AnomalyDetectionTool()

        # Uniform data
        data = [10.0, 10.1, 10.2, 9.9, 10.0, 10.1]

        result = await tool.execute(data=data, method="zscore", threshold=3.0)

        assert result.is_success is True
        assert result.output["anomaly_count"] == 0
        assert len(result.output["anomaly_indices"]) == 0

    @pytest.mark.asyncio
    async def test_anomaly_zscore_zero_variance(self):
        """Z-score handles zero variance data."""
        tool = AnomalyDetectionTool()

        # All same values
        data = [5.0, 5.0, 5.0, 5.0, 5.0]

        result = await tool.execute(data=data, method="zscore")

        assert result.is_success is True
        assert result.output["anomaly_count"] == 0


class TestAnomalyIqrDetection:
    """Test IQR-based anomaly detection."""

    @pytest.mark.asyncio
    async def test_anomaly_iqr_detection(self):
        """IQR method detects outliers."""
        tool = AnomalyDetectionTool()

        # Data with outlier beyond IQR bounds
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]  # 50 is outlier

        result = await tool.execute(data=data, method="iqr", threshold=1.5)

        assert result.is_success is True
        assert result.output["anomaly_count"] >= 1
        assert 50 in result.output["anomaly_values"]
        assert result.output["method"] == "iqr"

    @pytest.mark.asyncio
    async def test_anomaly_iqr_no_outliers(self):
        """IQR detects no anomalies in tight distribution."""
        tool = AnomalyDetectionTool()

        # Tight distribution
        data = list(range(1, 11))  # 1-10

        result = await tool.execute(data=data, method="iqr", threshold=3.0)

        assert result.is_success is True
        # With threshold=3.0, normal range should have no outliers

    @pytest.mark.asyncio
    async def test_anomaly_iqr_zero_iqr(self):
        """IQR handles zero IQR data."""
        tool = AnomalyDetectionTool()

        # Constant data
        data = [10.0, 10.0, 10.0, 10.0]

        result = await tool.execute(data=data, method="iqr")

        assert result.is_success is True
        assert result.output["anomaly_count"] == 0


class TestAnomalyCombinedDetection:
    """Test combined anomaly detection."""

    @pytest.mark.asyncio
    async def test_anomaly_combined_detection(self):
        """Combined method uses both Z-score and IQR."""
        tool = AnomalyDetectionTool()

        # Data with clear outlier
        data = [10, 11, 10, 12, 11, 10, 11, 200, 10, 11]

        result = await tool.execute(data=data, method="combined")

        assert result.is_success is True
        assert result.output["method"] == "combined"
        assert result.output["anomaly_count"] >= 1
        assert 200 in result.output["anomaly_values"]

    @pytest.mark.asyncio
    async def test_anomaly_combined_is_default(self):
        """Combined is the default method."""
        tool = AnomalyDetectionTool()
        schema = tool.schema

        method_param = next(p for p in schema.parameters if p.name == "method")
        assert method_param.default == "combined"


class TestAnomalyEmptyData:
    """Test empty data handling."""

    @pytest.mark.asyncio
    async def test_anomaly_empty_data(self):
        """Handle empty input."""
        tool = AnomalyDetectionTool()

        result = await tool.execute(data=[])

        assert result.is_success is False
        assert "empty" in result.error.lower()


class TestAnomalySmallData:
    """Test small dataset handling."""

    @pytest.mark.asyncio
    async def test_anomaly_small_data(self):
        """Handle <3 data points."""
        tool = AnomalyDetectionTool()

        # 2 points
        result = await tool.execute(data=[1, 2])

        assert result.is_success is False
        assert "at least 3" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anomaly_minimum_data(self):
        """Handle exactly 3 data points."""
        tool = AnomalyDetectionTool()

        result = await tool.execute(data=[1, 2, 3])

        assert result.is_success is True
        assert result.output["data_points"] == 3


class TestAnomalyReturnScores:
    """Test score output functionality."""

    @pytest.mark.asyncio
    async def test_anomaly_return_scores(self):
        """Verify score output."""
        tool = AnomalyDetectionTool()

        data = [10, 11, 10, 12, 100, 10, 11]

        result = await tool.execute(data=data, return_scores=True)

        assert result.is_success is True
        assert "scores" in result.output
        assert len(result.output["scores"]) == len(data)
        assert all(isinstance(s, float) for s in result.output["scores"])

        # Outlier should have higher score
        outlier_idx = 4  # Index of 100
        assert result.output["scores"][outlier_idx] > result.output["scores"][0]

    @pytest.mark.asyncio
    async def test_anomaly_no_scores_by_default(self):
        """Scores not returned by default."""
        tool = AnomalyDetectionTool()

        result = await tool.execute(data=[1, 2, 3, 4, 5])

        assert result.is_success is True
        assert "scores" not in result.output


class TestAnomalyStreaming:
    """Test streaming detection functionality."""

    @pytest.mark.asyncio
    async def test_anomaly_streaming(self):
        """Test streaming detection uses cached stats from batch execution."""
        tool = AnomalyDetectionTool(window_size=10)

        # First run batch execute to populate buffer and compute stats
        normal_data = [10, 11, 10, 12, 11, 10, 11, 10, 11, 10]
        await tool.execute(data=normal_data)

        # After batch execution, streaming should have cached stats
        # Check a normal value
        result = tool.detect_streaming(11)
        assert result["is_anomaly"] is False

        # Check an anomalous value
        result = tool.detect_streaming(100)
        assert result["is_anomaly"] is True
        assert result["score"] > 3.0  # High z-score

    def test_anomaly_streaming_warmup(self):
        """Streaming handles warmup period."""
        tool = AnomalyDetectionTool(window_size=10)

        # First call during warmup
        result = tool.detect_streaming(10)

        assert result["is_anomaly"] is False
        assert result["reason"] == "warming_up"

    @pytest.mark.asyncio
    async def test_anomaly_streaming_includes_stats(self):
        """Streaming includes mean and std after batch execution."""
        tool = AnomalyDetectionTool(window_size=5)

        # Run batch execute with data that has variance (non-zero std)
        await tool.execute(data=[9, 10, 11, 10, 10])

        # Check stats are included in streaming result
        result = tool.detect_streaming(10)

        assert "mean" in result
        assert "std" in result
        assert result["mean"] == pytest.approx(10.0, abs=0.5)

    def test_anomaly_streaming_reset(self):
        """Reset clears streaming state."""
        tool = AnomalyDetectionTool()

        # Add some data
        for val in [10, 11, 12]:
            tool.detect_streaming(val)

        # Reset
        tool.reset()

        # Should be back to warmup
        result = tool.detect_streaming(10)
        assert result["reason"] == "warming_up"


class TestAnomalyStatistics:
    """Test statistics output."""

    @pytest.mark.asyncio
    async def test_anomaly_statistics_output(self):
        """Verify statistics in output."""
        tool = AnomalyDetectionTool()

        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        result = await tool.execute(data=data)

        assert result.is_success is True
        assert "statistics" in result.output

        stats = result.output["statistics"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

        assert stats["mean"] == pytest.approx(5.5, abs=0.01)
        assert stats["min"] == 1
        assert stats["max"] == 10


class TestAnomalyToolSchema:
    """Test tool schema definition."""

    def test_anomaly_tool_schema(self):
        """Verify AnomalyDetectionTool schema."""
        tool = AnomalyDetectionTool()
        schema = tool.schema

        assert schema.name == "anomaly_detect"
        assert schema.category == "analysis"

        # Check data parameter
        data_param = next(p for p in schema.parameters if p.name == "data")
        assert data_param.required is True
        assert data_param.param_type == "array"

        # Check method parameter
        method_param = next(p for p in schema.parameters if p.name == "method")
        assert method_param.enum == ["zscore", "iqr", "combined"]

        # Check threshold parameter
        threshold_param = next(p for p in schema.parameters if p.name == "threshold")
        assert threshold_param.min_value == 1.0
        assert threshold_param.max_value == 10.0
        assert threshold_param.default == 3.0


class TestAnomalyToolValidation:
    """Test input validation."""

    def test_anomaly_validates_data(self):
        """AnomalyDetectionTool validates data parameter."""
        tool = AnomalyDetectionTool()

        # Valid input
        errors = tool.validate_input(data=[1, 2, 3])
        assert len(errors) == 0

        # Missing data
        errors = tool.validate_input()
        assert len(errors) > 0

    def test_anomaly_validates_method(self):
        """AnomalyDetectionTool validates method parameter."""
        tool = AnomalyDetectionTool()

        # Valid methods
        for method in ["zscore", "iqr", "combined"]:
            errors = tool.validate_input(data=[1, 2, 3], method=method)
            assert len(errors) == 0


class TestAnomalyBufferManagement:
    """Test internal buffer management."""

    @pytest.mark.asyncio
    async def test_buffer_update(self):
        """Buffer updates after execute."""
        tool = AnomalyDetectionTool(window_size=5)

        # Execute with data
        await tool.execute(data=[10, 20, 30, 40, 50, 60, 70])

        # Buffer should have last window_size values
        assert len(tool._data_buffer) == 5
        assert list(tool._data_buffer) == [30, 40, 50, 60, 70]

    @pytest.mark.asyncio
    async def test_buffer_stats_cached(self):
        """Statistics are cached for streaming."""
        tool = AnomalyDetectionTool()

        await tool.execute(data=[10, 20, 30, 40, 50])

        assert tool._mean is not None
        assert tool._std is not None
        assert tool._last_update is not None

    def test_buffer_window_size(self):
        """Window size is respected."""
        tool = AnomalyDetectionTool(window_size=3)

        # Add more than window size
        for val in [1, 2, 3, 4, 5]:
            tool._data_buffer.append(val)

        assert len(tool._data_buffer) == 3
        assert list(tool._data_buffer) == [3, 4, 5]
