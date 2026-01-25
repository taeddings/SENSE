import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from sense.memory.agemem_integration import ReasoningMemoryManager, ReasoningMemoryConfig
from sense.memory.engram_schemas import ReasoningTrace, DriftSnapshot
from sense.memory.agemem import AgeMem, MemoryType
from sense.core.config import MemoryConfig


@pytest.fixture
def mock_agemem():
    """Create a mock AgeMem instance."""
    agemem = AsyncMock(spec=AgeMem)
    agemem.store_async = AsyncMock(return_value=True)
    agemem.search = AsyncMock(return_value=[])
    agemem.query_async = AsyncMock(return_value=[])
    return agemem


@pytest.fixture
def reasoning_config():
    """Create a ReasoningMemoryConfig instance."""
    return ReasoningMemoryConfig()


@pytest.fixture
def memory_manager(mock_agemem, reasoning_config):
    """Create a ReasoningMemoryManager instance."""
    return ReasoningMemoryManager(
        agemem=mock_agemem,
        config=reasoning_config
    )


@pytest.fixture
def sample_trace():
    """Create a sample ReasoningTrace."""
    return ReasoningTrace(
        trace_id="test_trace_123",
        task_id="test_task",
        genome_id="genome_001",
        generation_id=1,
        problem_description="Test problem",
        thought_chain=["First thought", "Second thought"],
        tool_calls=[{"tool_name": "terminal", "args": []}],
        outcome=True,
        final_answer="Test answer",
        grounding_score=0.8,
        reasoning_tokens_used=512,
        drift_context=DriftSnapshot(drift_level=0.2)
    )


class TestReasoningMemoryManager:
    """Test suite for ReasoningMemoryManager."""

    @pytest.mark.asyncio
    async def test_store_reasoning_trace_success(self, memory_manager, mock_agemem, sample_trace):
        """Test successful storage of reasoning trace."""
        # Setup
        mock_agemem.embedding_provider.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Execute
        result = await memory_manager.store_reasoning_trace(sample_trace)

        # Assert
        assert result is True
        mock_agemem.store_async.assert_called_once()
        assert sample_trace.trace_id in memory_manager._trace_cache
        assert memory_manager.traces_stored == 1

    @pytest.mark.asyncio
    async def test_store_reasoning_trace_with_embedding_generation(self, memory_manager, mock_agemem, sample_trace):
        """Test that embedding is generated when not present."""
        # Setup
        sample_trace.problem_embedding = None
        mock_agemem.embedding_provider.embed_text = AsyncMock(return_value=[0.2] * 384)

        # Execute
        await memory_manager.store_reasoning_trace(sample_trace)

        # Assert
        mock_agemem.embedding_provider.embed_text.assert_called_once()
        call_args = mock_agemem.store_async.call_args[1]['entry'].metadata
        assert 'problem_embedding' in call_args

    @pytest.mark.asyncio
    async def test_retrieve_similar_traces(self, memory_manager, mock_agemem, sample_trace):
        """Test retrieval of similar traces."""
        # Setup
        mock_search_results = [{
            'metadata': {'trace_id': sample_trace.trace_id},
            'similarity': 0.9
        }]
        mock_agemem.search.return_value = mock_search_results
        memory_manager._load_full_trace = AsyncMock(return_value=sample_trace)
        mock_agemem.embedding_provider.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Execute
        result = await memory_manager.retrieve_similar_traces("test problem")

        # Assert
        assert len(result) == 1
        assert result[0][0] == sample_trace
        assert result[0][1] == 0.9
        assert memory_manager.retrievals_performed == 1

    @pytest.mark.asyncio
    async def test_retrieve_similar_traces_no_matches(self, memory_manager, mock_agemem):
        """Test retrieval when no similar traces found."""
        # Setup
        mock_agemem.search.return_value = []
        mock_agemem.embedding_provider.embed_text = AsyncMock(return_value=[0.1] * 384)

        # Execute
        result = await memory_manager.retrieve_similar_traces("test problem")

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_get_traces_by_genome(self, memory_manager, mock_agemem, sample_trace):
        """Test retrieval of traces by genome ID."""
        # Setup
        mock_search_results = [{
            'metadata': {'genome_id': 'genome_001', 'trace_id': sample_trace.trace_id}
        }]
        mock_agemem.search.return_value = mock_search_results
        memory_manager._load_full_trace = AsyncMock(return_value=sample_trace)

        # Execute
        result = await memory_manager.get_traces_by_genome("genome_001")

        # Assert
        assert len(result) == 1
        assert result[0] == sample_trace

    @pytest.mark.asyncio
    async def test_get_successful_patterns(self, memory_manager, mock_agemem, sample_trace):
        """Test extraction of successful reasoning patterns."""
        # Setup
        mock_search_results = [{
            'metadata': {
                'outcome': True,
                'grounding_score': 0.9,
                'trace_id': sample_trace.trace_id
            }
        }]
        mock_agemem.search.return_value = mock_search_results
        memory_manager._load_full_trace = AsyncMock(return_value=sample_trace)
        memory_manager._extract_pattern_signature = MagicMock(return_value="pattern_1")

        # Execute
        result = await memory_manager.get_successful_patterns()

        # Assert
        assert "pattern_1" in result
        assert len(result["pattern_1"]) == 1

    def test_get_memory_stats(self, memory_manager):
        """Test memory statistics retrieval."""
        # Setup
        memory_manager.traces_stored = 5
        memory_manager.retrievals_performed = 10
        memory_manager.consolidations_done = 2

        # Execute
        stats = memory_manager.get_memory_stats()

        # Assert
        assert stats["traces_stored"] == 5
        assert stats["retrievals_performed"] == 10
        assert stats["consolidations_done"] == 2
        assert stats["cache_size"] == 0  # Empty cache
        assert stats["avg_retrieval_success"] == 2.0  # 10 / 5

    def test_calculate_trace_priority(self, memory_manager, sample_trace):
        """Test trace priority calculation."""
        # Test normal case
        priority = memory_manager._calculate_trace_priority(sample_trace)
        assert priority > 1.0  # Should be boosted for success and grounding

        # Test high drift case
        sample_trace.drift_context.drift_level = 0.6
        priority = memory_manager._calculate_trace_priority(sample_trace)
        assert priority > 1.5  # Should be boosted more for high drift

    def test_determine_trace_tier(self, memory_manager, sample_trace):
        """Test trace tier determination."""
        # Should be COLD for LTM-worthy traces
        tier = memory_manager._determine_trace_tier(sample_trace)
        assert tier == "COLD"

        # Should be HOT for low-quality traces
        sample_trace.outcome = False
        sample_trace.grounding_score = 0.2
        sample_trace.should_flag_for_ltm = MagicMock(return_value=False)
        tier = memory_manager._determine_trace_tier(sample_trace)
        assert tier == "HOT"

    def test_extract_pattern_signature(self, memory_manager, sample_trace):
        """Test pattern signature extraction."""
        signature = memory_manager._extract_pattern_signature(sample_trace)
        expected = "len_2_tools_terminal"
        assert signature == expected

    def test_load_full_trace_from_cache(self, memory_manager, sample_trace):
        """Test loading trace from cache."""
        # Setup
        memory_manager._trace_cache[sample_trace.trace_id] = sample_trace

        # Execute
        result = asyncio.run(memory_manager._load_full_trace(sample_trace.trace_id))

        # Assert
        assert result == sample_trace


class TestReasoningMemoryConfig:
    """Test suite for ReasoningMemoryConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ReasoningMemoryConfig()
        assert config.trace_embedding_dim == 384
        assert config.max_similar_traces == 5
        assert config.similarity_threshold == 0.7
        assert config.consolidation_batch_size == 10
        assert config.drift_memory_boost == 1.2