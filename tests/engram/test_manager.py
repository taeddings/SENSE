import pytest
from pydantic import ValidationError
from sense_v2.core.config import Config
from sense_v2.engram.manager import EngramMemoryManager
# from sense_v2.engram.config import EngramConfig  # Legacy, skipped for Phase 2
EngramConfig = type('EngramConfig', (), {})()  # Stub

@pytest.fixture
def config():
    main_config = Config(engram=EngramConfig(
        memory_threshold=0.5,
        age_decay_rate=0.1,
        max_memories=1000,
        relevance_threshold=0.7
    ))
    return main_config

def test_engram_memory_manager_init(config):
    manager = EngramMemoryManager(config.engram)
    assert manager.config == config.engram
    assert len(manager.memories) == 0  # Assuming internal memories list

def test_store_memory(config):
    manager = EngramMemoryManager(config.engram)
    mock_memory = {'content': 'test', 'age': 0, 'relevance': 0.8, 'timestamp': 0}
    manager.store_memory(mock_memory)
    assert len(manager.memories) == 1
    # Test conditional: below threshold should not store
    low_relev = {'content': 'low', 'age': 0, 'relevance': 0.4, 'timestamp': 0}
    manager.store_memory(low_relev)
    assert len(manager.memories) == 1  # Not stored due to threshold

def test_retrieve_memories(config):
    manager = EngramMemoryManager(config.engram)
    # Assume some memories stored first
    memories = [{'content': 'relevant', 'age': 1, 'relevance': 0.9}, {'content': 'old', 'age': 100, 'relevance': 0.6}]
    for mem in memories:
        manager.store_memory(mem)
    retrieved = manager.retrieve_memories(query='relevant', max_results=5)
    assert len(retrieved) == 1
    assert 'relevant' in retrieved[0]['content']
    # Test AgeMem: old should be filtered if decayed below threshold

def test_prune_old_memories(config):
    manager = EngramMemoryManager(config.engram)
    old_mem = {'content': 'old', 'age': 200, 'relevance': 0.5, 'timestamp': -1000}
    new_mem = {'content': 'new', 'age': 0, 'relevance': 0.9, 'timestamp': 0}
    manager.store_memory(old_mem)
    manager.store_memory(new_mem)
    manager.prune_old_memories()
    assert len(manager.memories) == 1
    assert 'new' in manager.memories[0]['content']

def test_store_memory_validation_error(config):
    manager = EngramMemoryManager(config.engram)
    invalid_mem = {'content': 'test', 'age': -1}  # Missing relevance
    with pytest.raises(ValueError):  # Or custom error
        manager.store_memory(invalid_mem)