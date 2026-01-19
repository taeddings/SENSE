import pytest
from dataclasses import is_dataclass, field
from typing import List
from sense_v2.core.config import (
    Config, HardwareConfig, EvolutionConfig, OrchestrationConfig,
    MemoryConfig, EngramConfig, ReasoningConfig, GroundingConfig
)

# Test EngramConfig dataclass
def test_engram_config_dataclass():
    assert is_dataclass(EngramConfig)

def test_engram_config_defaults():
    config = EngramConfig()
    assert config.enabled is True
    assert config.storage_path == "data/engram_table.dat"
    assert config.shadow_map_path == "data/shadow_map.npy"
    assert config.table_size == 10_000_000
    assert config.engram_dim == 1024
    assert config.num_heads == 8
    assert config.ngram_orders == [2, 3]
    assert config.layer_indices == [2, 15]

def test_engram_config_custom_values():
    config = EngramConfig(
        enabled=False,
        storage_path="/tmp/test.dat",
        shadow_map_path="/tmp/test.npy",
        table_size=1_000_000,
        engram_dim=512,
        num_heads=4,
        ngram_orders=[1, 2],
        layer_indices=[1, 10]
    )
    assert config.enabled is False
    assert config.storage_path == "/tmp/test.dat"
    assert config.shadow_map_path == "/tmp/test.npy"
    assert config.table_size == 1_000_000
    assert config.engram_dim == 512
    assert config.num_heads == 4
    assert config.ngram_orders == [1, 2]
    assert config.layer_indices == [1, 10]

# Test integration into main Config class
def test_main_config_has_engram_config():
    main_config = Config()
    assert hasattr(main_config, 'engram')
    assert isinstance(main_config.engram, EngramConfig)

def test_main_config_engram_defaults():
    main_config = Config()
    engram_config = main_config.engram
    assert engram_config.enabled is True
    assert engram_config.storage_path == "data/engram_table.dat"

def test_config_from_file_with_engram(tmp_path):
    config_data = {
        "hardware": {},
        "evolution": {},
        "orchestration": {},
        "memory": {},
        "engram": {
            "enabled": False,
            "table_size": 5_000_000
        },
        "log_level": "DEBUG"
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        import json
        json.dump(config_data, f)

    loaded_config = Config.from_file(str(config_path))
    assert loaded_config.engram.enabled is False
    assert loaded_config.engram.table_size == 5_000_000
    assert loaded_config.engram.engram_dim == 1024 # Default should still apply
    assert loaded_config.log_level == "DEBUG"

def test_config_to_dict_with_engram():
    main_config = Config()
    main_config.engram.enabled = False
    main_config.engram.table_size = 2_000_000

    config_dict = main_config.to_dict()
    assert "engram" in config_dict
    assert config_dict["engram"]["enabled"] is False
    assert config_dict["engram"]["table_size"] == 2_000_000
    assert config_dict["engram"]["engram_dim"] == 1024


# Test ReasoningConfig dataclass
def test_reasoning_config_dataclass():
    assert is_dataclass(ReasoningConfig)

def test_reasoning_config_defaults():
    config = ReasoningConfig()
    assert config.base_reasoning_budget == 1024
    assert config.max_reasoning_budget == 8192
    assert config.min_reasoning_budget == 256
    assert config.default_mode == "BALANCED"
    assert config.compensatory_multiplier == 2.0
    assert config.vram_monitoring_enabled is True
    assert config.memory_similarity_boost == 0.7

def test_reasoning_config_custom_values():
    config = ReasoningConfig(
        base_reasoning_budget=512,
        max_reasoning_budget=4096,
        default_mode="EFFICIENT",
        compensatory_multiplier=1.5,
        vram_monitoring_enabled=False
    )
    assert config.base_reasoning_budget == 512
    assert config.max_reasoning_budget == 4096
    assert config.default_mode == "EFFICIENT"
    assert config.compensatory_multiplier == 1.5
    assert config.vram_monitoring_enabled is False


# Test GroundingConfig dataclass
def test_grounding_config_dataclass():
    assert is_dataclass(GroundingConfig)

def test_grounding_config_defaults():
    config = GroundingConfig()
    assert config.grounding_enabled is True
    assert config.grounding_score_threshold == 0.5
    assert config.hallucination_alert_threshold == 0.3
    assert config.sensor_timeout_seconds == 5.0
    assert config.max_concurrent_plugins == 10
    assert config.trusted_sensor_types == ["temperature", "humidity", "pressure", "light", "motion"]

def test_grounding_config_custom_values():
    config = GroundingConfig(
        grounding_enabled=False,
        grounding_score_threshold=0.7,
        sensor_timeout_seconds=10.0,
        trusted_sensor_types=["temperature", "custom_sensor"]
    )
    assert config.grounding_enabled is False
    assert config.grounding_score_threshold == 0.7
    assert config.sensor_timeout_seconds == 10.0
    assert config.trusted_sensor_types == ["temperature", "custom_sensor"]

def test_grounding_config_is_sensor_trusted():
    config = GroundingConfig()
    assert config.is_sensor_trusted("temperature") is True
    assert config.is_sensor_trusted("unknown_sensor") is False

def test_grounding_config_should_trigger_emergency():
    config = GroundingConfig()
    assert config.should_trigger_emergency(0.2) is True  # Below threshold
    assert config.should_trigger_emergency(0.4) is False  # Above threshold
    assert config.should_trigger_emergency(0.3) is True  # At threshold

    # Test with emergency disabled
    config.emergency_stop_on_hallucination = False
    assert config.should_trigger_emergency(0.2) is False
