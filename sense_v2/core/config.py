"""
SENSE-v2 Configuration Module
Defines system-wide configuration optimized for 128GB UMA and AMD ROCm.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import os
import json


class HardwareBackend(Enum):
    """Supported hardware backends."""
    ROCM = "rocm"          # AMD RDNA 3.5
    CPU = "cpu"            # Fallback CPU
    AUTO = "auto"          # Auto-detect


class MemoryTier(Enum):
    """Memory tier classification."""
    HOT = "hot"            # Frequently accessed, in-memory
    WARM = "warm"          # Moderately accessed, cached
    COLD = "cold"          # Rarely accessed, vector DB


@dataclass
class HardwareConfig:
    """
    Hardware-specific configuration optimized for 128GB UMA with 256-bit bus.
    Prioritizes bandwidth efficiency over raw compute.
    """
    backend: HardwareBackend = HardwareBackend.AUTO
    unified_memory_gb: int = 128
    memory_bus_width: int = 256

    # vLLM specific settings for ROCm
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 32768
    vllm_tensor_parallel_size: int = 1

    # Batch sizing for bandwidth optimization
    optimal_batch_size: int = 64
    max_batch_size: int = 256

    # Memory thresholds
    memory_warning_threshold: float = 0.75
    memory_critical_threshold: float = 0.90

    def get_effective_vram(self) -> int:
        """Calculate effective VRAM available for model inference."""
        return int(self.unified_memory_gb * self.vllm_gpu_memory_utilization)

    def calculate_optimal_batch(self, sequence_length: int) -> int:
        """
        Calculate optimal batch size based on sequence length and bus width.
        Optimizes for bandwidth efficiency on 256-bit bus.
        """
        # Bytes per token (assuming fp16)
        bytes_per_token = 2
        # Optimal transfer size for 256-bit bus (32 bytes per transfer)
        optimal_transfer = 32

        # Calculate batch that aligns with bus width
        tokens_per_transfer = optimal_transfer // bytes_per_token
        ideal_batch = (sequence_length // tokens_per_transfer) * tokens_per_transfer

        return max(1, min(ideal_batch, self.max_batch_size))


@dataclass
class EvolutionConfig:
    """Configuration for Agent 0 (The School) evolutionary system."""
    population_size: int = 16
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7
    selection_top_k: int = 4
    diversity_weight: float = 0.1

    # GRPO (Group Relative Policy Optimization) settings
    grpo_group_size: int = 8
    grpo_temperature: float = 1.0
    grpo_kl_coeff: float = 0.1

    # Curriculum settings
    curriculum_stages: int = 5
    difficulty_ramp_rate: float = 0.1
    max_curriculum_steps: int = 1000

    # Reward function settings
    reward_binary: bool = True  # Binary vs scalar rewards
    unit_test_weight: float = 0.6
    exit_code_weight: float = 0.4


@dataclass
class OrchestrationConfig:
    """Configuration for Agent Zero (The Workplace) orchestration layer."""
    max_delegation_depth: int = 3
    master_context_limit: int = 4096  # Keep master agent context lean
    sub_agent_context_limit: int = 16384

    # Delegation settings
    task_timeout_seconds: int = 300
    max_retries: int = 3
    retry_backoff_base: float = 2.0

    # Sub-agent types
    enabled_sub_agents: List[str] = field(default_factory=lambda: [
        "terminal",
        "filesystem",
        "browser"
    ])


@dataclass
class MemoryConfig:
    """Configuration for AgeMem (The Filing Cabinet) memory system."""
    # Vector database settings
    vector_db_type: str = "chromadb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # LTM settings
    ltm_collection_name: str = "sense_ltm"
    ltm_max_entries: int = 100000
    ltm_similarity_threshold: float = 0.75

    # STM settings
    stm_max_tokens: int = 8192
    stm_prune_threshold: float = 0.80  # Trigger prune at 80% capacity
    stm_summary_ratio: float = 0.3     # Summarize to 30% of original

    # Memory tiering
    hot_tier_max_age_hours: int = 1
    warm_tier_max_age_hours: int = 24

    # Persistence paths
    persistence_dir: str = field(default_factory=lambda: os.path.expanduser("~/.sense_v2/memory"))


@dataclass
class Config:
    """
    Master configuration for SENSE-v2 framework.
    All inference and data-handling logic optimized for 128GB UMA.
    """
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Global settings
    log_level: str = "INFO"
    log_file: str = "sense_v2.log"
    dev_log_file: str = "dev_log.json"

    # Model settings (vLLM with ROCm)
    model_name: str = "meta-llama/Llama-2-7b-hf"
    model_dtype: str = "float16"

    # Self-correction settings
    max_self_correction_attempts: int = 5
    stderr_parse_enabled: bool = True

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Recursively construct Config from dictionary."""
        hardware = HardwareConfig(**data.get("hardware", {})) if "hardware" in data else HardwareConfig()
        evolution = EvolutionConfig(**data.get("evolution", {})) if "evolution" in data else EvolutionConfig()
        orchestration = OrchestrationConfig(**data.get("orchestration", {})) if "orchestration" in data else OrchestrationConfig()
        memory = MemoryConfig(**data.get("memory", {})) if "memory" in data else MemoryConfig()

        return cls(
            hardware=hardware,
            evolution=evolution,
            orchestration=orchestration,
            memory=memory,
            log_level=data.get("log_level", "INFO"),
            log_file=data.get("log_file", "sense_v2.log"),
            dev_log_file=data.get("dev_log_file", "dev_log.json"),
            model_name=data.get("model_name", "meta-llama/Llama-2-7b-hf"),
            model_dtype=data.get("model_dtype", "float16"),
            max_self_correction_attempts=data.get("max_self_correction_attempts", 5),
            stderr_parse_enabled=data.get("stderr_parse_enabled", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        # Handle enums during serialization
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        data = self.to_dict()

        def convert_enums(d):
            if isinstance(d, dict):
                return {k: convert_enums(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_enums(item) for item in d]
            elif isinstance(d, Enum):
                return d.value
            return d

        with open(path, "w") as f:
            json.dump(convert_enums(data), f, indent=2)

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []

        if self.hardware.unified_memory_gb < 16:
            issues.append("WARNING: Less than 16GB unified memory may cause OOM errors")

        if self.memory.stm_prune_threshold > 0.95:
            issues.append("WARNING: STM prune threshold too high, may cause context overflow")

        if self.evolution.population_size < self.evolution.selection_top_k:
            issues.append("ERROR: population_size must be >= selection_top_k")

        return issues
