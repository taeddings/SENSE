"""
SENSE-v2 Plugin Interface
Abstract base class for hardware/sensor plugins in the physical grounding system.

Part of Sprint 1: The Core

Plugins provide:
- Sensor data streams for physical grounding
- Ground truth verification for hallucination detection
- Safety policies for actuator control
- Emergency stop capabilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from enum import Enum
from datetime import datetime
import asyncio
import logging


class PluginCapability(Enum):
    """Capabilities a plugin can provide."""
    READ_ONLY = "read_only"       # Can only read sensor data
    ACTUATOR = "actuator"         # Can control physical systems
    MIXED = "mixed"               # Both read and actuate
    VIRTUAL = "virtual"           # No physical hardware (simulation)


@dataclass
class PluginManifest:
    """
    Manifest describing plugin capabilities and constraints.

    Used by the system to understand what a plugin can do
    and what safety constraints apply.
    """
    name: str
    version: str
    capability: PluginCapability
    sensors: List[str] = field(default_factory=list)
    actuators: List[str] = field(default_factory=list)
    safety_critical: bool = False
    requires_calibration: bool = False
    update_rate_hz: float = 10.0
    description: str = ""
    author: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "capability": self.capability.value,
            "sensors": self.sensors,
            "actuators": self.actuators,
            "safety_critical": self.safety_critical,
            "requires_calibration": self.requires_calibration,
            "update_rate_hz": self.update_rate_hz,
            "description": self.description,
            "author": self.author,
        }


@dataclass
class SensorReading:
    """A single sensor reading with metadata."""
    sensor_name: str
    value: Union[float, int, str, List[float]]
    timestamp: datetime = field(default_factory=datetime.now)
    unit: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_name": self.sensor_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class SafetyConstraint:
    """A safety constraint for plugin operations."""
    name: str
    parameter: str
    operator: str  # "<", ">", "<=", ">=", "==", "!="
    threshold: Union[float, int, str]
    action: str = "halt"  # "halt", "warn", "log"
    description: str = ""

    def check(self, value: Union[float, int, str]) -> bool:
        """
        Check if value satisfies the constraint.

        Returns:
            True if constraint is satisfied, False if violated
        """
        operators = {
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        op_func = operators.get(self.operator)
        if op_func is None:
            return True
        try:
            return op_func(value, self.threshold)
        except TypeError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameter": self.parameter,
            "operator": self.operator,
            "threshold": self.threshold,
            "action": self.action,
            "description": self.description,
        }


class PluginABC(ABC):
    """
    Abstract Base Class for SENSE plugins.

    Plugins connect external hardware, sensors, or simulated environments
    to the SENSE evolutionary system. They provide:

    1. **Sensor Streams**: Continuous data for physical grounding
    2. **Ground Truth**: Verification data for hallucination detection
    3. **Safety Policies**: Constraints on actuator operations
    4. **Emergency Stop**: Critical safety interrupt capability

    Example Implementation:
        class TemperatureSensorPlugin(PluginABC):
            def get_manifest(self) -> Dict[str, Any]:
                return {"type": "read_only", "sensors": ["temperature"]}

            async def stream_auxiliary_input(self) -> AsyncIterator[SensorReading]:
                while True:
                    yield SensorReading("temperature", self.read_temp(), unit="C")
                    await asyncio.sleep(0.1)

            def emergency_stop(self) -> None:
                pass  # No actuators to stop

            def safety_policy(self) -> Dict[str, Any]:
                return {"max_temperature": 100.0}

            def get_grounding_truth(self, query: str) -> Optional[float]:
                if "temperature" in query.lower():
                    return self.read_temp()
                return None
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_running = False
        self._is_calibrated = False
        self._last_reading: Optional[SensorReading] = None
        self._error_count = 0
        self._started_at: Optional[datetime] = None

    @abstractmethod
    def get_manifest(self) -> Dict[str, Any]:
        """
        Return plugin capabilities and metadata.

        The manifest describes what the plugin can do:
        - "type": "read_only" | "actuator" | "mixed" | "virtual"
        - "sensors": List of sensor names
        - "actuators": List of actuator names
        - "safety_critical": Whether plugin controls safety-critical systems

        Returns:
            Dictionary with plugin manifest
        """
        pass

    @abstractmethod
    async def stream_auxiliary_input(self) -> AsyncIterator[Union[SensorReading, float, int]]:
        """
        Generator yielding sensor data into the Engram pipeline.

        This is the primary data stream from the plugin. Data flows into
        the EngramIngestionPipeline for compression and storage.

        Yields:
            SensorReading objects or raw numeric values

        Example:
            async def stream_auxiliary_input(self):
                while self._is_running:
                    temp = await self.read_temperature()
                    yield SensorReading("temperature", temp, unit="C")
                    await asyncio.sleep(0.1)  # 10 Hz
        """
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """
        Critical Safety: Trigger immediate hardware interrupt.

        This method MUST be synchronous and complete as fast as possible.
        It should:
        1. Halt all actuators immediately
        2. Put system in safe state
        3. Log the emergency stop event

        Called when:
        - Safety constraint violated
        - System error detected
        - User triggers emergency stop
        - Hallucination detected with high confidence
        """
        pass

    @abstractmethod
    def safety_policy(self) -> Dict[str, Any]:
        """
        Return constraints for safe operation.

        The safety policy defines operational limits that must not
        be exceeded. The system will call emergency_stop() if any
        constraint is violated.

        Returns:
            Dictionary mapping constraint names to limits
            Example: {
                "max_speed": 50.0,
                "min_temperature": -40.0,
                "max_temperature": 85.0,
                "max_acceleration": 10.0,
            }
        """
        pass

    @abstractmethod
    def get_grounding_truth(self, query: str) -> Optional[float]:
        """
        Return ground truth for verification.

        Used by the grounding system to verify agent claims against
        physical reality. If the agent claims "temperature is 25C",
        this method can return the actual sensor reading.

        Args:
            query: Natural language query about physical state

        Returns:
            Ground truth value if available, None otherwise
        """
        pass

    # --------------------------------------------------------------------------
    # Optional methods with default implementations
    # --------------------------------------------------------------------------

    async def initialize(self) -> bool:
        """
        Initialize the plugin hardware/connection.

        Override to perform hardware initialization, connection setup, etc.

        Returns:
            True if initialization successful
        """
        self._started_at = datetime.now()
        self._is_running = True
        self.logger.info(f"Plugin {self.__class__.__name__} initialized")
        return True

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the plugin.

        Override to perform cleanup, close connections, etc.
        """
        self._is_running = False
        self.logger.info(f"Plugin {self.__class__.__name__} shutdown")

    async def calibrate(self) -> bool:
        """
        Perform sensor calibration.

        Override if plugin requires calibration before use.

        Returns:
            True if calibration successful
        """
        self._is_calibrated = True
        return True

    def get_safety_constraints(self) -> List[SafetyConstraint]:
        """
        Get structured safety constraints.

        Converts safety_policy() output to SafetyConstraint objects.

        Returns:
            List of SafetyConstraint objects
        """
        policy = self.safety_policy()
        constraints = []

        for key, value in policy.items():
            if key.startswith("max_"):
                param = key[4:]
                constraints.append(SafetyConstraint(
                    name=key,
                    parameter=param,
                    operator="<=",
                    threshold=value,
                    action="halt",
                    description=f"Maximum {param} constraint",
                ))
            elif key.startswith("min_"):
                param = key[4:]
                constraints.append(SafetyConstraint(
                    name=key,
                    parameter=param,
                    operator=">=",
                    threshold=value,
                    action="halt",
                    description=f"Minimum {param} constraint",
                ))

        return constraints

    def check_safety(self, readings: Dict[str, Union[float, int]]) -> List[SafetyConstraint]:
        """
        Check readings against safety constraints.

        Args:
            readings: Dictionary of parameter names to values

        Returns:
            List of violated constraints (empty if all OK)
        """
        constraints = self.get_safety_constraints()
        violations = []

        for constraint in constraints:
            if constraint.parameter in readings:
                value = readings[constraint.parameter]
                if not constraint.check(value):
                    violations.append(constraint)
                    self.logger.warning(
                        f"Safety constraint violated: {constraint.name} "
                        f"({value} {constraint.operator} {constraint.threshold})"
                    )

        return violations

    def verify_claim(self, claim: str, claimed_value: float) -> Tuple[bool, float]:
        """
        Verify an agent's claim against ground truth.

        Args:
            claim: The claim being verified (e.g., "temperature is 25")
            claimed_value: The numeric value claimed

        Returns:
            Tuple of (is_valid, grounding_score)
            - is_valid: True if claim matches reality within tolerance
            - grounding_score: 0.0 to 1.0 indicating alignment
        """
        ground_truth = self.get_grounding_truth(claim)

        if ground_truth is None:
            # Cannot verify - no ground truth available
            return True, 0.5

        # Calculate grounding score
        if abs(ground_truth) < 1e-8:
            # Avoid division by zero
            if abs(claimed_value) < 1e-8:
                return True, 1.0
            else:
                return False, 0.0

        error = abs(claimed_value - ground_truth) / max(abs(ground_truth), abs(claimed_value))
        grounding_score = max(0.0, 1.0 - error)

        # Threshold for validity (configurable)
        threshold = self.config.get("grounding_threshold", 0.5)
        is_valid = grounding_score >= threshold

        return is_valid, grounding_score

    @property
    def status(self) -> Dict[str, Any]:
        """Get plugin status."""
        manifest = self.get_manifest()
        return {
            "name": manifest.get("name", self.__class__.__name__),
            "is_running": self._is_running,
            "is_calibrated": self._is_calibrated,
            "error_count": self._error_count,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "last_reading": self._last_reading.to_dict() if self._last_reading else None,
        }


# Type alias for grounding score tuple
GroundingResult = Tuple[bool, float]
