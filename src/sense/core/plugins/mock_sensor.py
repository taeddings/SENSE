"""
SENSE-v2 Mock Sensor Plugin
Test implementation of PluginABC for development and testing.

Part of Sprint 1: The Core

Provides simulated sensor data with configurable noise levels
for testing the grounding system without real hardware.
"""

import asyncio
import random
import math
from typing import Any, Dict, Optional, Union, AsyncIterator
from datetime import datetime
from dataclasses import dataclass, field

from sense.core.plugins.interface import (
    PluginABC,
    PluginCapability,
    PluginManifest,
    SensorReading,
)


@dataclass
class MockSensorConfig:
    """Configuration for mock sensor behavior."""
    noise_level: float = 0.1  # Standard deviation as fraction of base value
    drift_rate: float = 0.001  # Rate of sensor drift per second
    failure_probability: float = 0.0  # Probability of sensor failure
    update_rate_hz: float = 100.0  # Sensor update rate
    initial_values: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 22.0,
        "humidity": 45.0,
        "pressure": 1013.25,
        "light": 500.0,
    })


class MockSensorPlugin(PluginABC):
    """
    Mock sensor plugin for testing the grounding system.

    Provides simulated readings for:
    - Temperature (Celsius)
    - Humidity (%)
    - Pressure (hPa)
    - Light (lux)

    Features:
    - Configurable noise levels
    - Simulated sensor drift
    - Simulated sensor failures
    - Sinusoidal patterns for testing adaptation

    Example:
        plugin = MockSensorPlugin(noise_level=0.05)
        await plugin.initialize()

        async for reading in plugin.stream_auxiliary_input():
            print(f"{reading.sensor_name}: {reading.value} {reading.unit}")
    """

    def __init__(
        self,
        noise_level: float = 0.1,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize mock sensor plugin.

        Args:
            noise_level: Standard deviation as fraction of base value
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.noise_level = noise_level

        # Parse config
        self._sensor_config = MockSensorConfig(
            noise_level=noise_level,
            drift_rate=config.get("drift_rate", 0.001) if config else 0.001,
            failure_probability=config.get("failure_probability", 0.0) if config else 0.0,
            update_rate_hz=config.get("update_rate_hz", 100.0) if config else 100.0,
            initial_values=config.get("initial_values", {
                "temperature": 22.0,
                "humidity": 45.0,
                "pressure": 1013.25,
                "light": 500.0,
            }) if config else {
                "temperature": 22.0,
                "humidity": 45.0,
                "pressure": 1013.25,
                "light": 500.0,
            },
        )

        # Current sensor values (will drift over time)
        self._base_values: Dict[str, float] = dict(self._sensor_config.initial_values)

        # Sensor units
        self._units: Dict[str, str] = {
            "temperature": "°C",
            "humidity": "%",
            "pressure": "hPa",
            "light": "lux",
        }

        # Drift state
        self._drift_offsets: Dict[str, float] = {k: 0.0 for k in self._base_values}
        self._start_time: Optional[float] = None

        # Manifest
        self._manifest = PluginManifest(
            name="MockSensor",
            version="1.0.0",
            capability=PluginCapability.READ_ONLY,
            sensors=list(self._base_values.keys()),
            actuators=[],
            safety_critical=False,
            requires_calibration=False,
            update_rate_hz=self._sensor_config.update_rate_hz,
            description="Mock sensor plugin for testing",
            author="SENSE Team",
        )

    def get_manifest(self) -> Dict[str, Any]:
        """Return plugin capabilities."""
        return self._manifest.to_dict()

    async def stream_auxiliary_input(self) -> AsyncIterator[SensorReading]:
        """
        Yield simulated sensor readings.

        Generates readings for all sensors at the configured rate.
        Includes noise and optional drift.

        Yields:
            SensorReading objects with simulated values
        """
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()

        interval = 1.0 / self._sensor_config.update_rate_hz

        while self._is_running:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self._start_time

            for sensor_name, base_value in self._base_values.items():
                # Check for simulated failure
                if random.random() < self._sensor_config.failure_probability:
                    self._error_count += 1
                    continue

                # Calculate drift
                drift = self._drift_offsets[sensor_name]
                drift += self._sensor_config.drift_rate * interval
                self._drift_offsets[sensor_name] = drift

                # Add sinusoidal variation (simulates daily/cyclic patterns)
                # Different frequencies for different sensors
                frequencies = {
                    "temperature": 0.1,  # Slow variation
                    "humidity": 0.05,
                    "pressure": 0.02,
                    "light": 0.5,  # Faster variation (clouds, etc.)
                }
                freq = frequencies.get(sensor_name, 0.1)
                sinusoidal = math.sin(2 * math.pi * freq * elapsed) * base_value * 0.1

                # Calculate noisy value
                noise = random.gauss(0, self.noise_level * base_value)
                value = base_value + drift + sinusoidal + noise

                # Clamp to reasonable ranges
                value = self._clamp_value(sensor_name, value)

                reading = SensorReading(
                    sensor_name=sensor_name,
                    value=round(value, 2),
                    unit=self._units.get(sensor_name, ""),
                    confidence=1.0 - abs(noise / base_value) if base_value != 0 else 1.0,
                    metadata={
                        "drift": drift,
                        "noise_level": self.noise_level,
                        "elapsed_seconds": elapsed,
                    },
                )

                self._last_reading = reading
                yield reading

            await asyncio.sleep(interval)

    def _clamp_value(self, sensor_name: str, value: float) -> float:
        """Clamp sensor value to realistic range."""
        ranges = {
            "temperature": (-40, 85),
            "humidity": (0, 100),
            "pressure": (900, 1100),
            "light": (0, 100000),
        }
        min_val, max_val = ranges.get(sensor_name, (-float('inf'), float('inf')))
        return max(min_val, min(max_val, value))

    def emergency_stop(self) -> None:
        """
        Emergency stop - no-op for read-only sensor.

        Mock sensor has no actuators to stop.
        """
        self.logger.warning("Emergency stop called on MockSensorPlugin (no action needed)")
        self._is_running = False

    def safety_policy(self) -> Dict[str, Any]:
        """
        Return safety constraints for mock sensor.

        Returns:
            Dictionary with safety limits
        """
        return {
            "max_temperature": 100.0,
            "min_temperature": -50.0,
            "max_humidity": 100.0,
            "min_humidity": 0.0,
            "max_pressure": 1100.0,
            "min_pressure": 800.0,
        }

    def get_grounding_truth(self, query: str) -> Optional[float]:
        """
        Return ground truth for verification.

        Searches the query for known sensor names and returns
        the current base value (without noise).

        Args:
            query: Natural language query

        Returns:
            Ground truth value if query matches a sensor
        """
        query_lower = query.lower()

        for sensor_name, base_value in self._base_values.items():
            if sensor_name in query_lower:
                # Return base value plus drift (but no noise)
                drift = self._drift_offsets.get(sensor_name, 0.0)
                return base_value + drift

        return None

    def set_base_value(self, sensor_name: str, value: float) -> bool:
        """
        Set the base value for a sensor.

        Useful for testing specific scenarios.

        Args:
            sensor_name: Name of sensor to update
            value: New base value

        Returns:
            True if sensor exists and was updated
        """
        if sensor_name in self._base_values:
            self._base_values[sensor_name] = value
            return True
        return False

    def inject_anomaly(self, sensor_name: str, magnitude: float = 2.0) -> bool:
        """
        Inject an anomaly into sensor readings.

        Useful for testing anomaly detection.

        Args:
            sensor_name: Sensor to inject anomaly into
            magnitude: Multiplier for the anomaly (2.0 = double the base value)

        Returns:
            True if anomaly was injected
        """
        if sensor_name in self._base_values:
            # Temporarily modify base value
            original = self._base_values[sensor_name]
            self._base_values[sensor_name] = original * magnitude
            self.logger.info(f"Injected anomaly in {sensor_name}: {original} -> {original * magnitude}")

            # Schedule reset after 1 second
            async def reset():
                await asyncio.sleep(1.0)
                self._base_values[sensor_name] = original
                self.logger.info(f"Reset {sensor_name} to {original}")

            asyncio.create_task(reset())
            return True
        return False

    def reset_drift(self) -> None:
        """Reset all sensor drift to zero."""
        self._drift_offsets = {k: 0.0 for k in self._base_values}
        self._start_time = None
        self.logger.info("Sensor drift reset")

    async def calibrate(self) -> bool:
        """
        Perform mock calibration.

        Resets drift and sets calibrated flag.

        Returns:
            Always True for mock sensor
        """
        self.reset_drift()
        self._is_calibrated = True
        self.logger.info("Mock sensor calibrated")
        return True

    def get_current_readings(self) -> Dict[str, float]:
        """
        Get current readings for all sensors.

        Returns:
            Dictionary mapping sensor names to current values
        """
        readings = {}
        for sensor_name, base_value in self._base_values.items():
            drift = self._drift_offsets.get(sensor_name, 0.0)
            noise = random.gauss(0, self.noise_level * base_value)
            readings[sensor_name] = self._clamp_value(sensor_name, base_value + drift + noise)
        return readings


class MockActuatorPlugin(PluginABC):
    """
    Mock actuator plugin for testing actuator control and safety systems.

    Provides simulated actuators for:
    - Motor (speed control)
    - Servo (position control)
    - Heater (on/off)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._actuator_states: Dict[str, float] = {
            "motor_speed": 0.0,
            "servo_position": 90.0,
            "heater_power": 0.0,
        }

        self._stopped = False

        self._manifest = PluginManifest(
            name="MockActuator",
            version="1.0.0",
            capability=PluginCapability.ACTUATOR,
            sensors=[],
            actuators=list(self._actuator_states.keys()),
            safety_critical=True,
            requires_calibration=False,
            update_rate_hz=10.0,
            description="Mock actuator plugin for testing",
            author="SENSE Team",
        )

    def get_manifest(self) -> Dict[str, Any]:
        return self._manifest.to_dict()

    async def stream_auxiliary_input(self) -> AsyncIterator[SensorReading]:
        """Actuators can report their state as sensor data."""
        while self._is_running:
            for name, value in self._actuator_states.items():
                yield SensorReading(
                    sensor_name=f"{name}_feedback",
                    value=value,
                    confidence=1.0 if not self._stopped else 0.0,
                )
            await asyncio.sleep(0.1)

    def emergency_stop(self) -> None:
        """
        Emergency stop - halt all actuators.

        Sets all actuators to safe state:
        - Motor: 0 (stopped)
        - Servo: 90 (neutral)
        - Heater: 0 (off)
        """
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        self._actuator_states["motor_speed"] = 0.0
        self._actuator_states["servo_position"] = 90.0
        self._actuator_states["heater_power"] = 0.0
        self._stopped = True
        self._is_running = False

    def safety_policy(self) -> Dict[str, Any]:
        return {
            "max_motor_speed": 100.0,
            "min_motor_speed": 0.0,
            "max_servo_position": 180.0,
            "min_servo_position": 0.0,
            "max_heater_power": 100.0,
            "min_heater_power": 0.0,
        }

    def get_grounding_truth(self, query: str) -> Optional[float]:
        query_lower = query.lower()
        for name, value in self._actuator_states.items():
            if name.replace("_", " ") in query_lower or name in query_lower:
                return value
        return None

    def set_actuator(self, name: str, value: float) -> bool:
        """
        Set actuator value with safety checking.

        Args:
            name: Actuator name
            value: Desired value

        Returns:
            True if set successfully, False if blocked by safety
        """
        if self._stopped:
            self.logger.warning("Cannot set actuator - emergency stop active")
            return False

        if name not in self._actuator_states:
            return False

        # Check safety constraints
        policy = self.safety_policy()
        max_key = f"max_{name}"
        min_key = f"min_{name}"

        if max_key in policy and value > policy[max_key]:
            self.logger.warning(f"Blocked: {name}={value} exceeds max {policy[max_key]}")
            return False

        if min_key in policy and value < policy[min_key]:
            self.logger.warning(f"Blocked: {name}={value} below min {policy[min_key]}")
            return False

        self._actuator_states[name] = value
        return True

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop flag (requires explicit action)."""
        self._stopped = False
        self.logger.info("Emergency stop reset")
