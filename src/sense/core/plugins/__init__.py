"""
SENSE Plugins Module
Contains plugin interface and ingestion pipeline for sensor/hardware integration.
"""

from sense.core.plugins.interface import PluginABC, PluginCapability, PluginManifest
from sense.core.plugins.mock_sensor import MockSensorPlugin

__all__ = [
    "PluginABC",
    "PluginCapability",
    "PluginManifest",
    "MockSensorPlugin",
]
