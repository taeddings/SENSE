"""
SENSE Plugins Module
Contains plugin interface and ingestion pipeline for sensor/hardware integration.
"""

from core.plugins.interface import PluginABC, PluginCapability, PluginManifest
from core.plugins.mock_sensor import MockSensorPlugin

__all__ = [
    "PluginABC",
    "PluginCapability",
    "PluginManifest",
    "MockSensorPlugin",
]
