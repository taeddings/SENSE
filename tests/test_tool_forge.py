import pytest
from sense.core.plugins.forge import ToolForge

def test_tool_forge_instantiation():
    forge = ToolForge()
    assert forge is not None
    assert hasattr(forge, 'scan_memory')

def test_scan_memory_empty():
    forge = ToolForge()
    candidates = forge.scan_memory([])
    assert len(candidates) == 0

# Add more tests as implementation progresses
