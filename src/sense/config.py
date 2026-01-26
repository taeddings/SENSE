import os
import yaml
import logging

# 1. LOAD YAML CONFIG
# Try to find config.yaml in CWD or up the directory tree
CONFIG_PATH = os.getenv("SENSE_CONFIG", "config.yaml")
_config_data = {}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            _config_data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Could not load {CONFIG_PATH}: {e}")

# 2. DEFINE EXPORTS (With Defaults)

# System Profile
system_profile = _config_data.get("system_profile", "mobile_termux")

# Feature Flags
ENABLE_HARVESTED_TOOLS = _config_data.get("ENABLE_HARVESTED_TOOLS", True)
ENABLE_VISION = _config_data.get("ENABLE_VISION", False)

# Intelligence Layer Settings (v4.0)
INTELLIGENCE_ENABLED = _config_data.get("intelligence", {}).get("enabled", True)
INTELLIGENCE_CONFIG = _config_data.get("intelligence", {
    "uncertainty": {
        "threshold": 0.6,
        "max_clarification_attempts": 2
    },
    "knowledge": {
        "vector_dimension": 384,
        "max_context_tokens": 500,
        "use_faiss": True
    },
    "preferences": {
        "enabled": True,
        "decay_days": 30
    },
    "metacognition": {
        "trace_enabled": True,
        "log_level": "info"
    }
})

# Memory Settings
MEMORY_BACKEND = _config_data.get("MEMORY_BACKEND", "native_engram")

# 3. LEGACY CONFIG CLASS (For API/Flask compatibility)
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "a-very-secret-key"
    SYSTEM_PROFILE = system_profile
    INTELLIGENCE_ENABLED = INTELLIGENCE_ENABLED
    INTELLIGENCE_CONFIG = INTELLIGENCE_CONFIG