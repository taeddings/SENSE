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

# Memory Settings
MEMORY_BACKEND = _config_data.get("MEMORY_BACKEND", "native_engram")

# 3. LEGACY CONFIG CLASS (For API/Flask compatibility)
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "a-very-secret-key"
    SYSTEM_PROFILE = system_profile