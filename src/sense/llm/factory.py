import os
import logging
from typing import Optional
from sense.config import _config_data

class LLMFactory:
    """
    Creates an LLM client based on CLI args -> Config Profile -> Defaults.
    """
    @staticmethod
    def create_client(cli_provider=None, cli_url=None, cli_model=None, cli_key=None):
        # 1. Load Active Profile
        active_profile_name = _config_data.get("active_llm_profile", "termux_local")
        profile = _config_data.get("llm_profiles", {}).get(active_profile_name, {})

        # 2. Resolve Settings (CLI overrides Profile)
        provider = cli_provider or profile.get("provider", "openai_compatible")
        base_url = cli_url or profile.get("base_url", "http://127.0.0.1:8080/v1")
        model_name = cli_model or profile.get("model_name", "local-model")
        api_key = cli_key or profile.get("api_key", os.getenv("OPENAI_API_KEY", "sk-dummy"))

        logging.info(f"🧠 LLM CONNECT: {provider} @ {base_url} [{model_name}]")

        # 3. Instantiate Client
        try:
            if provider in ["openai", "openai_compatible"]:
                from openai import AsyncOpenAI
                return AsyncOpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            logging.error("❌ OpenAI library missing. pip install openai")
            return None
        
        logging.error(f"❌ Unknown provider: {provider}")
        return None

    @staticmethod
    def get_model_name(cli_model=None):
        active_profile_name = _config_data.get("active_llm_profile", "termux_local")
        profile = _config_data.get("llm_profiles", {}).get(active_profile_name, {})
        return cli_model or profile.get("model_name", "local-model")
