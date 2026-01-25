"""LLM Backend Abstraction"""

import os
import requests
from typing import Dict, Any, Optional

# Safe import for OpenAI
try:
    from openai import OpenAI, APIConnectionError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI = None
    APIConnectionError = Exception # Fallback to generic Exception if not available to avoid NameError
    OPENAI_AVAILABLE = False

class ModelBackend:
    """
    Unified interface for LLM backends: OpenAI, Anthropic, Ollama, LM Studio, Transformers, vLLM, custom HTTP.
    Now supports local OpenAI-compatible endpoints (e.g., llama.cpp).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_type, self.model_name = self._parse_model_name()
        self.endpoint: Optional[str] = self._resolve_endpoint()
        self.api_key: Optional[str] = self._get_api_key()
        self.local_pipeline = None
        self.openai_client = None

        if self.backend_type in ['openai', 'local', 'vllm', 'lmstudio']:
            if OPENAI_AVAILABLE:
                base_url = self.endpoint if self.backend_type != 'openai' else None
                self.openai_client = OpenAI(
                    api_key=self.api_key or "dummy",
                    base_url=base_url
                )
            else:
                # If configured for these backends but openai is missing, we shouldn't fallback silently
                print(f"Warning: 'openai' package not found. {self.backend_type} backend will not work.")

    def _parse_model_name(self) -> tuple[str, str]:
        """Parse model_name into backend and model."""
        full_name = self.config.get('model_name', 'gpt2')
        if '/' in full_name:
            backend, model = full_name.split('/', 1)
        else:
            # Default to local if just a model name is provided and we are configured for it
            if self.config.get('backend') == 'local' or self.config.get('use_local_llm', False):
                backend = 'local'
                model = full_name
            else:
                backend = 'transformers'
                model = full_name
        return backend.lower(), model

    def _resolve_endpoint(self) -> Optional[str]:
        """Resolve endpoint based on backend."""
        # Check config for explicit override
        if self.config.get('api_base'):
            return self.config['api_base']

        if self.backend_type == 'openai':
            return None # OpenAI client handles default
        elif self.backend_type == 'anthropic':
            return 'https://api.anthropic.com/v1/messages'
        elif self.backend_type == 'ollama':
            return os.getenv('OLLAMA_HOST', 'http://localhost:11434') + '/api/generate'
        elif self.backend_type == 'lmstudio':
            return 'http://192.168.0.243:1234/v1' # Client adds /chat/completions
        elif self.backend_type == 'vllm':
            return 'http://localhost:8000/v1'
        elif self.backend_type == 'local':
            return 'http://localhost:8080/v1'
        elif self.backend_type.startswith('http'):
            return self.backend_type  # Custom HTTP
        return None  # Transformers

    def _get_api_key(self) -> Optional[str]:
        """Get API key from env or config."""
        if self.backend_type in ['openai', 'anthropic']:
            key_name = f"{self.backend_type.upper()}_API_KEY"
            return os.getenv(key_name) or self.config.get(f"{self.backend_type}_api_key")
        return "sense-local" # Default for local/offline

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate response from prompt.
        """
        try:
            if self.openai_client:
                return self._call_openai_compatible(prompt, max_tokens, temperature)
            elif self.backend_type == 'local' and not self.openai_client:
                 return "SENSE ERROR: 'openai' python package is required for local backend but not installed."
            elif self.backend_type == 'anthropic':
                return self._call_anthropic(prompt, max_tokens, temperature)
            elif self.backend_type == 'ollama': # Ollama can be OpenAI compatible but keeping legacy path for now
                return self._call_http_api(prompt, max_tokens, temperature)
            elif self.backend_type.startswith('http'):
                return self._call_http_api(prompt, max_tokens, temperature)
            else:
                return self._call_transformers(prompt, max_tokens, temperature)
        except APIConnectionError:
             return "SENSE OFFLINE: Connection refused. Is the local LLM server running?"
        except Exception as e:
            # Fallback or re-raise
            raise RuntimeError(f"LLM generation failed: {e}")

    def _call_openai_compatible(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        message = response.choices[0].message
        content = message.content or ""
        
        # Handle reasoning_content if present (common in thinking models like DeepSeek-R1 or llama.cpp thinking models)
        # We try to get it from attribute or model_extra/additional fields
        reasoning = getattr(message, 'reasoning_content', None)
        
        if not reasoning and hasattr(message, 'model_extra'):
            reasoning = message.model_extra.get('reasoning_content')
            
        if reasoning:
            if content:
                return f"<thought>\n{reasoning}\n</thought>\n{content}"
            else:
                return reasoning
                
        return content

    def _call_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
        payload = {
            'model': self.model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': [{'role': 'user', 'content': prompt}]
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['content'][0]['text']

    def _call_http_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Legacy HTTP call for Ollama/Custom
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': False
        }
        if self.backend_type == 'ollama':
            payload['options'] = {'temperature': temperature, 'num_predict': max_tokens}
        
        endpoint = self.endpoint
        # Adjust endpoint for legacy calls if needed
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        if self.backend_type == 'ollama':
            return response.json()['response']
        else:
            return response.json()['choices'][0]['text']

    def _call_transformers(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if self.local_pipeline is None:
            from transformers import pipeline
            self.local_pipeline = pipeline("text-generation", model=self.model_name, device=-1)
        response = self.local_pipeline(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)[0]['generated_text']
        return response[len(prompt):].strip()

def get_model(config: Dict[str, Any]) -> ModelBackend:
    """
    Factory for ModelBackend.
    """
    return ModelBackend(config)
