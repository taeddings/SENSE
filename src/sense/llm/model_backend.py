"""LLM Backend Abstraction"""

import os
import requests
from typing import Dict, Any, Optional

class ModelBackend:
    """
    Unified interface for LLM backends: OpenAI, Anthropic, Ollama, LM Studio, Transformers, vLLM, custom HTTP.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_type, self.model_name = self._parse_model_name()
        self.endpoint: Optional[str] = self._resolve_endpoint()
        self.api_key: Optional[str] = self._get_api_key()
        self.local_pipeline = None

    def _parse_model_name(self) -> tuple[str, str]:
        """Parse model_name into backend and model."""
        full_name = self.config.get('model_name', 'gpt2')
        if '/' in full_name:
            backend, model = full_name.split('/', 1)
        else:
            backend = 'transformers'
            model = full_name
        return backend.lower(), model

    def _resolve_endpoint(self) -> Optional[str]:
        """Resolve endpoint based on backend."""
        if self.backend_type == 'openai':
            return 'https://api.openai.com/v1/chat/completions'
        elif self.backend_type == 'anthropic':
            return 'https://api.anthropic.com/v1/messages'
        elif self.backend_type == 'ollama':
            return os.getenv('OLLAMA_HOST', 'http://localhost:11434') + '/api/generate'
        elif self.backend_type == 'lmstudio':
            return 'http://localhost:1234/v1/completions'
        elif self.backend_type == 'vllm':
            return 'http://localhost:8000/v1/completions'
        elif self.backend_type.startswith('http'):
            return self.backend_type  # Custom HTTP
        return None  # Transformers

    def _get_api_key(self) -> Optional[str]:
        """Get API key from env or config."""
        if self.backend_type in ['openai', 'anthropic']:
            key_name = f"{self.backend_type.upper()}_API_KEY"
            return os.getenv(key_name) or self.config.get(f"{self.backend_type}_api_key")
        return None

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate response from prompt.
        """
        try:
            if self.backend_type == 'openai':
                return self._call_openai(prompt, max_tokens, temperature)
            elif self.backend_type == 'anthropic':
                return self._call_anthropic(prompt, max_tokens, temperature)
            elif self.backend_type in ['ollama', 'lmstudio', 'vllm'] or self.backend_type.startswith('http'):
                return self._call_http_api(prompt, max_tokens, temperature)
            else:
                return self._call_transformers(prompt, max_tokens, temperature)
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")

    def _call_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

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
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': False
        }
        if self.backend_type == 'ollama':
            payload['options'] = {'temperature': temperature, 'num_predict': max_tokens}
        response = requests.post(self.endpoint, json=payload, timeout=60)
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