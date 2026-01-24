"""SENSE LLM Module - Model backend abstraction layer."""

from .model_backend import ModelBackend, get_model

__all__ = ['ModelBackend', 'get_model']
