"""
Models module for LlamaBench.

This module provides utilities for configuring and interacting with different LLM providers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for a language model to benchmark."""

    provider: str
    """The provider of the model (e.g., 'openai', 'anthropic', 'huggingface', 'llamacpp')."""

    model: str
    """The specific model to use (e.g., 'gpt-4', 'claude-3-opus-20240229')."""

    temperature: float = 0.0
    """The sampling temperature to use (default: 0.0 for deterministic outputs)."""

    max_tokens: Optional[int] = 1024
    """The maximum number of tokens to generate (default: 1024)."""

    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional model-specific parameters to pass to the provider's API."""

    def __str__(self) -> str:
        """Return a string representation of the model config."""
        return f"{self.provider}:{self.model}"


__all__ = ["ModelConfig"]
