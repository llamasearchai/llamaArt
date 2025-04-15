"""
LlamaBench - A benchmarking framework for evaluating and comparing LLMs.

This package provides tools and utilities for running standardized benchmarks
across different language models, with support for various providers and tasks.
"""

__version__ = "0.1.0"

from llamabench.core import run
from llamabench.evaluators import ExactMatchEvaluator
from llamabench.models import ModelConfig
from llamabench.tasks import BenchmarkSuite, Task

__all__ = [
    "ModelConfig",
    "Task",
    "BenchmarkSuite",
    "ExactMatchEvaluator",
    "run",
]
