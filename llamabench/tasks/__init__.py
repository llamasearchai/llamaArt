"""
Tasks module for LlamaBench.

This module provides the core classes for defining benchmark tasks and suites.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from llamabench.evaluators import Evaluator, ExactMatchEvaluator


@dataclass
class Task:
    """A benchmarking task consisting of examples, inputs, and evaluation."""

    name: str
    """Short descriptive name of the task."""

    description: str
    """Detailed description of what the task is testing."""

    instructions: str
    """Instructions provided to the model for the task."""

    examples: List[Dict[str, str]] = field(default_factory=list)
    """List of examples to provide to the model (few-shot prompting)."""

    inputs: List[Dict[str, str]] = field(default_factory=list)
    """List of inputs for the task, each is a dictionary with string fields."""

    reference_outputs: List[str] = field(default_factory=list)
    """Reference outputs for evaluation."""

    evaluator: Optional[Evaluator] = None
    """Evaluator to use for scoring model outputs."""

    def __post_init__(self):
        """Set a default evaluator if none was provided."""
        if self.evaluator is None:
            self.evaluator = ExactMatchEvaluator()


class BenchmarkSuite:
    """A collection of related benchmark tasks."""

    def __init__(self, name: str, description: str, tasks: List[Task]):
        """
        Initialize a benchmark suite.

        Args:
            name: Name of the benchmark suite
            description: Description of the suite
            tasks: List of tasks in the suite
        """
        self.name = name
        self.description = description
        self.tasks = tasks

    def __len__(self) -> int:
        """Return the number of tasks in the suite."""
        return len(self.tasks)

    def __getitem__(self, idx) -> Task:
        """Get a task by index."""
        return self.tasks[idx]


__all__ = ["Task", "BenchmarkSuite"]
