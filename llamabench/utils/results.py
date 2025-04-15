"""
Result handling utilities for LlamaBench.

This module provides classes for storing and analyzing benchmark results.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from llamabench.models import ModelConfig
from llamabench.tasks import Task


@dataclass
class TaskResult:
    """Results for a single model on a single task."""

    task: Task
    model: ModelConfig
    model_output: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkResults:
    """Container for benchmark results across models and tasks."""

    def __init__(self, models: List[ModelConfig], tasks: List[Task]):
        """
        Initialize a results container.

        Args:
            models: List of models that were benchmarked
            tasks: List of tasks that were evaluated
        """
        self.models = models
        self.tasks = tasks
        self.results: List[TaskResult] = []

    def add_result(self, result: TaskResult):
        """
        Add a task result to the container.

        Args:
            result: The task result to add
        """
        self.results.append(result)

    def summary(self) -> str:
        """
        Generate a summary of the benchmark results.

        Returns:
            A string containing a tabular summary of the results
        """
        # In a real implementation, this would create a nicely formatted table
        # of model scores across tasks
        return "Benchmark results summary (not implemented in this skeleton)"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns:
            A DataFrame containing all benchmark results
        """
        # Convert results to a pandas DataFrame for analysis
        data = []
        for result in self.results:
            data.append(
                {
                    "model": str(result.model),
                    "task": result.task.name,
                    "score": result.score,
                    # Add other fields as needed
                }
            )
        return pd.DataFrame(data)

    def to_json(self) -> str:
        """
        Convert results to a JSON string.

        Returns:
            A JSON string representation of the results
        """
        # Convert results to a JSON-serializable format
        # (Would need proper serialization logic for complex objects)
        return json.dumps(
            {
                "models": [str(model) for model in self.models],
                "tasks": [task.name for task in self.tasks],
                "results": [
                    {
                        "model": str(result.model),
                        "task": result.task.name,
                        "score": result.score,
                    }
                    for result in self.results
                ],
            },
            indent=2,
        )


__all__ = ["TaskResult", "BenchmarkResults"]
