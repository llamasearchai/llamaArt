"""
Benchmark suites for LlamaBench.

This module provides predefined benchmark suites for common evaluation scenarios.
"""

from typing import Dict, List, Optional

from llamabench.tasks import BenchmarkSuite, Task

# Registry for predefined suites
_SUITES: Dict[str, BenchmarkSuite] = {}


def register_suite(suite: BenchmarkSuite) -> None:
    """
    Register a benchmark suite for later retrieval.

    Args:
        suite: The benchmark suite to register
    """
    _SUITES[suite.name] = suite


def get_suite(name: str) -> BenchmarkSuite:
    """
    Get a predefined benchmark suite by name.

    Args:
        name: Name of the benchmark suite to retrieve

    Returns:
        The requested benchmark suite

    Raises:
        KeyError: If the suite doesn't exist
    """
    if name not in _SUITES:
        raise KeyError(
            f"Benchmark suite '{name}' not found. Available suites: {list(_SUITES.keys())}"
        )
    return _SUITES[name]


def list_suites() -> List[str]:
    """
    List all available benchmark suites.

    Returns:
        A list of suite names
    """
    return list(_SUITES.keys())


# Import predefined suites to register them
# In a real implementation, these would be imported here to auto-register

__all__ = ["get_suite", "register_suite", "list_suites"]
