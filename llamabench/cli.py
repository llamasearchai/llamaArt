"""
Command-line interface for LlamaBench.

This module provides a command-line interface for running benchmarks.
"""

import argparse
import os
import sys
from typing import List, Optional

from llamabench import __version__
from llamabench.core import run
from llamabench.models import ModelConfig
from llamabench.suites import get_suite, list_suites


def parse_model_arg(model_arg: str) -> ModelConfig:
    """
    Parse a model argument in the format 'provider:model'.

    Args:
        model_arg: Model argument string (e.g., 'openai:gpt-4')

    Returns:
        A ModelConfig object

    Raises:
        ValueError: If the model argument is invalid
    """
    try:
        provider, model = model_arg.split(":", 1)
        return ModelConfig(provider=provider, model=model)
    except ValueError:
        raise ValueError(
            f"Invalid model format: {model_arg}. Expected 'provider:model' (e.g., 'openai:gpt-4')"
        )


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the LlamaBench CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="LlamaBench: A benchmarking framework for LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"LlamaBench v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    run_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to benchmark (format: provider:model, e.g., openai:gpt-4)",
    )
    run_parser.add_argument(
        "--suite", help=f"Benchmark suite to run. Available: {', '.join(list_suites())}"
    )
    run_parser.add_argument(
        "--tasks", nargs="+", help="Specific tasks to run (instead of a suite)"
    )
    run_parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run benchmarks in parallel",
    )
    run_parser.add_argument(
        "--output-dir", default="./results", help="Directory to save results"
    )
    run_parser.add_argument(
        "--report-formats",
        nargs="+",
        default=["json"],
        choices=["json", "csv", "markdown", "html"],
        help="Output report formats",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available suites and tasks")
    list_parser.add_argument(
        "--suites", action="store_true", help="List available benchmark suites"
    )

    parsed_args = parser.parse_args(args)

    # Handle commands
    if parsed_args.command is None:
        parser.print_help()
        return 1

    if parsed_args.command == "list":
        # List available suites and tasks
        if parsed_args.suites:
            print("Available benchmark suites:")
            for suite_name in list_suites():
                print(f"  - {suite_name}")
        else:
            # Default to listing both if no specific option is provided
            print("Available benchmark suites:")
            for suite_name in list_suites():
                print(f"  - {suite_name}")
        return 0

    elif parsed_args.command == "run":
        # Run benchmarks
        try:
            # Parse model configs
            models = [parse_model_arg(model_arg) for model_arg in parsed_args.models]

            # Get benchmark suite
            suite = None
            if parsed_args.suite:
                suite = get_suite(parsed_args.suite)

            # Create output directory
            os.makedirs(parsed_args.output_dir, exist_ok=True)

            # Run the benchmark
            results = run(
                models=models,
                suite=suite,
                parallel=parsed_args.parallel,
            )

            # Save results in requested formats
            for format_name in parsed_args.report_formats:
                output_path = os.path.join(
                    parsed_args.output_dir, f"results.{format_name}"
                )
                if format_name == "json":
                    with open(output_path, "w") as f:
                        f.write(results.to_json())
                elif format_name == "csv":
                    results.to_dataframe().to_csv(output_path, index=False)
                else:
                    print(
                        f"Warning: Report format '{format_name}' not implemented in this skeleton"
                    )

            # Print summary to console
            print(results.summary())
            print(f"Results saved to {parsed_args.output_dir}")
            return 0

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
