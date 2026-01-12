"""Souffle runner implementations for Datalog execution.

This module provides an abstraction layer for executing Souffle/Datalog programs,
allowing easy switching between different Souffle implementations:
- OfficialSouffleRunner: Uses the official Souffle CLI
- MiniSouffleRunner: Uses mini-souffle (future implementation)
"""

from z3adapter.runners.base import RunResult, SouffleRunner
from z3adapter.runners.official import OfficialSouffleRunner

__all__ = ["SouffleRunner", "RunResult", "OfficialSouffleRunner", "get_runner"]


def get_runner(prefer: str = "official") -> SouffleRunner:
    """Get an available Souffle runner.

    Args:
        prefer: Preferred runner type ("official" or "mini")

    Returns:
        An available SouffleRunner instance

    Raises:
        RuntimeError: If no Souffle runner is available
    """
    if prefer == "official":
        runner = OfficialSouffleRunner()
        if runner.is_available():
            return runner
        # TODO: Fall back to mini-souffle when available
        raise RuntimeError(
            "Souffle is not available. Please install Souffle: "
            "https://souffle-lang.github.io/install"
        )
    elif prefer == "mini":
        # TODO: Implement MiniSouffleRunner
        raise NotImplementedError("Mini-souffle runner not yet implemented")
    else:
        raise ValueError(f"Unknown runner type: {prefer}")
