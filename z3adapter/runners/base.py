"""Base protocol for Souffle runners.

This module defines the interface that all Souffle runners must implement,
enabling easy switching between official Souffle and mini-souffle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class RunResult:
    """Result from running a Souffle/Datalog program.

    Attributes:
        success: Whether execution completed without errors
        output_files: Mapping of relation name to output file path
        output_tuples: Mapping of relation name to list of tuples (if parsed)
        stdout: Standard output from execution
        stderr: Standard error from execution
        error: Error message if execution failed
    """

    success: bool
    output_files: dict[str, Path] = field(default_factory=dict)
    output_tuples: dict[str, list[tuple]] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@runtime_checkable
class SouffleRunner(Protocol):
    """Protocol for Souffle execution backends.

    This protocol defines the interface that both OfficialSouffleRunner
    and MiniSouffleRunner (future) must implement.
    """

    def run(
        self,
        program_path: Path,
        facts_dir: Path,
        output_dir: Path,
        timeout: float = 30.0,
    ) -> RunResult:
        """Execute a Souffle program.

        Args:
            program_path: Path to the .dl program file
            facts_dir: Directory containing .facts input files
            output_dir: Directory where output .csv files will be written
            timeout: Maximum execution time in seconds

        Returns:
            RunResult with execution status and outputs
        """
        ...

    def is_available(self) -> bool:
        """Check if this runner is available on the system.

        Returns:
            True if the runner can be used, False otherwise
        """
        ...

    def get_version(self) -> str | None:
        """Get the version of the Souffle implementation.

        Returns:
            Version string if available, None otherwise
        """
        ...
