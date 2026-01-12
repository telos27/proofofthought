"""Official Souffle CLI runner.

This module provides a runner that executes Datalog programs using
the official Souffle compiler via subprocess.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from z3adapter.runners.base import RunResult

logger = logging.getLogger(__name__)


class OfficialSouffleRunner:
    """Runner that uses the official Souffle CLI.

    This runner invokes Souffle via subprocess with the standard
    command-line interface:
        souffle -F <facts_dir> -D <output_dir> <program.dl>
    """

    def __init__(self, souffle_path: str | None = None) -> None:
        """Initialize the runner.

        Args:
            souffle_path: Optional path to souffle binary.
                         If not provided, searches PATH.
        """
        self._souffle_path = souffle_path or shutil.which("souffle")
        self._version: str | None = None

    def is_available(self) -> bool:
        """Check if Souffle is available."""
        if not self._souffle_path:
            return False
        try:
            result = subprocess.run(
                [self._souffle_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def get_version(self) -> str | None:
        """Get Souffle version string."""
        if self._version:
            return self._version

        if not self._souffle_path:
            return None

        try:
            result = subprocess.run(
                [self._souffle_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse version from output like "Souffle 2.4.1"
                self._version = result.stdout.strip()
                return self._version
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

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
        if not self._souffle_path:
            return RunResult(
                success=False,
                error="Souffle binary not found. Please install Souffle.",
            )

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            self._souffle_path,
            "-F",
            str(facts_dir),
            "-D",
            str(output_dir),
            str(program_path),
        ]

        logger.debug(f"Running Souffle: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                return RunResult(
                    success=False,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error=f"Souffle exited with code {result.returncode}: {result.stderr}",
                )

            # Collect output files
            output_files: dict[str, Path] = {}
            output_tuples: dict[str, list[tuple]] = {}

            for csv_file in output_dir.glob("*.csv"):
                relation_name = csv_file.stem
                output_files[relation_name] = csv_file

                # Parse CSV content
                tuples = self._parse_csv(csv_file)
                output_tuples[relation_name] = tuples

            return RunResult(
                success=True,
                output_files=output_files,
                output_tuples=output_tuples,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        except subprocess.TimeoutExpired:
            return RunResult(
                success=False,
                error=f"Souffle execution timed out after {timeout} seconds",
            )
        except FileNotFoundError:
            return RunResult(
                success=False,
                error=f"Souffle binary not found at: {self._souffle_path}",
            )
        except OSError as e:
            return RunResult(
                success=False,
                error=f"Failed to run Souffle: {e}",
            )

    def _parse_csv(self, csv_path: Path) -> list[tuple]:
        """Parse a Souffle output CSV file.

        Souffle outputs tab-separated values, one tuple per line.

        Args:
            csv_path: Path to the CSV file

        Returns:
            List of tuples from the file
        """
        tuples = []
        try:
            with open(csv_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Split by tab
                        values = line.split("\t")
                        # Try to convert to appropriate types
                        converted = []
                        for v in values:
                            try:
                                converted.append(int(v))
                            except ValueError:
                                try:
                                    converted.append(float(v))
                                except ValueError:
                                    converted.append(v)
                        tuples.append(tuple(converted))
        except OSError as e:
            logger.warning(f"Failed to parse CSV {csv_path}: {e}")
        return tuples
