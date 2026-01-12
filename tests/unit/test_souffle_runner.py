"""Unit tests for Souffle runners.

Tests the runner abstraction layer that enables switching between
official Souffle and mini-souffle implementations.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip if z3 is not available (required for z3adapter imports)
try:
    from z3adapter.runners.base import RunResult, SouffleRunner
    from z3adapter.runners.official import OfficialSouffleRunner

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestRunResult(unittest.TestCase):
    """Tests for RunResult dataclass."""

    def test_run_result_success(self):
        """Test successful RunResult creation."""
        result = RunResult(
            success=True,
            output_files={"query_result": Path("/tmp/query_result.csv")},
            output_tuples={"query_result": [()]},
            stdout="",
            stderr="",
        )

        self.assertTrue(result.success)
        self.assertIn("query_result", result.output_tuples)
        self.assertIsNone(result.error)

    def test_run_result_failure(self):
        """Test failed RunResult creation."""
        result = RunResult(
            success=False,
            error="Souffle not found",
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Souffle not found")
        self.assertEqual(result.output_tuples, {})


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestOfficialSouffleRunner(unittest.TestCase):
    """Tests for OfficialSouffleRunner."""

    def test_is_available_when_souffle_exists(self):
        """Test is_available returns True when souffle is found."""
        with patch("shutil.which", return_value="/usr/bin/souffle"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="Souffle 2.4")
                runner = OfficialSouffleRunner()
                self.assertTrue(runner.is_available())

    def test_is_available_when_souffle_missing(self):
        """Test is_available returns False when souffle not found."""
        with patch("shutil.which", return_value=None):
            runner = OfficialSouffleRunner()
            self.assertFalse(runner.is_available())

    def test_get_version(self):
        """Test get_version returns version string."""
        with patch("shutil.which", return_value="/usr/bin/souffle"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="Souffle 2.4.1 (1234567)"
                )
                runner = OfficialSouffleRunner()
                version = runner.get_version()
                self.assertIn("Souffle", version)

    def test_run_success(self):
        """Test successful Souffle execution."""
        with patch("shutil.which", return_value="/usr/bin/souffle"):
            runner = OfficialSouffleRunner()

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Create mock program and facts
                program_path = tmpdir / "test.dl"
                program_path.write_text(".decl test()\n.output test\ntest().")

                facts_dir = tmpdir / "facts"
                facts_dir.mkdir()

                output_dir = tmpdir / "output"

                with patch("subprocess.run") as mock_run:
                    # Mock successful execution
                    mock_run.return_value = MagicMock(
                        returncode=0, stdout="", stderr=""
                    )

                    # Create fake output file
                    output_dir.mkdir()
                    (output_dir / "test.csv").write_text("")

                    result = runner.run(program_path, facts_dir, output_dir)

                    self.assertTrue(result.success)
                    self.assertIn("test", result.output_files)

    def test_run_with_tuples(self):
        """Test that output tuples are parsed correctly."""
        with patch("shutil.which", return_value="/usr/bin/souffle"):
            runner = OfficialSouffleRunner()

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                program_path = tmpdir / "test.dl"
                program_path.write_text("")

                facts_dir = tmpdir / "facts"
                facts_dir.mkdir()

                output_dir = tmpdir / "output"
                output_dir.mkdir()

                # Create output file with tuples
                (output_dir / "path.csv").write_text("1\t2\n2\t3\nalice\tbob\n")

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=0, stdout="", stderr=""
                    )

                    result = runner.run(program_path, facts_dir, output_dir)

                    self.assertTrue(result.success)
                    self.assertIn("path", result.output_tuples)

                    # Check type conversion
                    tuples = result.output_tuples["path"]
                    self.assertEqual(tuples[0], (1, 2))  # Converted to int
                    self.assertEqual(tuples[1], (2, 3))
                    self.assertEqual(tuples[2], ("alice", "bob"))  # Strings

    def test_run_failure(self):
        """Test handling of Souffle execution failure."""
        with patch("shutil.which", return_value="/usr/bin/souffle"):
            runner = OfficialSouffleRunner()

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                program_path = tmpdir / "test.dl"
                program_path.write_text("")

                facts_dir = tmpdir / "facts"
                facts_dir.mkdir()

                output_dir = tmpdir / "output"

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=1,
                        stdout="",
                        stderr="Error: syntax error at line 1",
                    )

                    result = runner.run(program_path, facts_dir, output_dir)

                    self.assertFalse(result.success)
                    self.assertIn("syntax error", result.error)

    def test_run_timeout(self):
        """Test handling of Souffle timeout."""
        import subprocess

        with patch("shutil.which", return_value="/usr/bin/souffle"):
            runner = OfficialSouffleRunner()

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                program_path = tmpdir / "test.dl"
                program_path.write_text("")

                facts_dir = tmpdir / "facts"
                facts_dir.mkdir()

                output_dir = tmpdir / "output"

                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = subprocess.TimeoutExpired("souffle", 30)

                    result = runner.run(program_path, facts_dir, output_dir, timeout=30)

                    self.assertFalse(result.success)
                    self.assertIn("timed out", result.error)

    def test_run_binary_not_found(self):
        """Test handling when souffle binary disappears."""
        with patch("shutil.which", return_value=None):
            runner = OfficialSouffleRunner(souffle_path=None)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                result = runner.run(
                    tmpdir / "test.dl", tmpdir / "facts", tmpdir / "output"
                )

                self.assertFalse(result.success)
                self.assertIn("not found", result.error)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestSouffleRunnerProtocol(unittest.TestCase):
    """Tests that OfficialSouffleRunner implements SouffleRunner protocol."""

    def test_implements_protocol(self):
        """Test that OfficialSouffleRunner implements SouffleRunner."""
        self.assertTrue(isinstance(OfficialSouffleRunner, type))

        # Check required methods exist
        self.assertTrue(hasattr(OfficialSouffleRunner, "run"))
        self.assertTrue(hasattr(OfficialSouffleRunner, "is_available"))
        self.assertTrue(hasattr(OfficialSouffleRunner, "get_version"))


if __name__ == "__main__":
    unittest.main()
