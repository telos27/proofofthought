"""Unit tests for Souffle backend.

Tests the SouffleBackend which compiles IKR to Datalog and executes via Souffle.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip if z3 is not available
try:
    from z3adapter.backends.souffle_backend import SouffleBackend
    from z3adapter.runners.base import RunResult

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


def create_simple_ikr() -> dict:
    """Create a simple IKR for testing."""
    return {
        "meta": {"question": "Is the sky blue?", "question_type": "yes_no"},
        "types": [{"name": "Object"}],
        "entities": [{"name": "sky", "type": "Object"}],
        "relations": [{"name": "is_blue", "signature": ["Object"], "range": "Bool"}],
        "facts": [{"predicate": "is_blue", "arguments": ["sky"]}],
        "rules": [],
        "query": {"predicate": "is_blue", "arguments": ["sky"]},
    }


def create_vegetarian_ikr() -> dict:
    """Create IKR for vegetarian burger question."""
    return {
        "meta": {"question": "Would a vegetarian eat a plant burger?"},
        "types": [{"name": "Person"}, {"name": "Food"}],
        "entities": [
            {"name": "vegetarian_person", "type": "Person"},
            {"name": "plant_burger", "type": "Food"},
        ],
        "relations": [
            {"name": "is_vegetarian", "signature": ["Person"], "range": "Bool"},
            {"name": "is_plant_based", "signature": ["Food"], "range": "Bool"},
            {"name": "would_eat", "signature": ["Person", "Food"], "range": "Bool"},
        ],
        "facts": [
            {"predicate": "is_vegetarian", "arguments": ["vegetarian_person"]},
            {"predicate": "is_plant_based", "arguments": ["plant_burger"]},
        ],
        "rules": [
            {
                "name": "vegetarians eat plants",
                "quantified_vars": [
                    {"name": "p", "type": "Person"},
                    {"name": "f", "type": "Food"},
                ],
                "antecedent": {
                    "and": [
                        {"predicate": "is_vegetarian", "arguments": ["p"]},
                        {"predicate": "is_plant_based", "arguments": ["f"]},
                    ]
                },
                "consequent": {"predicate": "would_eat", "arguments": ["p", "f"]},
            }
        ],
        "query": {
            "predicate": "would_eat",
            "arguments": ["vegetarian_person", "plant_burger"],
        },
    }


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestSouffleBackend(unittest.TestCase):
    """Tests for SouffleBackend."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock runner
        self.mock_runner = MagicMock()
        self.mock_runner.is_available.return_value = True

    def test_backend_initialization_with_runner(self):
        """Test backend initialization with custom runner."""
        backend = SouffleBackend(runner=self.mock_runner)
        self.assertIsNotNone(backend)

    def test_backend_initialization_fails_without_souffle(self):
        """Test backend raises error when Souffle not available."""
        mock_runner = MagicMock()
        mock_runner.is_available.return_value = False

        with self.assertRaises(RuntimeError) as context:
            SouffleBackend(runner=mock_runner)

        self.assertIn("not available", str(context.exception))

    def test_execute_derivable_query(self):
        """Test execution returns True for derivable query."""
        # Mock runner to return query_result with tuples
        self.mock_runner.run.return_value = RunResult(
            success=True,
            output_tuples={"query_result": [()]},  # Empty tuple means derivable
            stdout="",
            stderr="",
        )

        backend = SouffleBackend(runner=self.mock_runner)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(create_simple_ikr(), f)
            f.flush()

            result = backend.execute(f.name)

        self.assertTrue(result.success)
        self.assertTrue(result.answer)  # Derivable = True
        self.assertEqual(result.sat_count, 1)
        self.assertEqual(result.unsat_count, 0)

    def test_execute_not_derivable_query(self):
        """Test execution returns False for non-derivable query."""
        # Mock runner to return empty query_result
        self.mock_runner.run.return_value = RunResult(
            success=True,
            output_tuples={"query_result": []},  # Empty = not derivable
            stdout="",
            stderr="",
        )

        backend = SouffleBackend(runner=self.mock_runner)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(create_simple_ikr(), f)
            f.flush()

            result = backend.execute(f.name)

        self.assertTrue(result.success)
        self.assertFalse(result.answer)  # Not derivable = False
        self.assertEqual(result.sat_count, 0)
        self.assertEqual(result.unsat_count, 1)

    def test_execute_with_rules(self):
        """Test execution with rules that derive the query."""
        # Mock runner to return derivable result
        self.mock_runner.run.return_value = RunResult(
            success=True,
            output_tuples={"query_result": [()]},
            stdout="",
            stderr="",
        )

        backend = SouffleBackend(runner=self.mock_runner)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(create_vegetarian_ikr(), f)
            f.flush()

            result = backend.execute(f.name)

        self.assertTrue(result.success)
        self.assertTrue(result.answer)

        # Verify runner was called with correct arguments
        self.mock_runner.run.assert_called_once()
        call_args = self.mock_runner.run.call_args
        self.assertIsInstance(call_args.kwargs["program_path"], Path)
        self.assertIsInstance(call_args.kwargs["facts_dir"], Path)

    def test_execute_runner_failure(self):
        """Test handling of runner execution failure."""
        self.mock_runner.run.return_value = RunResult(
            success=False,
            error="Souffle syntax error",
            stdout="",
            stderr="Error at line 1",
        )

        backend = SouffleBackend(runner=self.mock_runner)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(create_simple_ikr(), f)
            f.flush()

            result = backend.execute(f.name)

        self.assertFalse(result.success)
        self.assertIsNone(result.answer)
        self.assertIn("Souffle syntax error", result.error)

    def test_execute_invalid_json(self):
        """Test handling of invalid JSON input."""
        backend = SouffleBackend(runner=self.mock_runner)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {{{")
            f.flush()

            result = backend.execute(f.name)

        self.assertFalse(result.success)
        self.assertIn("Invalid JSON", result.error)

    def test_execute_invalid_ikr_schema(self):
        """Test handling of invalid IKR schema."""
        backend = SouffleBackend(runner=self.mock_runner)

        invalid_ikr = {"meta": {"question": "Test"}}  # Missing required fields

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(invalid_ikr, f)
            f.flush()

            result = backend.execute(f.name)

        self.assertFalse(result.success)
        self.assertIn("validation failed", result.error.lower())

    def test_execute_ikr_validation_error(self):
        """Test handling of IKR reference validation errors."""
        backend = SouffleBackend(runner=self.mock_runner)

        invalid_ikr = {
            "meta": {"question": "Test"},
            "types": [],
            "entities": [{"name": "x", "type": "UndefinedType"}],  # Invalid ref
            "relations": [],
            "facts": [],
            "rules": [],
            "query": {"predicate": "undefined", "arguments": ["x"]},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(invalid_ikr, f)
            f.flush()

            result = backend.execute(f.name)

        self.assertFalse(result.success)
        self.assertIn("compilation error", result.error.lower())

    def test_get_file_extension(self):
        """Test file extension is .json for IKR input."""
        backend = SouffleBackend(runner=self.mock_runner)
        self.assertEqual(backend.get_file_extension(), ".json")

    def test_get_prompt_template(self):
        """Test prompt template is returned."""
        backend = SouffleBackend(runner=self.mock_runner)
        template = backend.get_prompt_template()
        self.assertIsInstance(template, str)
        self.assertGreater(len(template), 0)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestSouffleBackendIntegration(unittest.TestCase):
    """Integration tests that verify the full compilation pipeline."""

    def setUp(self):
        """Set up test fixtures with a mock runner that captures inputs."""
        self.mock_runner = MagicMock()
        self.mock_runner.is_available.return_value = True
        self.captured_program = None
        self.captured_facts = None

        def capture_run(program_path, facts_dir, output_dir, timeout=30.0):
            # Capture the generated program and facts
            self.captured_program = program_path.read_text()

            self.captured_facts = {}
            for facts_file in facts_dir.glob("*.facts"):
                rel_name = facts_file.stem
                with open(facts_file) as f:
                    self.captured_facts[rel_name] = f.read()

            return RunResult(
                success=True,
                output_tuples={"query_result": [()]},
                stdout="",
                stderr="",
            )

        self.mock_runner.run.side_effect = capture_run

    def test_generates_valid_souffle_program(self):
        """Test that a valid Souffle program is generated."""
        backend = SouffleBackend(runner=self.mock_runner)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(create_simple_ikr(), f)
            f.flush()

            backend.execute(f.name)

        # Verify program structure
        self.assertIn(".decl is_blue", self.captured_program)
        self.assertIn(".input is_blue", self.captured_program)
        self.assertIn(".output query_result", self.captured_program)
        self.assertIn("query_result() :- is_blue(sky).", self.captured_program)

    def test_generates_valid_facts_files(self):
        """Test that valid facts files are generated."""
        backend = SouffleBackend(runner=self.mock_runner)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(create_simple_ikr(), f)
            f.flush()

            backend.execute(f.name)

        # Verify facts
        self.assertIn("is_blue", self.captured_facts)
        self.assertEqual(self.captured_facts["is_blue"].strip(), "sky")

    def test_generates_rules_correctly(self):
        """Test that rules are compiled to Horn clauses."""
        backend = SouffleBackend(runner=self.mock_runner)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(create_vegetarian_ikr(), f)
            f.flush()

            backend.execute(f.name)

        # Verify rule compilation
        self.assertIn(
            "would_eat(p, f) :- is_vegetarian(p), is_plant_based(f).",
            self.captured_program,
        )


if __name__ == "__main__":
    unittest.main()
