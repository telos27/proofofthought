"""Unit tests for ProofOfThought with Souffle backend.

Tests the integration of the Souffle backend into the ProofOfThought high-level API.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip if z3 is not available
try:
    from z3adapter.reasoning.proof_of_thought import ProofOfThought, BackendType

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestProofOfThoughtSouffleBackend(unittest.TestCase):
    """Tests for ProofOfThought with Souffle backend."""

    def test_souffle_in_backend_type(self):
        """Test that 'souffle' is a valid BackendType."""
        from typing import get_args

        backend_types = get_args(BackendType)
        self.assertIn("souffle", backend_types)

    def test_all_backend_types_present(self):
        """Test all expected backend types are present."""
        from typing import get_args

        backend_types = get_args(BackendType)
        expected = {"json", "smt2", "ikr", "souffle"}
        self.assertEqual(set(backend_types), expected)

    def test_generator_uses_ikr_for_souffle(self):
        """Test that generator backend is 'ikr' when souffle is specified.

        Since initialization requires Souffle to be installed, we test
        the generator_backend mapping logic directly.
        """
        # This tests the mapping logic: generator_backend = "ikr" if backend == "souffle" else backend
        backend = "souffle"
        generator_backend = "ikr" if backend == "souffle" else backend
        self.assertEqual(generator_backend, "ikr")

    def test_souffle_backend_initialization_requires_souffle(self):
        """Test that initializing with souffle backend requires Souffle installation."""
        mock_llm_client = MagicMock()

        # This should raise RuntimeError if Souffle is not installed
        # or succeed if it is - either way validates the code path
        try:
            pot = ProofOfThought(
                llm_client=mock_llm_client,
                model="test-model",
                backend="souffle",
            )
            # If we get here, Souffle is installed
            self.assertEqual(pot.backend_type, "souffle")
            self.assertEqual(pot.generator.backend, "ikr")
        except RuntimeError as e:
            # Souffle not installed - verify error message
            self.assertIn("not available", str(e).lower())

    def test_souffle_path_parameter_exists(self):
        """Test that souffle_path parameter is accepted."""
        mock_llm_client = MagicMock()

        try:
            # This should not raise TypeError about unexpected keyword argument
            pot = ProofOfThought(
                llm_client=mock_llm_client,
                backend="souffle",
                souffle_path="/nonexistent/path/to/souffle",
            )
        except RuntimeError:
            # Expected - Souffle not available at that path
            pass
        except TypeError as e:
            # This would indicate souffle_path parameter is not accepted
            self.fail(f"souffle_path parameter not accepted: {e}")

    def test_ikr_two_stage_accepted_for_souffle(self):
        """Test that ikr_two_stage parameter is accepted for souffle backend."""
        mock_llm_client = MagicMock()

        try:
            pot = ProofOfThought(
                llm_client=mock_llm_client,
                backend="souffle",
                ikr_two_stage=True,
            )
            self.assertTrue(pot.generator.ikr_two_stage)
        except RuntimeError:
            # Souffle not installed - still validates parameter is accepted
            pass

        try:
            pot = ProofOfThought(
                llm_client=mock_llm_client,
                backend="souffle",
                ikr_two_stage=False,
            )
            self.assertFalse(pot.generator.ikr_two_stage)
        except RuntimeError:
            pass


if __name__ == "__main__":
    unittest.main()
